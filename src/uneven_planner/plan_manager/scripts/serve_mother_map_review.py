#!/usr/bin/env python3
"""Serve a local, human-in-the-loop reviewer for LAS/LAZ mother maps.

The browser receives a manageable, spatially representative preview of the
source cloud. Manual and random 20 m crop candidates are evaluated by reading
the selected region from the original LAS/LAZ file, so the review coordinates
refer to the mother map rather than to a derived dataset. If the source has
RGB dimensions they are preserved; otherwise the browser exposes LAS class,
elevation, and intensity colouring as useful fallbacks.
"""

import argparse
import base64
import cgi
import datetime
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import laspy
import numpy as np

from review_slots import active_records, fill_slot, read_jsonl, resolve_slots


def workspace_root():
    return Path(__file__).resolve().parents[4]


def resolve_workspace_path(value, root):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return Path(os.path.realpath(str(path)))


def safe_component(value, label):
    value = str(value or "").strip()
    if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(
            f"{label} must contain only letters, numbers, '.', '_' or '-'")
    return value


def register_uploaded_maps(form, root):
    domain = safe_component(form.getfirst("domain", ""), "domain")
    site = safe_component(form.getfirst("site", "imported") or "imported", "site")
    fields = form["map_files"] if "map_files" in form else []
    if not isinstance(fields, list):
        fields = [fields]
    fields = [field for field in fields if field.filename]
    if not fields:
        raise ValueError("select at least one .las or .laz file")

    destination = (root / "dataset" / "external" / domain
                   / f"{domain}_{site}" / "raw")
    targets = []
    for field in fields:
        filename = Path(field.filename).name
        if Path(filename).suffix.lower() not in {".las", ".laz"}:
            raise ValueError(f"unsupported mother-map file: {filename}")
        target = destination / filename
        if target in [path for _, path in targets] or target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        targets.append((field, target))

    destination.mkdir(parents=True, exist_ok=True)
    for field, target in targets:
        with target.open("wb") as stream:
            shutil.copyfileobj(field.file, stream)
    return {
        "domain": domain,
        "site": site,
        "raw_dir": str(destination),
        "files": [str(target) for _, target in targets],
        "review_input": str(destination),
    }


def import_uploaded_review(form, root):
    domain = safe_component(form.getfirst("domain", ""), "domain")
    field = form["review_file"] if "review_file" in form else None
    if field is None or not field.filename:
        raise ValueError("select a JSONL review file")
    filename = Path(field.filename).name
    if Path(filename).suffix.lower() not in {".jsonl", ".ndjson"}:
        raise ValueError(f"unsupported review file: {filename}")
    raw = field.file.read()
    text = raw.decode("utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            json.loads(line)

    destination = root / "dataset" / "reviews" / domain / filename
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return {"domain": domain, "review_file": str(destination)}


def browse_directory(value, root):
    requested = resolve_workspace_path(value or root, root)
    path = requested
    while not path.exists() and path != path.parent:
        path = path.parent
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise FileNotFoundError(path)
    directories = []
    files = []
    for child in sorted(path.iterdir(), key=lambda item: item.name.casefold()):
        if child.is_dir():
            directories.append({"name": child.name, "path": str(child)})
        elif child.is_file():
            files.append({"name": child.name, "path": str(child)})
    return {
        "requested_path": str(requested),
        "path": str(path),
        "parent": None if path == path.parent else str(path.parent),
        "directories": directories,
        "files": files,
    }


def encode_array(array):
    return base64.b64encode(
        np.ascontiguousarray(array).tobytes()).decode("ascii")


def finite_xyz(points):
    xyz = np.column_stack((
        np.asarray(points.x),
        np.asarray(points.y),
        np.asarray(points.z),
    ))
    return xyz, np.isfinite(xyz).all(axis=1)


def normalize_rgb(values):
    values = np.asarray(values)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    values = values.astype(np.float64, copy=False)
    scale = 65535.0 if float(np.nanmax(values)) > 255.0 else 255.0
    return np.clip(np.rint(values / scale * 255.0), 0, 255).astype(np.uint8)


def class_counts(values):
    if values is None or len(values) == 0:
        return {}
    counts = np.bincount(np.asarray(values, dtype=np.uint8), minlength=256)
    return {str(i): int(value) for i, value in enumerate(counts) if value}


def counts_array_to_dict(values):
    values = np.asarray(values, dtype=np.int64)
    return {str(i): int(value) for i, value in enumerate(values) if value}


def packed_points(xyz, origin_xy, origin_z, classifications=None,
                  intensity=None, rgb=None):
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz):
        relative_xy = (xyz[:, :2] - np.asarray(origin_xy, dtype=np.float64))
        relative_z = xyz[:, 2] - float(origin_z)
    else:
        relative_xy = np.empty((0, 2), dtype=np.float64)
        relative_z = np.empty(0, dtype=np.float64)
    result = {
        "count": int(len(xyz)),
        "x": encode_array(relative_xy[:, 0].astype("<f4")),
        "y": encode_array(relative_xy[:, 1].astype("<f4")),
        "z": encode_array(relative_z.astype("<f4")),
    }
    if classifications is not None:
        result["classification"] = encode_array(
            np.asarray(classifications, dtype=np.uint8))
    if intensity is not None:
        result["intensity"] = encode_array(
            np.asarray(intensity, dtype="<u2"))
    if rgb is not None:
        result["rgb"] = encode_array(
            np.asarray(rgb, dtype=np.uint8).reshape(-1, 3))
    return result


def source_group_name(path):
    parent = path.parent
    return (parent.parent.name
            if parent.name.lower() == "raw"
            else parent.name)


def source_category_name(path):
    parts = path.parts
    if "external" in parts:
        external_index = len(parts) - 1 - parts[::-1].index("external")
        if external_index + 1 < len(parts):
            return parts[external_index + 1]
    return source_group_name(path)


class SourceData:
    def __init__(self, source_id, path, max_display_points):
        self.source_id = source_id
        self.path = path
        self.max_display_points = max_display_points
        self.summary = None
        self.preview = None

    def _header_summary(self):
        with laspy.open(self.path) as reader:
            dimensions = list(reader.header.point_format.dimension_names)
            return {
                "id": self.source_id,
                "name": self.path.name,
                "source_category": source_category_name(self.path),
                "source_group": source_group_name(self.path),
                "path": str(self.path),
                "point_count": int(reader.header.point_count),
                "bounds": {
                    "min": [float(v) for v in reader.header.mins],
                    "max": [float(v) for v in reader.header.maxs],
                },
                "dimensions": dimensions,
                "has_rgb": all(name in dimensions for name in ("red", "green", "blue")),
                "has_classification": "classification" in dimensions,
                "has_intensity": "intensity" in dimensions,
            }

    def get_summary(self):
        if self.summary is None:
            self.summary = self._header_summary()
        return dict(self.summary)

    def load_preview(self):
        if self.preview is not None:
            return self.preview

        summary = self.get_summary()
        total = max(1, int(summary["point_count"]))
        has_classification = summary["has_classification"]
        has_intensity = summary["has_intensity"]
        has_rgb = summary["has_rgb"]
        rng = np.random.default_rng(20260815)
        xyz_parts = []
        class_parts = []
        intensity_parts = []
        rgb_parts = []
        with laspy.open(self.path) as reader:
            for points in reader.chunk_iterator(2_000_000):
                xyz, finite = finite_xyz(points)
                indices = np.flatnonzero(finite)
                if not len(indices):
                    continue
                take = int(round(len(indices) / float(total)
                                 * self.max_display_points))
                take = max(1, min(len(indices), take))
                if take < len(indices):
                    indices = rng.choice(indices, size=take, replace=False)
                selected = xyz[indices]
                xyz_parts.append(selected)
                if has_classification:
                    class_parts.append(
                        np.asarray(points.classification, dtype=np.uint8)[indices])
                if has_intensity:
                    intensity_parts.append(
                        np.asarray(points.intensity, dtype=np.uint16)[indices])
                if has_rgb:
                    rgb_parts.append(normalize_rgb(np.column_stack((
                        np.asarray(points.red)[indices],
                        np.asarray(points.green)[indices],
                        np.asarray(points.blue)[indices],
                    ))))

        if not xyz_parts:
            raise RuntimeError(f"No finite XYZ points found in {self.path}")
        xyz = np.concatenate(xyz_parts)
        if len(xyz) > self.max_display_points:
            keep = rng.choice(len(xyz), self.max_display_points, replace=False)
            xyz = xyz[keep]
            class_values = (np.concatenate(class_parts)[keep]
                            if class_parts else None)
            intensity_values = (np.concatenate(intensity_parts)[keep]
                                if intensity_parts else None)
            rgb_values = (np.concatenate(rgb_parts)[keep]
                          if rgb_parts else None)
        else:
            class_values = np.concatenate(class_parts) if class_parts else None
            intensity_values = (
                np.concatenate(intensity_parts) if intensity_parts else None)
            rgb_values = np.concatenate(rgb_parts) if rgb_parts else None

        origin_xy = summary["bounds"]["min"][:2]
        origin_z = summary["bounds"]["min"][2]
        self.preview = {
            "summary": summary,
            "origin_xy": origin_xy,
            "origin_z": origin_z,
            "points": packed_points(
                xyz, origin_xy, origin_z, class_values,
                intensity_values, rgb_values),
            "class_counts": class_counts(class_values),
            "intensity_range": (
                [int(np.min(intensity_values)), int(np.max(intensity_values))]
                if intensity_values is not None and len(intensity_values)
                else None),
        }
        return self.preview

    @staticmethod
    def _rotated_coordinates(xyz, center_x, center_y, yaw):
        dx = xyz[:, 0] - center_x
        dy = xyz[:, 1] - center_y
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        return local_x, local_y

    def _local_crop_points(self, center_x, center_y, size, yaw_deg, margin,
                           include_classification=False):
        """Read exact and padded points in the candidate's local frame."""
        yaw = math.radians(yaw_deg)
        selection_half = 0.5 * size + margin
        source_extent = selection_half * (
            abs(math.cos(yaw)) + abs(math.sin(yaw)))
        exact_parts = []
        padded_parts = []
        exact_class_parts = []
        padded_class_parts = []
        with laspy.open(self.path) as reader:
            dimensions = set(reader.header.point_format.dimension_names)
            if include_classification and "classification" not in dimensions:
                raise ValueError("source has no LAS classification field")
            for points in reader.chunk_iterator(2_000_000):
                xyz, finite = finite_xyz(points)
                if not np.any(finite):
                    continue
                xyz = xyz[finite]
                nearby = (
                    (np.abs(xyz[:, 0] - center_x) < source_extent)
                    & (np.abs(xyz[:, 1] - center_y) < source_extent))
                if not np.any(nearby):
                    continue
                nearby_xyz = xyz[nearby]
                nearby_classes = None
                if include_classification:
                    nearby_classes = np.asarray(
                        points.classification, dtype=np.uint8)[finite][nearby]
                local_x, local_y = self._rotated_coordinates(
                    nearby_xyz, center_x, center_y, yaw)
                local = nearby_xyz.copy()
                local[:, 0] = local_x
                local[:, 1] = local_y
                padded = (
                    (np.abs(local_x) < selection_half)
                    & (np.abs(local_y) < selection_half))
                if np.any(padded):
                    padded_parts.append(local[padded])
                    if nearby_classes is not None:
                        padded_class_parts.append(nearby_classes[padded])
                exact = (
                    padded
                    & (np.abs(local_x) < 0.5 * size)
                    & (np.abs(local_y) < 0.5 * size))
                if np.any(exact):
                    exact_parts.append(local[exact])
                    if nearby_classes is not None:
                        exact_class_parts.append(nearby_classes[exact])

        empty = np.empty((0, 3), dtype=np.float64)
        exact = np.concatenate(exact_parts) if exact_parts else empty
        padded = np.concatenate(padded_parts) if padded_parts else empty
        if include_classification:
            empty_classes = np.empty(0, dtype=np.uint8)
            exact_classes = (
                np.concatenate(exact_class_parts)
                if exact_class_parts else empty_classes)
            padded_classes = (
                np.concatenate(padded_class_parts)
                if padded_class_parts else empty_classes)
            return exact, padded, exact_classes, padded_classes
        return exact, padded

    def fit(self, center_x, center_y, size, yaw_deg):
        """Fit the same YRF ground surface used by canonical generation."""
        from prepare_laz_terrain_map import (
            YRF_DEFAULT_COARSE_RESOLUTION,
            YRF_DEFAULT_ENVELOPE_OUTLIER,
            YRF_DEFAULT_GROUND_BAND_ABOVE,
            YRF_DEFAULT_GROUND_BAND_BELOW,
            YRF_DEFAULT_LOWER_ENVELOPE_FILTER_SIZE,
            YRF_DEFAULT_VOXEL_SIZE,
            fit_yrf_ground_grid,
        )
        from sample_laz_mother_map import grid_coverage

        fit_radius = 0.90
        exact, padded = self._local_crop_points(
            center_x, center_y, size, yaw_deg, fit_radius,
            include_classification=False)
        if len(exact) == 0 or len(padded) == 0:
            raise ValueError("candidate contains no points for fitting")

        resolution = 0.20
        (local_x, local_y, elevation, normals, valid, observed, neighbors,
         split_diagnostics) = fit_yrf_ground_grid(
            padded,
            size,
            resolution,
            coarse_resolution=YRF_DEFAULT_COARSE_RESOLUTION,
            voxel_size=YRF_DEFAULT_VOXEL_SIZE,
            ground_below=YRF_DEFAULT_GROUND_BAND_BELOW,
            ground_above=YRF_DEFAULT_GROUND_BAND_ABOVE,
            envelope_outlier=YRF_DEFAULT_ENVELOPE_OUTLIER,
            lower_envelope_filter_size=YRF_DEFAULT_LOWER_ENVELOPE_FILTER_SIZE,
        )
        valid = np.asarray(valid, dtype=bool)
        elevation = np.asarray(elevation, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        vertical_origin = float(np.min(elevation[valid]))
        elevation -= vertical_origin
        slope = np.degrees(np.arccos(
            np.clip(normals[:, :, 2], -1.0, 1.0)))
        valid_slope = slope[valid]
        grid = {
            "width": int(elevation.shape[1]),
            "height": int(elevation.shape[0]),
            "x": encode_array(
                np.asarray(local_x, dtype="<f4").ravel()),
            "y": encode_array(
                np.asarray(local_y, dtype="<f4").ravel()),
            "z": encode_array(elevation.astype("<f4").ravel()),
            "valid": encode_array(valid.astype("<u1").ravel()),
        }
        return {
            "source_id": self.source_id,
            "center_xy": [float(center_x), float(center_y)],
            "size_m": float(size),
            "yaw_deg": float(yaw_deg),
            "method": (
                "canonical YRF ground fit: 0.2m voxel centroids + cleaned "
                "0.4m lower envelope + 3x3 upper-median reference + cubic "
                "interpolation"),
            "vertical_origin_m": vertical_origin,
            "stats": {
                "raw_points_in_candidate": int(len(exact)),
                "raw_points_in_fit_margin": int(len(padded)),
                "fit_point_scope": (
                    "finite XYZ after 0.2m voxel centroids; YRF ground band"),
                "fit_points": int(split_diagnostics["ground_candidate_count"]),
                "fit_point_fraction": float(
                    split_diagnostics["ground_candidate_count"] / len(padded)),
                "classification_policy": (
                    "all LAS classes loaded; geometric low-envelope cleanup"),
                "surface_cell_size_m": YRF_DEFAULT_COARSE_RESOLUTION,
                "ground_band_below_m": YRF_DEFAULT_GROUND_BAND_BELOW,
                "ground_band_above_m": YRF_DEFAULT_GROUND_BAND_ABOVE,
                "envelope_outlier_m": YRF_DEFAULT_ENVELOPE_OUTLIER,
                "lower_envelope_filter_size": (
                    YRF_DEFAULT_LOWER_ENVELOPE_FILTER_SIZE),
                "split_diagnostics": split_diagnostics,
                "robust_refinement": "YRF cubic elevation interpolation",
                "raw_grid_coverage_1m": float(
                    grid_coverage(exact, size, 1.0)),
                "observed_fraction": float(np.mean(observed)),
                "valid_fraction": float(np.mean(valid)),
                "elevation_range_m": [
                    float(np.min(elevation)), float(np.max(elevation))],
                "slope_degrees": {
                    "median": float(np.median(valid_slope)),
                    "p95": float(np.quantile(valid_slope, 0.95)),
                    "maximum": float(np.max(valid_slope)),
                },
                "grid_shape": [int(elevation.shape[0]), int(elevation.shape[1])],
                "grid_resolution_m": resolution,
                "vertical_origin_m": vertical_origin,
            },
            "grid": grid,
        }

    def crop(self, center_x, center_y, size, yaw_deg, max_points):
        if size <= 0.0:
            raise ValueError("size must be positive")
        if size > 500.0:
            raise ValueError("size is unexpectedly large")
        yaw = math.radians(yaw_deg)
        selected_xyz = []
        selected_classes = []
        selected_intensity = []
        selected_rgb = []
        total_points = 0
        counts = np.zeros(256, dtype=np.int64)
        class_z_min = np.full(256, math.inf, dtype=np.float64)
        class_z_max = np.full(256, -math.inf, dtype=np.float64)
        z_min = math.inf
        z_max = -math.inf
        reference_z_parts = []
        occupied_cells = set()
        with laspy.open(self.path) as reader:
            dimensions = set(reader.header.point_format.dimension_names)
            has_classification = "classification" in dimensions
            has_intensity = "intensity" in dimensions
            has_rgb = all(name in dimensions for name in ("red", "green", "blue"))
            for points in reader.chunk_iterator(2_000_000):
                xyz, finite = finite_xyz(points)
                if not np.any(finite):
                    continue
                xyz = xyz[finite]
                local_x, local_y = self._rotated_coordinates(
                    xyz, center_x, center_y, yaw)
                inside = ((np.abs(local_x) <= size * 0.5)
                          & (np.abs(local_y) <= size * 0.5))
                if not np.any(inside):
                    continue
                crop_xyz = xyz[inside]
                total_points += len(crop_xyz)
                z_min = min(z_min, float(np.min(crop_xyz[:, 2])))
                z_max = max(z_max, float(np.max(crop_xyz[:, 2])))
                cells = max(1, int(math.ceil(size)))
                cell_x = np.clip(
                    np.floor(local_x[inside] + size * 0.5).astype(np.int64),
                    0, cells - 1)
                cell_y = np.clip(
                    np.floor(local_y[inside] + size * 0.5).astype(np.int64),
                    0, cells - 1)
                occupied_cells.update(zip(cell_x.tolist(), cell_y.tolist()))
                classes = None
                if has_classification:
                    classes = np.asarray(points.classification, dtype=np.uint8)[finite][inside]
                    counts += np.bincount(classes, minlength=256)
                    np.minimum.at(class_z_min, classes, crop_xyz[:, 2])
                    np.maximum.at(class_z_max, classes, crop_xyz[:, 2])
                else:
                    reference_z_parts.append(crop_xyz[:, 2])
                if total_points <= max_points:
                    selected_xyz.append(crop_xyz)
                    if has_classification:
                        selected_classes.append(classes)
                    if has_intensity:
                        selected_intensity.append(
                            np.asarray(points.intensity, dtype=np.uint16)[finite][inside])
                    if has_rgb:
                        selected_rgb.append(normalize_rgb(np.column_stack((
                            np.asarray(points.red)[finite][inside],
                            np.asarray(points.green)[finite][inside],
                            np.asarray(points.blue)[finite][inside],
                        ))))

        if selected_xyz:
            xyz = np.concatenate(selected_xyz)
            classes = np.concatenate(selected_classes) if selected_classes else None
            intensity = (np.concatenate(selected_intensity)
                         if selected_intensity else None)
            rgb = np.concatenate(selected_rgb) if selected_rgb else None
            if len(xyz) > max_points:
                rng = np.random.default_rng(20260815)
                keep = rng.choice(len(xyz), max_points, replace=False)
                xyz = xyz[keep]
                if classes is not None:
                    classes = classes[keep]
                if intensity is not None:
                    intensity = intensity[keep]
                if rgb is not None:
                    rgb = rgb[keep]
        else:
            xyz = np.empty((0, 3), dtype=np.float64)
            classes = np.empty(0, dtype=np.uint8) if has_classification else None
            intensity = np.empty(0, dtype=np.uint16) if has_intensity else None
            rgb = np.empty((0, 3), dtype=np.uint8) if has_rgb else None

        summary = self.get_summary()
        origin_xy = summary["bounds"]["min"][:2]
        origin_z = z_min if math.isfinite(z_min) else summary["bounds"]["min"][2]
        z_range_by_class = {
            str(index): {
                "z_min": float(class_z_min[index]),
                "z_max": float(class_z_max[index]),
                "z_range": float(class_z_max[index] - class_z_min[index]),
            }
            for index in range(256) if math.isfinite(class_z_min[index])
        }
        class_1_2_min = min(class_z_min[1], class_z_min[2])
        class_1_2_max = max(class_z_max[1], class_z_max[2])
        class_1_2_z_range = (
            {
                "z_min": float(class_1_2_min),
                "z_max": float(class_1_2_max),
                "z_range": float(class_1_2_max - class_1_2_min),
            }
            if math.isfinite(class_1_2_min) else None
        )
        reference_z_range = class_1_2_z_range
        reference_z_range_method = (
            "class1+class2" if class_1_2_z_range is not None else None)
        if reference_z_range is None and reference_z_parts:
            reference_values = np.concatenate(reference_z_parts)
            reference_min, reference_max = np.quantile(
                reference_values, [0.01, 0.99])
            reference_z_range = {
                "z_min": float(reference_min),
                "z_max": float(reference_max),
                "z_range": float(reference_max - reference_min),
            }
            reference_z_range_method = "z_1pct_to_99pct"
        return {
            "source_id": self.source_id,
            "center_xy": [float(center_x), float(center_y)],
            "size_m": float(size),
            "yaw_deg": float(yaw_deg),
            "points": packed_points(
                xyz, origin_xy, origin_z, classes, intensity, rgb),
            "origin_xy": origin_xy,
            "origin_z": float(origin_z),
            "stats": {
                "point_count": int(total_points),
                "display_point_count": int(len(xyz)),
                "class_counts": counts_array_to_dict(counts),
                "z_min": None if not math.isfinite(z_min) else z_min,
                "z_max": None if not math.isfinite(z_max) else z_max,
                "z_range": None if not math.isfinite(z_min) else z_max - z_min,
                "z_range_by_class": z_range_by_class,
                "class_1_plus_2_z_range": class_1_2_z_range,
                "reference_z_range": reference_z_range,
                "reference_z_range_method": reference_z_range_method,
                "occupied_1m_cells": int(len(occupied_cells)),
                "occupied_1m_fraction": float(
                    len(occupied_cells) / (max(1, int(math.ceil(size))) ** 2)),
            },
        }


class ReviewStore:
    def __init__(self, path):
        self.path = path.resolve()
        self.replacement_path = self.path.parent / "canonical_replacements.jsonl"
        self.lock = threading.RLock()
        self.records = []
        self._mtime_ns = None
        self.reload()

    @staticmethod
    def _read(path):
        records = []
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    def reload(self):
        mtime_ns = (self.path.stat().st_mtime_ns
                    if self.path.is_file() else None)
        records = self._read(self.path)
        with self.lock:
            self.records = records
            self._mtime_ns = mtime_ns

    def reload_if_changed(self):
        mtime_ns = (self.path.stat().st_mtime_ns
                    if self.path.is_file() else None)
        with self.lock:
            if mtime_ns == self._mtime_ns:
                return
            self.records = self._read(self.path)
            self._mtime_ns = mtime_ns

    def switch_path(self, path):
        path = path.resolve()
        records = self._read(path)
        with self.lock:
            self.path = path
            self.replacement_path = path.parent / "canonical_replacements.jsonl"
            self.records = records
            self._mtime_ns = (path.stat().st_mtime_ns
                              if path.is_file() else None)

    def slot_state(self):
        self.reload_if_changed()
        return resolve_slots(self.path, self.replacement_path)

    def open_slots(self):
        return self.slot_state()["open_slots"]

    def counts(self, source_id=None):
        state = self.slot_state()
        approve = sum(
            1 for _, _, record in active_records(state)
            if source_id is None or record.get("source_id") == source_id)
        reject = sum(
            1 for record in self.records
            if record.get("decision") == "reject"
            and (source_id is None or record.get("source_id") == source_id))
        return {"approve": approve, "reject": reject}

    def split_counts(self, source_id=None):
        result = {"train": 0, "val": 0, "unassigned": 0}
        for _, _, record in active_records(self.slot_state()):
            if source_id is not None and record.get("source_id") != source_id:
                continue
            split = record.get("split")
            result[split if split in {"train", "val"} else "unassigned"] += 1
        return result

    def records_for(self, source_id):
        with self.lock:
            state = self.slot_state()
            baseline_by_line = {
                entry["line_number"]: (key, entry)
                for key, entry in state["baseline"].items()
            }
            result = []
            for line_number, record in read_jsonl(self.path):
                if record.get("source_id") != source_id:
                    continue
                updated = dict(record)
                baseline = baseline_by_line.get(line_number)
                if baseline is not None:
                    key, _ = baseline
                    slot = state["slots"].get(key)
                    if slot and slot.get("status") == "pending":
                        updated["slot_status"] = "open"
                        updated["slot_key"] = (
                            f"{key[0]}/map_{key[1]:03d}")
                    elif slot and slot.get("filled"):
                        updated["slot_status"] = "replaced"
                        updated["slot_key"] = (
                            f"{key[0]}/map_{key[1]:03d}")
                result.append(updated)
            return result

    def append(self, record):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.records.append(record)
            self._mtime_ns = self.path.stat().st_mtime_ns
            return self.counts(record.get("source_id"))

    def update_split(self, source_id, timestamp_utc, split):
        result = self.update_splits(source_id, [timestamp_utc], split)
        result["record"] = result["updated_records"][0]
        return result

    def update_splits(self, source_id, timestamps_utc, split):
        with self.lock:
            timestamp_set = set(timestamps_utc)
            if not timestamp_set:
                raise ValueError("at least one review record is required")
            matches = [
                index for index, record in enumerate(self.records)
                if (record.get("source_id") == source_id
                    and record.get("timestamp_utc") in timestamp_set)
            ]
            matched_timestamps = {
                self.records[index].get("timestamp_utc") for index in matches
            }
            if matched_timestamps != timestamp_set:
                raise FileNotFoundError(
                    "one or more review records were not found; reload the review file")
            if len(matches) != len(timestamp_set):
                raise ValueError("review record timestamp is not unique")
            records = list(self.records)
            updated_records = []
            for index in matches:
                updated = dict(records[index])
                updated["split"] = split
                records[index] = updated
                updated_records.append(updated)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records)
            self.path.write_text(payload, encoding="utf-8")
            self.records = records
            self._mtime_ns = self.path.stat().st_mtime_ns
            return {
                "updated_count": len(updated_records),
                "updated_records": [dict(record) for record in updated_records],
                "review_counts": self.counts(source_id),
                "review_split_counts": self.split_counts(source_id),
                "review_records": [
                    dict(record) for record in records
                    if record.get("source_id") == source_id
                ],
            }


class Catalog:
    def __init__(self, input_paths, max_display_points, max_crop_points,
                 review_store):
        self.input_paths = []
        self.sources = {}
        self.max_display_points = max_display_points
        self.max_crop_points = max_crop_points
        self.review_store = review_store
        for input_path in input_paths:
            self.add_input_path(input_path)
        if input_paths and not self.sources:
            paths = ", ".join(str(path) for path in self.input_paths)
            raise FileNotFoundError(f"No LAS/LAZ files found under {paths}")

    @staticmethod
    def _files_for(path):
        if path.is_file():
            return [path] if path.suffix.lower() in {".las", ".laz"} else []
        return sorted(set(path.rglob("*.laz")) | set(path.rglob("*.las")))

    @staticmethod
    def _source_map(paths, max_display_points):
        sources = {}
        seen_paths = set()
        for input_path in paths:
            for path in Catalog._files_for(input_path):
                path = path.resolve()
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                source_id = path.name
                if source_id in sources:
                    source_id = f"{path.parent.name}/{path.name}"
                    suffix = 2
                    while source_id in sources:
                        source_id = f"{path.parent.name}/{suffix}_{path.name}"
                        suffix += 1
                sources[source_id] = SourceData(
                    source_id, path, max_display_points)
        return sources

    @staticmethod
    def _contains_path(parent, child):
        return parent == child or (parent.is_dir() and parent in child.parents)

    def add_input_path(self, path):
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        files = self._files_for(path)
        if not files:
            raise FileNotFoundError(f"No LAS/LAZ files found under {path}")
        if not any(self._contains_path(existing, path)
                   for existing in self.input_paths):
            self.input_paths = [
                existing for existing in self.input_paths
                if not self._contains_path(path, existing)]
            self.input_paths.append(path)
        self.sources = self._source_map(
            self.input_paths, self.max_display_points)
        return [source_id for source_id, source in self.sources.items()
                if source.path in files]

    def reload(self):
        self.sources = self._source_map(
            self.input_paths, self.max_display_points)

    def source(self, source_id):
        try:
            return self.sources[source_id]
        except KeyError:
            raise FileNotFoundError(f"Unknown source: {source_id}")

    def index(self):
        sources = []
        open_slots = self.review_store.open_slots()
        for source_id in sorted(self.sources):
            summary = self.sources[source_id].get_summary()
            summary["review_counts"] = self.review_store.counts(source_id)
            summary["review_split_counts"] = self.review_store.split_counts(source_id)
            sources.append(summary)
        return {
            "roots": [str(path) for path in self.input_paths],
            "review_file": str(self.review_store.path),
            "open_slots": open_slots,
            "sources": sources,
        }

    def source_record(self, source_id):
        source = self.source(source_id)
        preview = source.load_preview()
        record = dict(preview)
        record["review_counts"] = self.review_store.counts(source_id)
        record["review_split_counts"] = self.review_store.split_counts(source_id)
        record["review_records"] = self.review_store.records_for(source_id)
        record["open_slots"] = self.review_store.open_slots()
        return record

    def random_candidate(self, source_id, size, random_yaw):
        source = self.source(source_id)
        preview = source.load_preview()
        bounds = preview["summary"]["bounds"]
        point_data = preview["points"]
        count = point_data["count"]
        raw_x = np.frombuffer(base64.b64decode(point_data["x"]), dtype="<f4")
        raw_y = np.frombuffer(base64.b64decode(point_data["y"]), dtype="<f4")
        x = raw_x + float(preview["origin_xy"][0])
        y = raw_y + float(preview["origin_xy"][1])
        yaw = float(np.random.default_rng().uniform(-180.0, 180.0)
                    if random_yaw else 0.0)
        margin_x = 0.5 * size * (
            abs(math.cos(math.radians(yaw)))
            + abs(math.sin(math.radians(yaw))))
        min_x, min_y = bounds["min"][:2]
        max_x, max_y = bounds["max"][:2]
        eligible = ((x >= min_x + margin_x) & (x <= max_x - margin_x)
                    & (y >= min_y + margin_x) & (y <= max_y - margin_x))
        if np.any(eligible):
            indices = np.flatnonzero(eligible)
            index = int(indices[int(np.random.default_rng().integers(len(indices)))])
            center = [float(x[index]), float(y[index])]
        else:
            center = [float((min_x + max_x) * 0.5),
                       float((min_y + max_y) * 0.5)]
        return {"source_id": source_id, "center_xy": center,
                "size_m": float(size), "yaw_deg": yaw,
                "preview_point_count": count}


class JobManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}

    @staticmethod
    def _tail(path, limit=6000):
        if not path.is_file():
            return ""
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read().decode("utf-8", errors="replace")

    def _snapshot(self, job):
        process = job["process"]
        return_code = process.poll()
        if return_code is not None and job["returncode"] is None:
            job["returncode"] = int(return_code)
            job["finished_at"] = datetime.datetime.now(
                datetime.timezone.utc).isoformat()
        return {
            "id": job["id"],
            "kind": job["kind"],
            "pid": int(process.pid),
            "running": return_code is None,
            "returncode": job["returncode"],
            "started_at": job["started_at"],
            "finished_at": job.get("finished_at"),
            "log_path": str(job["log_path"]),
            "tail": self._tail(job["log_path"]),
        }

    def snapshots(self):
        with self.lock:
            return [self._snapshot(job) for job in self.jobs.values()]

    def start(self, kind, command, cwd, log_path):
        with self.lock:
            for job in self.jobs.values():
                if job["process"].poll() is None:
                    raise RuntimeError("another dataset job is already running")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    [str(value) for value in command],
                    cwd=str(cwd),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            job_id = f"{kind}-{uuid.uuid4().hex[:10]}"
            self.jobs[job_id] = {
                "id": job_id,
                "kind": kind,
                "process": process,
                "log_path": log_path,
                "started_at": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
                "returncode": None,
            }
            return self._snapshot(self.jobs[job_id])

    def stop(self):
        with self.lock:
            running = [job for job in self.jobs.values()
                       if job["process"].poll() is None]
            if not running:
                return None
            job = running[-1]
            try:
                os.killpg(job["process"].pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            return self._snapshot(job)


class ViewerManager:
    def __init__(self, viewer_url):
        self.lock = threading.Lock()
        self.viewer_url = viewer_url
        self.job = None

    @staticmethod
    def _tail(path, limit=6000):
        if not path.is_file():
            return ""
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read().decode("utf-8", errors="replace")

    def _snapshot(self):
        if self.job is None:
            return {
                "kind": "dataset_viewer",
                "url": self.viewer_url,
                "running": False,
                "returncode": None,
                "dataset_root": None,
                "log_path": None,
                "tail": "",
            }
        process = self.job["process"]
        return_code = process.poll()
        if return_code is not None and self.job["returncode"] is None:
            self.job["returncode"] = int(return_code)
            self.job["finished_at"] = datetime.datetime.now(
                datetime.timezone.utc).isoformat()
        return {
            "kind": "dataset_viewer",
            "url": self.viewer_url,
            "running": return_code is None,
            "returncode": self.job["returncode"],
            "dataset_root": self.job["dataset_root"],
            "log_path": str(self.job["log_path"]),
            "started_at": self.job["started_at"],
            "finished_at": self.job.get("finished_at"),
            "tail": self._tail(self.job["log_path"]),
        }

    def status(self):
        with self.lock:
            return self._snapshot()

    def start(self, command, cwd, dataset_root, log_path):
        with self.lock:
            if self.job is not None and self.job["process"].poll() is None:
                raise RuntimeError("dataset viewer is already running")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    [str(value) for value in command],
                    cwd=str(cwd),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            self.job = {
                "process": process,
                "dataset_root": str(dataset_root),
                "log_path": log_path,
                "started_at": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
                "returncode": None,
            }
            return self._snapshot()

    def stop(self):
        with self.lock:
            if self.job is None or self.job["process"].poll() is not None:
                return self._snapshot()
            try:
                os.killpg(self.job["process"].pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            return self._snapshot()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_paths", type=Path, nargs="*",
        help="Optional LAS/LAZ files or directories containing them")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--title", default="原始点云地图人工筛选")
    parser.add_argument("--domain", default="terrain")
    parser.add_argument(
        "--dataset-viewer-url", default="http://127.0.0.1:8765/",
        help="URL used to start and open the dataset viewer")
    parser.add_argument(
        "--review-file", type=Path,
        default=Path("dataset/reviews/mother_map_reviews.jsonl"),
        help="JSONL file receiving approve/reject decisions")
    parser.add_argument("--max-display-points", type=int, default=180000)
    parser.add_argument("--max-crop-points", type=int, default=120000)
    return parser.parse_args()


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#0f1417;--panel:#182126;--panel2:#222d33;--line:#34424a;--text:#e9f0f3;--muted:#9caeb8;--accent:#61c4e0;--good:#4fc38a;--bad:#ef7668;--warn:#e8b85e}
.crop-wrap canvas[hidden],.empty[hidden]{display:none!important}
.app{height:100vh;display:grid;grid-template-columns:330px minmax(400px,1fr) 360px;grid-template-rows:auto 1fr}.app.audit-collapsed{grid-template-columns:330px minmax(400px,1fr)}.right.hidden,.app.audit-collapsed .right{display:none}header{grid-column:1/-1;display:flex;align-items:center;gap:12px;padding:10px 14px;background:var(--panel);border-bottom:1px solid var(--line)}header h1{margin:0;font-size:17px;white-space:nowrap}.header-source{display:grid;grid-template-columns:minmax(110px,.7fr) minmax(260px,1.8fr);gap:8px;min-width:420px;max-width:54vw}.header-source label{display:grid;gap:3px;min-width:0;color:var(--muted);font-size:11px}.header-source select{width:100%;min-width:0;color:var(--text);font-size:13px}.header-spacer{flex:1}.status{color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:32vw}.left,.right{min-height:0;background:var(--panel);overflow:auto}.left{border-right:1px solid var(--line);padding:12px}.right{border-left:1px solid var(--line);padding:12px}.left-tabs{display:flex;gap:5px;border-bottom:1px solid var(--line);margin:-2px 0 12px;padding-bottom:7px}.left-tab{flex:1;padding:7px 8px;background:transparent;border:1px solid transparent;color:var(--muted)}.left-tab[aria-selected=true]{background:var(--panel2);border-color:var(--accent);color:var(--text)}.left-tab-panel[hidden]{display:none!important}.section{border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:12px}.section h2{font-size:14px;margin:0 0 9px}.control{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:7px 0}.control.full{grid-template-columns:1fr}.control label{display:flex;align-items:center;gap:6px;color:var(--muted);min-width:0}.control input,.control select{min-width:0;width:0;flex:1}.control input[type=number]{width:0}.control select{width:0}.hint{color:var(--muted);font-size:12px}.counts{display:flex;gap:8px;margin-top:9px}.count{border:1px solid var(--line);border-radius:7px;padding:6px 8px;flex:1}.count .k{font-size:11px;color:var(--muted)}.count .v{font-size:18px;font-weight:700}.map-area{position:relative;min-width:0;min-height:0;background:#0a0f12}.map-wrap{position:absolute;inset:0;padding:10px}.map-wrap canvas{display:block;width:100%;height:100%;background:#0a0f12;border:1px solid #29363d;touch-action:none;cursor:crosshair}.map-hint{position:absolute;left:20px;bottom:18px;color:#a9bbc4;background:#0b1013dd;border:1px solid var(--line);padding:6px 9px;border-radius:6px;pointer-events:none}.toolbar-row{display:flex;gap:7px;flex-wrap:wrap}.crop-title{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}.crop-title h2{margin:0;font-size:14px}.crop-wrap{height:300px;border:1px solid #29363d;background:#0a0f12}.crop-wrap canvas{display:block;width:100%;height:100%}.empty{height:100%;display:grid;place-items:center;color:var(--muted);padding:20px;text-align:center}.stats{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:10px}.stat{background:var(--panel2);border:1px solid var(--line);border-radius:7px;padding:7px}.stat .k{font-size:11px;color:var(--muted)}.stat .v{font-size:15px;font-weight:650;margin-top:2px}.class-table{width:100%;border-collapse:collapse;margin-top:8px}.class-table th{padding:4px 2px;color:var(--muted);font-size:11px;text-align:left;border-bottom:1px solid var(--line)}.class-table td{padding:4px 2px;border-bottom:1px solid #2c383e;font-size:12px}.class-table th:not(:first-child),.class-table td:not(:first-child){text-align:right}.note{width:100%;min-height:68px;resize:vertical}.review-update{display:grid;gap:6px;margin-top:8px}.review-update .hint{line-height:1.35}.review-actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}.review-actions button:last-child{grid-column:1/-1}.source-meta{font-size:12px;color:var(--muted);overflow-wrap:anywhere}.legend{font-size:12px;color:var(--muted);margin-top:8px}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}.legend span{margin-right:9px;white-space:nowrap}@media(max-width:1100px){.app{grid-template-columns:300px minmax(340px,1fr)}.app.audit-collapsed{grid-template-columns:300px minmax(340px,1fr)}.header-source{min-width:300px;max-width:52vw;grid-template-columns:minmax(80px,.7fr) minmax(170px,1.3fr)}.right{position:absolute;right:0;top:53px;bottom:0;width:360px;box-shadow:-8px 0 24px #0009;z-index:3}.right.hidden{display:none}}
.crop-3d-wrap{height:330px;position:relative;margin-top:10px}.crop-3d-wrap>div[hidden],.three-hint[hidden]{display:none!important}#crop3d{height:100%;width:100%;touch-action:none}#crop3d canvas{display:block;width:100%;height:100%}.three-hint{position:absolute;left:8px;bottom:7px;color:#b5c8d0;background:#0b1013cc;border:1px solid var(--line);padding:4px 7px;border-radius:5px;pointer-events:none;font-size:11px}.fit-actions{display:flex;align-items:center;gap:8px;margin-top:9px}.fit-actions button{flex:1}.fit-summary{margin-top:9px;border:1px solid #376178;background:#142833;padding:8px;border-radius:7px;font-size:12px}.fit-summary .muted{color:var(--muted)}.workflow-actions{display:flex;gap:7px;flex-wrap:wrap}.workflow-actions button,.workflow-actions a{flex:1;min-width:110px;text-align:center}.generation-action{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:7px}.generation-action button{flex:1;min-width:180px}.generation-action label{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:12px}.workflow-path{font-size:11px;color:var(--muted);overflow-wrap:anywhere;margin-top:6px}.job-box{margin-top:8px;max-height:180px;overflow:auto;white-space:pre-wrap;background:#0b1013;border:1px solid var(--line);padding:7px;border-radius:6px;font:11px/1.4 ui-monospace,SFMono-Regular,monospace}.job-actions{display:flex;gap:7px;margin-top:7px}.job-actions button{flex:1}.file-input{display:none}.path-control{display:block}.path-label{display:block;color:var(--muted);margin-bottom:5px}.path-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:7px}.path-row input{width:100%;min-width:0}.picker-backdrop{position:fixed;inset:0;z-index:20;display:grid;place-items:center;background:#000b;padding:20px}.picker-backdrop[hidden]{display:none!important}.picker{width:min(760px,calc(100vw - 40px));max-height:min(680px,calc(100vh - 40px));display:grid;grid-template-rows:auto auto minmax(180px,1fr) auto;gap:10px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;box-shadow:0 16px 50px #000b}.picker-head,.picker-bar,.picker-foot{display:flex;align-items:center;gap:8px}.picker-head h2{margin:0;font-size:15px;flex:1}.picker-bar input{min-width:0;flex:1}.picker-list{min-height:180px;overflow:auto;display:grid;align-content:start;gap:6px;padding:2px}.picker-item{width:100%;text-align:left}.picker-item.file{border-color:#496d78}.picker-foot{border-top:1px solid var(--line);padding-top:9px}.picker-foot .workflow-path{flex:1;margin:0}
</style>
<style>
.workflow-copy{color:var(--muted);font-size:12px;margin:0 0 8px}.advanced{margin-top:9px;border-top:1px solid var(--line);padding-top:8px}.advanced summary{color:var(--muted);cursor:pointer;font-size:12px}.advanced[open] summary{color:var(--text)}
.review-actions button:last-child{grid-column:auto}
.history-list{display:grid;gap:5px;margin-top:8px;max-height:260px;overflow:auto}
.review-bulk{display:grid;gap:7px;margin-top:8px;padding:8px;background:var(--panel2);border:1px solid var(--line);border-radius:7px;font-size:12px}
.review-bulk-row{display:flex;align-items:center;gap:7px;min-width:0}
.review-bulk-row label{display:flex;align-items:center;gap:6px;min-width:0;flex:1;color:var(--muted)}
.review-bulk-row select{min-width:0;flex:1}
.review-bulk-row button{flex:1}
.slot-box{display:grid;gap:5px;margin-top:8px;padding:8px;background:#25343a;border:1px solid #496671;border-radius:7px;font-size:12px}.slot-box strong{color:#b9e7f0}.slot-box select{width:100%}
.history-item{display:grid;grid-template-columns:20px minmax(0,1fr);gap:7px;align-items:center;text-align:left;padding:6px 7px;font-size:12px;border:1px solid var(--line);border-radius:6px}
.history-select{margin:0;justify-self:center}
.history-select-spacer{width:16px;height:16px;display:block}
.history-open{width:100%;min-width:0;display:grid;grid-template-columns:auto minmax(0,1fr);gap:7px;text-align:left;padding:0;background:transparent;border:0;color:var(--text)}
.history-item.approve{border-color:#3d9d76}
.history-item.reject{border-color:#ac554d}
.history-item.selected{outline:2px solid var(--accent);outline-offset:1px}
.history-open .history-mark{font-weight:700}
.history-open .history-detail{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.history-empty{color:var(--muted);font-size:12px;margin-top:8px}
</style>
<style>
.domain-control>label{display:grid;gap:5px}
.domain-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:7px}
.domain-row input{width:100%;box-sizing:border-box}
.generation-counts{grid-template-columns:1fr}
.generation-counts label{display:grid;grid-template-columns:minmax(0,1fr) 104px;gap:8px}
.generation-counts input{width:100%!important;box-sizing:border-box}
.viewer-actions{display:grid;grid-template-columns:1fr 1fr}
.viewer-actions #stopDatasetViewer{grid-column:1/-1}
.viewer-status{margin-top:6px}
</style>
<style>
.app{--left-width:330px;--right-width:360px;--pane-splitter-width:8px;grid-template-columns:var(--left-width) var(--pane-splitter-width) minmax(400px,1fr) var(--pane-splitter-width) var(--right-width)}
.app.audit-collapsed{grid-template-columns:var(--left-width) var(--pane-splitter-width) minmax(400px,1fr)}
.pane-splitter{position:relative;z-index:4;background:#10181c;cursor:col-resize;touch-action:none}
.pane-splitter::after{content:"";position:absolute;top:40%;bottom:40%;left:3px;width:2px;background:#52636c;border-radius:2px}
.pane-splitter:hover,.pane-splitter:focus{background:#1b3943;outline:none}
.pane-splitter:hover::after,.pane-splitter:focus::after{background:var(--accent)}
.app.audit-collapsed .right-splitter{display:none}
@media(max-width:1100px){
  .app,.app.audit-collapsed{grid-template-columns:var(--left-width) var(--pane-splitter-width) minmax(340px,1fr)}
  .right-splitter{position:fixed;display:block;right:var(--right-width);top:53px;bottom:0;width:var(--pane-splitter-width)}
  .app.audit-collapsed .right-splitter{display:none}
  .right{width:var(--right-width)}
}
</style>
</head>
<body>
<div class="app" id="app">
<header><h1>__TITLE__</h1><div class="header-source"><label>地形目录<select id="sourceCategory" disabled></select></label><label>具体原始地图<select id="source" disabled></select></label></div><div class="header-spacer"></div><div class="status" id="status">正在读取原始地图…</div><button id="toggleRight">打开审核面板</button></header>
<aside class="left">
  <div class="left-tabs" role="tablist" aria-label="工作区"><button class="left-tab" data-tab="map" role="tab" aria-selected="true">地图</button><button class="left-tab" data-tab="review" role="tab" aria-selected="false">审核</button><button class="left-tab" data-tab="generation" role="tab" aria-selected="false">生成</button></div>
  <div class="left-tab-panel" data-panel="map" role="tabpanel">
    <div class="section"><h2>选择原始地图</h2>
      <div class="workflow-copy">你下载的 LAS/LAZ 点云就是原始地图。选择电脑上的文件会自动复制到项目的地图库；已经放在项目里的地图可以直接选择文件夹。</div>
      <div class="control full domain-control"><label>本次数据名称<div class="domain-row"><input id="domain" value="terrain" spellcheck="false"><button id="confirmDomain" type="button">确认</button></div></label></div>
      <div class="hint">确认后会同步默认输出路径，并切换到 dataset/reviews/&lt;名称&gt;/manual_regions.jsonl；你手动改过或浏览选择过的路径会保留。</div>
      <input id="site" type="hidden" value="imported">
      <div class="workflow-actions"><button id="importMap">从电脑选择地图文件</button><button id="loadRoot">选择已有地图文件夹</button><button id="refreshCatalog">重新读取地图</button></div>
      <input id="mapFiles" class="file-input" type="file" accept=".las,.laz" multiple>
      <input id="loadPath" class="file-input" spellcheck="false">
      <div id="workflowMeta" class="workflow-path"></div>
    </div>
    <div class="section"><h2>候选框</h2>
      <div class="control"><label>边长(m)<input id="size" type="number" min="1" max="500" step="1" value="20"></label><label>旋转(°)<input id="yaw" type="number" step="1" value="0"></label></div>
      <div class="control full"><label><input id="randomYaw" type="checkbox">随机候选同时随机旋转</label></div>
      <div class="toolbar-row"><button id="random">随机框选</button><button id="clear">清除框</button><button id="fit">适应母图</button></div>
      <p class="hint">默认 20m 方框。直接在母图上拖动，松开后以拖动区域中心放置方框；Shift+拖动用于平移，滚轮缩放。</p>
    </div>
    <div class="section"><h2>显示</h2>
      <div class="control"><label>着色<select id="colorMode"><option value="auto">自动</option><option value="classification">LAS 分类</option><option value="rgb">原始 RGB</option><option value="elevation">高程</option><option value="intensity">强度</option></select></label><label>点大小<input id="pointSize" type="number" min="1" max="5" step="1" value="2"></label></div>
      <div class="legend" id="legend"></div>
    </div>
    <div class="section"><h2>当前原始地图</h2><div id="sourceMeta" class="source-meta"></div><div class="counts"><div class="count"><div class="k">已通过</div><div class="v" id="approved">0</div></div><div class="count"><div class="k">已拒绝</div><div class="v" id="rejected">0</div></div></div></div>
  </div>
  <div class="left-tab-panel" data-panel="review" role="tabpanel" hidden>
    <div class="section"><h2>人工筛选并保存</h2>
      <div class="workflow-copy">在地图上框选区域，结合二维图、三维点云和拟合结果判断是否通过。点击“通过”或“拒绝”后，工具会自动保存你的选择，通常不需要填写保存位置。</div>
      <input id="reviewFile" class="file-input" type="file" accept=".jsonl,.ndjson">
      <div class="workflow-actions"><a id="downloadReviews" download="manual_regions.jsonl">下载标注备份</a></div>
      <details class="advanced"><summary>恢复以前的标注</summary>
        <div class="workflow-copy">标注记录只保存“通过/拒绝、位置和备注”，不是地图文件。换电脑或换一轮任务时，才需要在这里加载以前的记录。</div>
        <div class="control full path-control"><span class="path-label">标注记录文件</span><div class="path-row"><input id="reviewPath" spellcheck="false"><button id="browseReviewPath">浏览</button></div></div>
        <div class="workflow-actions"><button id="chooseReview">从电脑导入备份</button><button id="switchReview">加载这个记录</button></div>
      </details>
    </div>
    <div class="section"><h2>历史审核</h2><div class="control full"><label><input id="showAllReviews" type="checkbox">显示全部审核选区（含拒绝）</label></div><div id="openSlots" class="slot-box"></div><div class="review-bulk"><div class="review-bulk-row"><label><input id="selectAllReviews" type="checkbox">全选当前可见的通过记录</label><span id="selectedReviewCount" class="hint">已选 0 条</span></div><div class="review-bulk-row"><label>批量设置为<select id="bulkSplit"><option value="train">训练集</option><option value="val">验证集</option></select></label><button id="bulkUpdateReview" disabled>批量更新归属</button></div></div><div id="reviewHistory" class="history-list"></div></div>
  </div>
  <div class="left-tab-panel" data-panel="generation" role="tabpanel" hidden>
    <div class="section"><h2>生成训练用地图</h2>
      <div class="workflow-copy">把你标记为“通过”的区域制作成训练程序能使用的地图文件。这是自动中间步骤，点击按钮即可，下一步会自动使用生成结果。</div>
      <div id="canonicalInfo" class="workflow-path">等待人工筛选结果</div>
      <div class="generation-action"><button id="exportCanonical">生成训练用地图</button><label><input id="overwriteCanonical" type="checkbox">覆盖已有训练用地图</label></div>
      <div class="hint">勾选后会清空上面保存位置中的旧中间地图，再按当前审核记录重新生成；不勾选时已有文件会被保留并拒绝覆盖。</div>
      <details class="advanced"><summary>更改训练地图的保存位置</summary>
        <div class="workflow-copy">通常不需要修改。这里保存的是中间地图，不是最终训练数据。</div>
        <div class="control full path-control"><span class="path-label">训练地图文件夹</span><div class="path-row"><input id="canonicalDir" spellcheck="false"><button id="browseCanonicalDir">浏览</button></div></div>
        <div class="control full path-control"><span class="path-label">程序清单（自动生成）</span><div class="path-row"><input id="manifestPath" spellcheck="false"><button id="browseManifest">浏览</button></div></div>
      </details>
    </div>
    <div class="section"><h2>生成最终训练数据</h2>
      <div class="workflow-copy">使用上一步的训练用地图，生成最终的训练集、验证集和轨迹。第一次使用时直接点击按钮即可。</div>
      <div id="datasetInfo" class="workflow-path">请先完成上一步</div>
      <button id="generateDataset">生成最终训练数据</button>
      <details class="advanced"><summary>调整生成数量或保存位置</summary>
        <div class="control full path-control"><span class="path-label">最终数据文件夹</span><div class="path-row"><input id="datasetDir" spellcheck="false"><button id="browseDatasetDir">浏览</button></div></div>
        <div class="control full generation-counts"><label>训练集轨迹数/每张地图<input id="trainPaths" type="number" min="1" step="1" value="100"></label><label>验证集轨迹数/每张地图<input id="valPaths" type="number" min="1" step="1" value="10"></label></div>
        <div class="control"><label>后台并行数<input id="workers" type="number" min="1" step="1" value="4"></label><label>ROS 端口<input id="rosPort" type="number" min="1" step="1" value="11412"></label></div>
        <input id="trainSiteId" type="hidden" value="terrain_train"><input id="valSiteId" type="hidden" value="terrain_val">
        <select id="trainProfile" hidden><option value="als" selected>ALS</option><option value="uls">ULS</option></select><select id="valProfile" hidden><option value="als" selected>ALS</option><option value="uls">ULS</option></select>
      </details>
      <div class="job-actions"><button id="stopJob" class="bad" disabled>停止当前生成</button></div>
      <div class="job-actions viewer-actions"><button id="startDatasetViewer">启动结果查看器</button><button id="openDatasetViewer" disabled>打开结果查看器</button><button id="stopDatasetViewer" class="bad" disabled>关闭结果查看器</button><a id="datasetViewer" hidden></a></div>
      <div id="viewerStatus" class="workflow-path viewer-status">结果查看器未启动</div>
      <div class="workflow-path">最近一次生成状态</div><div id="jobStatus" class="job-box">当前没有运行任务</div>
    </div>
  </div>
</aside>
<div class="pane-splitter left-splitter" id="leftSplitter" role="separator" aria-label="调整左侧栏宽度" aria-orientation="vertical" tabindex="0" title="拖动调整左侧栏宽度"></div>
<main class="map-area"><div class="map-wrap"><canvas id="map"></canvas><div class="map-hint">拖动框选候选区域 · Shift+拖动平移 · 滚轮缩放</div></div></main>
<div class="pane-splitter right-splitter" id="rightSplitter" role="separator" aria-label="调整右侧栏宽度" aria-orientation="vertical" tabindex="0" title="拖动调整右侧栏宽度"></div>
<aside class="right" id="right"><div class="crop-title"><h2>候选区域详情</h2><span id="decisionStatus" class="hint">未选择</span></div><div class="crop-wrap" id="cropWrap"><div class="empty">在原始地图上框选或点击“随机选一块”</div><canvas id="crop" hidden></canvas></div><div class="crop-wrap crop-3d-wrap" id="crop3dWrap"><div class="empty">选择候选区域后显示 3D 点云</div><div id="crop3d" hidden></div><div class="three-hint" id="threeHint" hidden>拖动旋转 · 滚轮缩放</div></div><div id="cropStats"></div><div class="fit-actions"><button id="runFit" disabled>拟合地形</button></div><div class="section" style="margin-top:12px;border:0"><div class="control full"><label>放入<select id="split"><option value="train">训练集</option><option value="val">验证集</option><option value="unassigned">未分配（历史记录）</option></select></label></div><div class="slot-box"><label>补充空位<select id="fillSlot"><option value="">不补充空位</option></select></label><div id="fillSlotHint" class="hint"></div></div><div class="review-update"><span id="reviewEditStatus" class="hint">未加载历史记录</span><button id="updateReviewSplit" disabled>更新当前历史记录归属</button></div><textarea id="note" class="note" placeholder="备注（可选）"></textarea><div class="review-actions"><button class="good" id="approve">通过并保存</button><button class="bad" id="reject">拒绝并保存</button><button id="approveNext">通过并随机下一块</button><button class="bad" id="rejectNext">拒绝并随机下一块</button></div></div></aside>
</div>
<div id="pathPicker" class="picker-backdrop" hidden><div class="picker" role="dialog" aria-modal="true" aria-labelledby="pickerTitle"><div class="picker-head"><h2 id="pickerTitle">选择位置</h2><button id="closePicker">取消</button></div><div class="picker-bar"><button id="pickerUp">上一级</button><input id="pickerPath" readonly aria-label="当前位置"><button id="pickerGo" hidden>转到</button></div><div id="pickerList" class="picker-list"></div><div class="picker-foot"><span id="pickerCurrent" class="workflow-path"></span><button id="usePickerPath" hidden>使用当前路径</button><button id="usePickerDirectory">选择这个文件夹</button></div></div></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script>
let SOURCES=__SOURCES__;
let OPEN_SLOTS=[];
const CONFIG=__CONFIG__;
const state={sourceId:SOURCES[0]?.id||'',sourceCategory:'',source:null,crop:null,fit:null,displayMode:'raw',selection:null,reviewRecord:null,fillSlotKey:'',selectedReviewTimestamps:new Set(),leftTab:'map',sourceToken:0,scale:1,panX:0,panY:0,drag:null,cache:{},three:null};
let confirmedDomain='';
let workflowAutoPaths={canonicalDir:'',manifestPath:'',datasetDir:'',trainSiteId:'',valSiteId:'',reviewPath:''};
const $=id=>document.getElementById(id),map=$('map'),mctx=map.getContext('2d'),cropCanvas=$('crop'),cctx=cropCanvas.getContext('2d');
const paneApp=$('app'),leftSplitter=$('leftSplitter'),rightSplitter=$('rightSplitter');
function paneWidth(side){return parseFloat(getComputedStyle(paneApp).getPropertyValue(`--${side}-width`))||({left:330,right:360}[side])}
function setPaneWidth(side,width){const rect=paneApp.getBoundingClientRect(),collapsed=paneApp.classList.contains('audit-collapsed'),wide=window.matchMedia('(min-width:1101px)').matches,minCenter=wide?400:340,splitter=parseFloat(getComputedStyle(paneApp).getPropertyValue('--pane-splitter-width'))||8,rightShown=!collapsed,fixed=splitter*(rightShown?2:1),other=side==='left'?paneWidth('right'):paneWidth('left'),limits=side==='left'?[240,560]:[280,560],max=Math.max(limits[0],rect.width-other-fixed-minCenter),next=Math.max(limits[0],Math.min(Math.min(limits[1],max),width));paneApp.style.setProperty(`--${side}-width`,`${next}px`);const splitterElement=side==='left'?leftSplitter:rightSplitter;splitterElement?.setAttribute('aria-valuenow',String(Math.round(next)))}
function beginPaneResize(side,event){if(side==='right'&&paneApp.classList.contains('audit-collapsed'))return;event.preventDefault();const startX=event.clientX,startWidth=paneWidth(side),move=e=>{e.preventDefault();const delta=e.clientX-startX;setPaneWidth(side,startWidth+(side==='left'?delta:-delta))},stop=()=>{document.removeEventListener('pointermove',move);document.removeEventListener('pointerup',stop)};document.addEventListener('pointermove',move);document.addEventListener('pointerup',stop,{once:true})}
function nudgePane(side,delta){setPaneWidth(side,paneWidth(side)+delta)}
leftSplitter.addEventListener('pointerdown',event=>beginPaneResize('left',event));rightSplitter.addEventListener('pointerdown',event=>beginPaneResize('right',event));for(const [element,side] of [[leftSplitter,'left'],[rightSplitter,'right']])element.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();event.stopPropagation();const direction=side==='left'?(event.key==='ArrowRight'?1:-1):(event.key==='ArrowLeft'?1:-1);nudgePane(side,direction*20)});
const classPalette={0:'#77838a',1:'#f5a623',2:'#76b852',3:'#d8c26b',4:'#caa45b',5:'#ef6d54',6:'#d75ae8',7:'#ffbe55',8:'#9c7bf2',9:'#5ab4e8',10:'#d4d4d4',11:'#f277b6',12:'#a9b6bc',13:'#54d0bd',14:'#7b8cff',15:'#ff9564',16:'#da83c8',17:'#b6dc67',18:'#f1e477'};
function bytes(s){const raw=atob(s||'');const out=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)out[i]=raw.charCodeAt(i);return out}
function typed(s,type){const b=bytes(s);return type==='f4'?new Float32Array(b.buffer):type==='u2'?new Uint16Array(b.buffer):new Uint8Array(b.buffer)}
function decode(p){return {x:typed(p.x,'f4'),y:typed(p.y,'f4'),z:typed(p.z,'f4'),classification:p.classification?typed(p.classification,'u8'):null,intensity:p.intensity?typed(p.intensity,'u2'):null,rgb:p.rgb?typed(p.rgb,'u8'):null,count:p.count}}
function resizeCanvas(c){const dpr=Math.min(devicePixelRatio||1,2),r=c.getBoundingClientRect();const w=Math.max(1,Math.round(r.width)),h=Math.max(1,Math.round(r.height));if(c.width!==w*dpr||c.height!==h*dpr){c.width=w*dpr;c.height=h*dpr}return {w,h,dpr}}
function currentSummary(){return state.source?.summary||null}
function bounds(){const b=currentSummary().bounds;return {xmin:b.min[0],ymin:b.min[1],xmax:b.max[0],ymax:b.max[1],cx:(b.min[0]+b.max[0])/2,cy:(b.min[1]+b.max[1])/2}}
function fit(){const {w,h}=resizeCanvas(map),b=bounds(),pad=35;state.scale=Math.min((w-2*pad)/(b.xmax-b.xmin),(h-2*pad)/(b.ymax-b.ymin));state.panX=0;state.panY=0;draw()}
function worldToScreen(x,y){const b=bounds(),r=resizeCanvas(map);return {x:r.w/2+(x-b.cx)*state.scale+state.panX,y:r.h/2-(y-b.cy)*state.scale+state.panY}}
function screenToWorld(x,y){const b=bounds(),r=resizeCanvas(map);return {x:b.cx+(x-r.w/2-state.panX)/state.scale,y:b.cy-(y-r.h/2-state.panY)/state.scale}}
function sourcePoints(){return state.source?.pointData}
function reviewRecords(){return state.source?.review_records||[]}
function visibleReviewEntries(){const all=reviewRecords();return all.map((record,index)=>({record,index})).filter(entry=>$('showAllReviews').checked||entry.record.decision==='approve')}
function reviewAt(world){const entries=visibleReviewEntries();for(let i=entries.length-1;i>=0;i--){const record=entries[i].record;if(!Array.isArray(record.center_xy))continue;const a=(Number(record.yaw_deg)||0)*Math.PI/180,ca=Math.cos(a),sa=Math.sin(a),dx=world.x-record.center_xy[0],dy=world.y-record.center_xy[1],localX=ca*dx+sa*dy,localY=-sa*dx+ca*dy,half=(Number(record.size_m)||20)/2+Math.max(.5,6/state.scale);if(Math.abs(localX)<=half&&Math.abs(localY)<=half)return record}return null}
function reviewSplitLabel(record){return record.split==='val'?'验证集':record.split==='train'?'训练集':'未分配'}
function reviewDetail(record){const decision=record.slot_status==='open'?'待补位':record.slot_status==='replaced'?'已被补位':record.decision==='approve'?(record.fills_slot?'补位通过':'通过'):record.decision==='return'?'打回空位':'拒绝';const center=record.center_xy||record.previous_center_xy||[0,0],slot=record.map_index==null?(record.slot_key?` · ${record.slot_key}`:''): ` · ${record.split}/map_${String(record.map_index).padStart(3,'0')}`;return`${decision}/${reviewSplitLabel(record)}${slot} · ${Number(center[0]).toFixed(2)}, ${Number(center[1]).toFixed(2)} · ${(Number(record.size_m)||20).toFixed(0)}m · ${(Number(record.yaw_deg)||0).toFixed(1)}°`}
function isCurrentReview(record){return Boolean(state.reviewRecord&&record&&state.reviewRecord.timestamp_utc&&record.timestamp_utc===state.reviewRecord.timestamp_utc)}
function reviewIsSelectable(record){return record.decision==='approve'&&!record.fills_slot&&!record.slot_status&&typeof record.timestamp_utc==='string'&&record.timestamp_utc.length>0}
function visibleSelectableReviews(){return visibleReviewEntries().filter(({record})=>reviewIsSelectable(record))}
function renderOpenSlots(){const host=$('openSlots'),select=$('fillSlot'),slots=OPEN_SLOTS||[];if(host)host.innerHTML=slots.length?`<strong>待补位 ${slots.length} 个</strong><div>${slots.map(slot=>slot.key).join(' · ')}</div>`:'<span>当前没有待补位地图</span>';if(!select)return;const previous=state.fillSlotKey;select.innerHTML='';select.add(new Option(slots.length?'不补充空位':'没有待补位',''));for(const slot of slots)select.add(new Option(`补充 ${slot.key}`,slot.key));state.fillSlotKey=slots.some(slot=>slot.key===previous)?previous:'';select.value=state.fillSlotKey;$('fillSlotHint').textContent=slots.length?'选择一个空位后点击“通过并保存”，新区域会占用这个编号。':'选择候选区域后可直接新增审核记录'}
function syncBulkReviewControls(){const selectAll=$('selectAllReviews'),count=$('selectedReviewCount'),button=$('bulkUpdateReview');if(!selectAll||!count||!button)return;const available=new Set(reviewRecords().filter(reviewIsSelectable).map(record=>record.timestamp_utc));for(const timestamp of state.selectedReviewTimestamps){if(!available.has(timestamp))state.selectedReviewTimestamps.delete(timestamp)}const visible=visibleSelectableReviews().map(({record})=>record),selectedVisible=visible.filter(record=>state.selectedReviewTimestamps.has(record.timestamp_utc));count.textContent=`已选 ${state.selectedReviewTimestamps.size} 条`;selectAll.disabled=!visible.length;selectAll.checked=visible.length>0&&selectedVisible.length===visible.length;selectAll.indeterminate=selectedVisible.length>0&&selectedVisible.length<visible.length;button.disabled=state.selectedReviewTimestamps.size===0}
function renderReviewHistory(){const host=$('reviewHistory');const entries=visibleReviewEntries().reverse();if(!entries.length){host.innerHTML='<div class="history-empty">当前没有可回看的审核选区</div>';syncBulkReviewControls();return}host.innerHTML=entries.map(({record,index})=>{const approve=record.decision==='approve',selectable=reviewIsSelectable(record),checked=selectable&&state.selectedReviewTimestamps.has(record.timestamp_utc);return`<div class="history-item ${approve?'approve':'reject'}${isCurrentReview(record)?' selected':''}">${selectable?`<input class="history-select" type="checkbox" data-review-index="${index}" ${checked?'checked':''} aria-label="选择这条通过记录">`:'<span class="history-select-spacer"></span>'}<button type="button" class="history-open" data-review-index="${index}" title="点击回看"><span class="history-mark">${approve?'通过':'拒绝'}</span><span class="history-detail">${reviewDetail(record)}</span></button></div>`}).join('');syncBulkReviewControls()}
function loadReview(record){if(!record||!Array.isArray(record.center_xy))return;$('note').value=record.note||'';setSelection(record.center_xy,Number(record.yaw_deg)||0,Number(record.size_m)||20,record);setStatus('正在回看历史审核选区…')}
function drawReviewMarkers(){if(!state.source)return;for(const {record,index} of visibleReviewEntries()){if(!Array.isArray(record.center_xy))continue;const approve=record.decision==='approve',open=record.slot_status==='open',activeApprove=approve&&!open,pts=corners({center_xy:record.center_xy,size_m:Number(record.size_m)||20,yaw_deg:Number(record.yaw_deg)||0});mctx.save();mctx.beginPath();const screen=pts.map(([x,y])=>worldToScreen(x,y));mctx.moveTo(screen[0].x,screen[0].y);for(let i=1;i<screen.length;i++)mctx.lineTo(screen[i].x,screen[i].y);mctx.closePath();mctx.fillStyle=activeApprove?'#4fc38a18':open?'#e8b85e18':'#ef766818';mctx.strokeStyle=activeApprove?'#4fc38acc':open?'#e8b85ecc':'#ef7668cc';mctx.lineWidth=1.5;mctx.setLineDash(activeApprove?[]:open?[7,4]:[5,4]);mctx.fill();mctx.stroke();mctx.setLineDash([]);const center=worldToScreen(...record.center_xy);mctx.fillStyle=activeApprove?'#75dba8':open?'#e8b85e':'#ff9388';mctx.beginPath();mctx.arc(center.x,center.y,3,0,Math.PI*2);mctx.fill();mctx.font='11px system-ui';mctx.fillText(`${activeApprove?'✓':open?'○':'×'}${index+1}`,center.x+5,center.y-5);mctx.restore()}}
function rgbColor(r,g,b,a=1){return `rgba(${r},${g},${b},${a})`}
function color(i,data,mode){const source=currentSummary();const actual=mode==='auto'?(source.has_rgb?'rgb':'classification'):mode;if(actual==='rgb'&&data.rgb){return rgbColor(data.rgb[i*3],data.rgb[i*3+1],data.rgb[i*3+2],.78)}if(actual==='classification'&&data.classification){return classPalette[data.classification[i]]||'#c4ccd0'}if(actual==='intensity'&&data.intensity){const max=source.intensity_max||65535,t=data.intensity[i]/max;return rgbColor(55+Math.round(200*t),100+Math.round(100*t),235-Math.round(170*t),.8)}const lo=source.bounds.min[2],hi=source.bounds.max[2],t=Math.max(0,Math.min(1,(data.z[i]+state.source.origin_z-lo)/(hi-lo||1)));return rgbColor(35+Math.round(215*t),85+Math.round(155*t),160-Math.round(100*t),.8)}
function drawPoints(ctx,c,data,origin,transform,mode,size,localSize){const n=data.count;if(!n)return;const pointSize=localSize||Math.max(1,+$('pointSize').value||2);const width=c.width/(devicePixelRatio||1),height=c.height/(devicePixelRatio||1);for(let i=0;i<n;i++){const x=origin[0]+data.x[i],y=origin[1]+data.y[i],p=transform(x,y);if(p.x<-pointSize||p.y<-pointSize||p.x>width+pointSize||p.y>height+pointSize)continue;ctx.fillStyle=color(i,data,mode);ctx.fillRect(p.x-pointSize*.5,p.y-pointSize*.5,pointSize,pointSize)}}
function corners(sel){const h=sel.size_m/2,a=sel.yaw_deg*Math.PI/180,ca=Math.cos(a),sa=Math.sin(a);return [[-h,-h],[h,-h],[h,h],[-h,h]].map(([x,y])=>[sel.center_xy[0]+ca*x-sa*y,sel.center_xy[1]+sa*x+ca*y])}
function drawSelection(){if(!state.selection)return;const pts=corners(state.selection).map(([x,y])=>worldToScreen(x,y));mctx.save();mctx.beginPath();mctx.moveTo(pts[0].x,pts[0].y);for(let i=1;i<pts.length;i++)mctx.lineTo(pts[i].x,pts[i].y);mctx.closePath();mctx.fillStyle='#61c4e022';mctx.fill();mctx.strokeStyle='#61d4f2';mctx.lineWidth=2;mctx.setLineDash([8,5]);mctx.stroke();mctx.setLineDash([]);const c=worldToScreen(...state.selection.center_xy);mctx.fillStyle='#fff';mctx.beginPath();mctx.arc(c.x,c.y,3,0,Math.PI*2);mctx.fill();mctx.fillStyle='#bfe8f2';mctx.font='12px system-ui';mctx.fillText(`${state.selection.size_m}m × ${state.selection.size_m}m · ${Number(state.selection.yaw_deg).toFixed(1)}°`,c.x+8,c.y-8);mctx.restore()}
function draw(){if(!state.source)return;const r=resizeCanvas(map),dpr=r.dpr;mctx.setTransform(dpr,0,0,dpr,0,0);mctx.clearRect(0,0,r.w,r.h);mctx.fillStyle='#0a0f12';mctx.fillRect(0,0,r.w,r.h);const data=sourcePoints();if(data)drawPoints(mctx,map,data,state.source.origin_xy,worldToScreen,$('colorMode').value,2);drawSelection();const b=bounds();mctx.fillStyle='#9db0ba';mctx.font='12px system-ui';mctx.fillText(`X ${b.xmin.toFixed(0)}–${b.xmax.toFixed(0)} · Y ${b.ymin.toFixed(0)}–${b.ymax.toFixed(0)} · 预览 ${data?.count||0} 点`,12,20);drawCrop()}
function drawCrop(){if(!state.crop){$('crop').hidden=true;$('cropWrap').querySelector('.empty').hidden=false;return}$('crop').hidden=false;$('cropWrap').querySelector('.empty').hidden=true;const r=resizeCanvas(cropCanvas),dpr=r.dpr;cctx.setTransform(dpr,0,0,dpr,0,0);cctx.clearRect(0,0,r.w,r.h);cctx.fillStyle='#0a0f12';cctx.fillRect(0,0,r.w,r.h);const sel=state.selection,half=sel.size_m/2,scale=Math.min(r.w,r.h)*.82/(sel.size_m),ox=r.w/2,oy=r.h/2;function tr(x,y){const dx=x-sel.center_xy[0],dy=y-sel.center_xy[1],a=sel.yaw_deg*Math.PI/180,ca=Math.cos(a),sa=Math.sin(a);return {x:ox+(ca*dx+sa*dy)*scale,y:oy-(-sa*dx+ca*dy)*scale}}drawPoints(cctx,cropCanvas,state.crop.points,state.crop.origin_xy,tr,$('colorMode').value,2,2);cctx.strokeStyle='#899ca5';cctx.strokeRect(ox-half*scale,oy-half*scale,sel.size_m*scale,sel.size_m*scale);cctx.strokeStyle='#ffffff55';cctx.beginPath();cctx.moveTo(ox-half*scale,oy);cctx.lineTo(ox+half*scale,oy);cctx.moveTo(ox,oy-half*scale);cctx.lineTo(ox,oy+half*scale);cctx.stroke()}
function clearThreeGroup(view){while(view.group&&view.group.children.length){const object=view.group.children[view.group.children.length-1];if(object.geometry)object.geometry.dispose();if(object.material){if(Array.isArray(object.material))object.material.forEach(m=>m.dispose());else object.material.dispose()}view.group.remove(object)}}
function initThreeView(){if(state.three)return state.three;if(!window.THREE)return null;const host=$('crop3d'),scene=new THREE.Scene();scene.background=new THREE.Color('#0a0f12');const camera=new THREE.PerspectiveCamera(45,1,.1,2000),renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));renderer.domElement.setAttribute('aria-label','候选区域三维点云');host.appendChild(renderer.domElement);const view={host,scene,camera,renderer,group:new THREE.Group(),yaw:.8,pitch:.55,distance:34,target:new THREE.Vector3(0,0,0),drag:null};scene.add(view.group);const resize=()=>{const r=host.getBoundingClientRect(),w=Math.max(1,r.width),h=Math.max(1,r.height);renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()};view.resize=resize;view.render=()=>{resize();const cp=Math.cos(view.pitch);camera.position.set(Math.sin(view.yaw)*cp*view.distance,Math.sin(view.pitch)*view.distance,Math.cos(view.yaw)*cp*view.distance);camera.lookAt(view.target);renderer.render(scene,camera)};host.addEventListener('pointerdown',e=>{view.drag={x:e.clientX,y:e.clientY,yaw:view.yaw,pitch:view.pitch};host.setPointerCapture(e.pointerId)});host.addEventListener('pointermove',e=>{if(!view.drag)return;view.yaw=view.drag.yaw-(e.clientX-view.drag.x)*.01;view.pitch=Math.max(-1.25,Math.min(1.25,view.drag.pitch+(e.clientY-view.drag.y)*.01));view.render()});host.addEventListener('pointerup',e=>{view.drag=null;host.releasePointerCapture(e.pointerId)});host.addEventListener('pointercancel',()=>{view.drag=null});host.addEventListener('wheel',e=>{e.preventDefault();view.distance=Math.max(4,Math.min(500,view.distance*Math.exp(e.deltaY*.001)));view.render()},{passive:false});state.three=view;return view}
function threeColor(i,data,mode,source,crop,baseZ,zMin,zMax){const actual=mode==='auto'?(source.has_rgb?'rgb':'classification'):mode;if(actual==='rgb'&&data.rgb)return new THREE.Color(data.rgb[i*3]/255,data.rgb[i*3+1]/255,data.rgb[i*3+2]/255);if(actual==='classification'&&data.classification)return new THREE.Color(classPalette[data.classification[i]]||'#c4ccd0');if(actual==='intensity'&&data.intensity){const t=data.intensity[i]/65535;return new THREE.Color(.2+.75*t,.4+.4*t,.9-.55*t)}const t=Math.max(0,Math.min(1,((data.z[i]+crop.origin_z)-zMin)/(zMax-zMin||1)));return new THREE.Color().setHSL(.62-.56*t,.75,.52)}
function addRawThreePoints(view,data,source,crop,selection,baseZ,verticalScale){const count=data.count;if(!count)return;let zMin=Infinity,zMax=-Infinity;for(let i=0;i<count;i++){const z=data.z[i]+crop.origin_z;zMin=Math.min(zMin,z);zMax=Math.max(zMax,z)}const a=selection.yaw_deg*Math.PI/180,ca=Math.cos(a),sa=Math.sin(a),positions=new Float32Array(count*3),colors=new Float32Array(count*3);for(let i=0;i<count;i++){const gx=crop.origin_xy[0]+data.x[i],gy=crop.origin_xy[1]+data.y[i],dx=gx-selection.center_xy[0],dy=gy-selection.center_xy[1],lx=ca*dx+sa*dy,ly=-sa*dx+ca*dy;positions[i*3]=lx;positions[i*3+1]=(data.z[i]+crop.origin_z-baseZ)*verticalScale;positions[i*3+2]=ly;threeColor(i,data,$('colorMode').value,source,crop,baseZ,zMin,zMax).toArray(colors,i*3)}const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));const material=new THREE.PointsMaterial({size:Math.max(.06,selection.size_m*.008),sizeAttenuation:true,vertexColors:true,transparent:true,opacity:.9});view.group.add(new THREE.Points(geometry,material));return {zMin,zMax}}
function addFitThreePoints(view,fit,verticalScale){const grid=fit.grid,x=typed(grid.x,'f4'),y=typed(grid.y,'f4'),z=typed(grid.z,'f4'),valid=typed(grid.valid,'u8'),positions=[],colors=[],range=fit.stats.elevation_range_m;for(let i=0;i<valid.length;i++){if(!valid[i])continue;positions.push(x[i],z[i]*verticalScale,y[i]);const t=Math.max(0,Math.min(1,(z[i]-range[0])/(range[1]-range[0]||1)));const c=new THREE.Color().setHSL(.52-.42*t,.82,.55);colors.push(c.r,c.g,c.b)}const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));const material=new THREE.PointsMaterial({size:.07,sizeAttenuation:true,vertexColors:true,transparent:true,opacity:.78});view.group.add(new THREE.Points(geometry,material))}
function renderCrop3d(){const wrap=$('crop3dWrap'),empty=wrap.querySelector('.empty');if(!state.crop){$('crop3d').hidden=true;$('threeHint').hidden=true;empty.hidden=false;if(state.three)clearThreeGroup(state.three);return}const view=initThreeView();if(!view){$('crop3d').hidden=true;$('threeHint').hidden=true;empty.hidden=false;empty.textContent='3D 组件未加载，仍可使用左侧二维视图';return}$('crop3d').hidden=false;$('threeHint').hidden=false;empty.hidden=true;clearThreeGroup(view);const visibleFit=state.fit&&state.displayMode==='fit',fitRange=visibleFit?.stats?.elevation_range_m||null,reference=state.crop.stats.reference_z_range||state.crop.stats.class_1_plus_2_z_range||null,referenceRange=reference?.z_range||state.crop.stats.z_range||1,groundRange=fitRange?Math.max(.1,fitRange[1]-fitRange[0]):Math.max(.1,referenceRange),verticalScale=Math.max(.5,Math.min(4,state.selection.size_m/Math.max(groundRange*2,state.selection.size_m*.25)));view.group.scale.set(1,1,1);const baseZ=visibleFit?state.fit.vertical_origin_m:reference?.z_min??state.crop.origin_z;addRawThreePoints(view,state.crop.points,state.source.summary,state.crop,state.selection,baseZ,verticalScale);if(visibleFit)addFitThreePoints(view,state.fit,verticalScale);const floor=new THREE.GridHelper(state.selection.size_m,10,0x49616b,0x263840);floor.material.transparent=true;floor.material.opacity=.45;view.group.add(floor);const axis=new THREE.AxesHelper(state.selection.size_m*.3);view.group.add(axis);view.group.scale.y=1;view.target.set(0,Math.max(0,groundRange*verticalScale*.25),0);view.distance=Math.max(state.selection.size_m*1.55,groundRange*verticalScale*2.6);view.yaw=.8;view.pitch=.55;view.render()}
function fitSummaryHtml(){if(!state.fit)return'';const s=state.fit.stats;return`<div class="fit-summary"><div>候选区原始点 ${s.raw_points_in_fit_margin.toLocaleString()} 点，参与拟合 ${s.fit_points.toLocaleString()} 点</div><div>${s.classification_policy}；仅剔除低于下包络 ${s.ground_band_below_m.toFixed(2)} m 的低噪声点，不设置上方高度上限</div><div>高程网格观测 ${(s.observed_fraction*100).toFixed(1)}% · 高程范围 ${s.elevation_range_m[0].toFixed(2)}–${s.elevation_range_m[1].toFixed(2)} m</div><div>坡度 median ${s.slope_degrees.median.toFixed(1)}° / p95 ${s.slope_degrees.p95.toFixed(1)}° / max ${s.slope_degrees.maximum.toFixed(1)}°</div></div>`}
function setStatus(text){$('status').textContent=text}
async function responseJson(response){let payload;try{payload=await response.json()}catch(_){payload={error:await response.text()}}if(!response.ok)throw new Error(payload.error||`HTTP ${response.status}`);return payload}
async function postJson(url,body){return responseJson(await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}))}
function sourceLabel(source){return source.source_group?`${source.source_group} / ${source.name}`:source.name}
function sourceCategoryLabel(source){return source.source_category||source.source_group||'未分类'}
function clearSourceSelection(){const category=$('sourceCategory'),select=$('source');category.innerHTML='';select.innerHTML='';category.disabled=true;select.disabled=true;state.sourceCategory='';state.sourceId='';state.source=null;state.crop=null;state.fit=null;state.reviewRecord=null;state.selectedReviewTimestamps.clear();renderCropStats();renderCrop3d();setStatus('请先选择或导入一张原始地图')}
function renderSourceOptions(preferredId=''){const category=$('sourceCategory'),select=$('source'),current=preferredId||state.sourceId;const categories=[...new Set(SOURCES.map(sourceCategoryLabel))].sort((a,b)=>a.localeCompare(b));category.innerHTML='';for(const name of categories){const count=SOURCES.filter(source=>sourceCategoryLabel(source)===name).length;category.add(new Option(`${name}（${count}张）`,name))}if(!categories.length){clearSourceSelection();return''}const preferredSource=SOURCES.find(source=>source.id===preferredId);const currentSource=SOURCES.find(source=>source.id===current);const targetCategory=preferredSource?sourceCategoryLabel(preferredSource):categories.includes(state.sourceCategory)?state.sourceCategory:currentSource?sourceCategoryLabel(currentSource):categories[0];state.sourceCategory=targetCategory;category.disabled=false;category.value=targetCategory;const candidates=SOURCES.filter(source=>sourceCategoryLabel(source)===targetCategory);select.innerHTML='';for(const s of candidates){const option=new Option(`${sourceLabel(s)} · ${(s.point_count/1e6).toFixed(2)}M 点`,s.id);option.title=s.path;select.add(option)}const target=candidates.some(source=>source.id===current)?current:(candidates[0]?.id||'');if(!target){clearSourceSelection();return''}select.disabled=false;select.value=target;state.sourceId=target;return target}
function updateWorkflowLinks(payload){if(payload.review_file)$('reviewPath').value=payload.review_file;OPEN_SLOTS=payload.open_slots||OPEN_SLOTS||[];renderOpenSlots();CONFIG.workspace_root=payload.workspace_root||CONFIG.workspace_root;const domain=$('domain').value||payload.default_domain||'terrain';if(!$('canonicalDir').value)$('canonicalDir').value=`dataset/canonical/${domain}/reviewed_regions`;if(!$('manifestPath').value)$('manifestPath').value=`${$('canonicalDir').value}/approved_canonical_manifest.json`;if(!$('datasetDir').value)$('datasetDir').value=`dataset/public_terrain_20m/${domain}`;$('datasetViewer').href=payload.dataset_viewer_url||CONFIG.dataset_viewer_url||'#';$('downloadReviews').href='/api/review-file';const totals=payload.sources.reduce((result,source)=>({approve:result.approve+(source.review_counts?.approve||0),reject:result.reject+(source.review_counts?.reject||0)}),{approve:0,reject:0});const splits=payload.sources.reduce((result,source)=>{const counts=source.review_split_counts||{};return{train:result.train+(counts.train||0),val:result.val+(counts.val||0),unassigned:result.unassigned+(counts.unassigned||0)}},{train:0,val:0,unassigned:0});const gapText=OPEN_SLOTS.length?` · 待补位 ${OPEN_SLOTS.length} 个`:'';$('workflowMeta').textContent=`已加载 ${payload.sources.length} 张原始地图 · 已保存 ${totals.approve+totals.reject} 条标注 · 通过区域：训练 ${splits.train} / 验证 ${splits.val} / 未分配 ${splits.unassigned}${gapText}`;$('canonicalInfo').textContent=!totals.approve?'还没有通过的区域':splits.unassigned?`已有 ${totals.approve} 个通过区域，其中 ${splits.unassigned} 个未分配训练集/验证集${gapText}`:`已有 ${totals.approve} 个通过区域（训练 ${splits.train} / 验证 ${splits.val}），可以生成训练用地图${gapText}`;$('datasetInfo').textContent=!totals.approve?'请先在审核页通过至少一个区域':splits.unassigned?`还有 ${splits.unassigned} 个通过区域未分配，完成归属后再生成`:'训练用地图生成后，这里会自动生成训练集、验证集和轨迹'}
const baseUpdateWorkflowLinks=updateWorkflowLinks;updateWorkflowLinks=function(payload){const previousReviewPath=$('reviewPath').value.trim(),preserveReviewPath=Boolean(previousReviewPath&&workflowAutoPaths.reviewPath&&previousReviewPath!==workflowAutoPaths.reviewPath);baseUpdateWorkflowLinks(payload);if(preserveReviewPath)$('reviewPath').value=previousReviewPath}
function rememberWorkflowPaths(){workflowAutoPaths={canonicalDir:$('canonicalDir').value.trim(),manifestPath:$('manifestPath').value.trim(),datasetDir:$('datasetDir').value.trim(),trainSiteId:$('trainSiteId').value.trim(),valSiteId:$('valSiteId').value.trim(),reviewPath:$('reviewPath').value.trim()}}
function syncWorkflowPaths(domain){const canonicalDefault=`dataset/canonical/${domain}/reviewed_regions`,currentCanonical=$('canonicalDir').value.trim(),canonicalAutomatic=!currentCanonical||!workflowAutoPaths.canonicalDir||currentCanonical===workflowAutoPaths.canonicalDir,canonical=canonicalAutomatic?canonicalDefault:currentCanonical,currentManifest=$('manifestPath').value.trim(),manifestAutomatic=!currentManifest||!workflowAutoPaths.manifestPath||currentManifest===workflowAutoPaths.manifestPath,manifest=manifestAutomatic?`${canonical}/approved_canonical_manifest.json`:currentManifest,datasetDefault=`dataset/public_terrain_20m/${domain}`,currentDataset=$('datasetDir').value.trim(),datasetAutomatic=!currentDataset||!workflowAutoPaths.datasetDir||currentDataset===workflowAutoPaths.datasetDir,dataset=datasetAutomatic?datasetDefault:currentDataset,trainSiteDefault=`${domain}_train`,currentTrainSite=$('trainSiteId').value.trim(),trainSiteAutomatic=!currentTrainSite||!workflowAutoPaths.trainSiteId||currentTrainSite===workflowAutoPaths.trainSiteId,valSiteDefault=`${domain}_val`,currentValSite=$('valSiteId').value.trim(),valSiteAutomatic=!currentValSite||!workflowAutoPaths.valSiteId||currentValSite===workflowAutoPaths.valSiteId;$('canonicalDir').value=canonical;$('manifestPath').value=manifest;$('datasetDir').value=dataset;$('trainSiteId').value=trainSiteAutomatic?trainSiteDefault:currentTrainSite;$('valSiteId').value=valSiteAutomatic?valSiteDefault:currentValSite;workflowAutoPaths={canonicalDir:canonicalAutomatic?canonical:canonicalDefault,manifestPath:manifestAutomatic?manifest:manifestDefaultFor(canonicalDefault),datasetDir:datasetAutomatic?dataset:datasetDefault,trainSiteId:trainSiteAutomatic?trainSiteDefault:currentTrainSite,valSiteId:valSiteAutomatic?valSiteDefault:currentValSite}}
function manifestDefaultFor(canonical){return`${canonical}/approved_canonical_manifest.json`}
function syncReviewPath(domain){const current=$('reviewPath').value.trim(),automatic=!current||!workflowAutoPaths.reviewPath||current===workflowAutoPaths.reviewPath;if(!automatic)return false;const root=CONFIG.workspace_root||'',next=`${root}/dataset/reviews/${domain}/manual_regions.jsonl`;$('reviewPath').value=next;workflowAutoPaths.reviewPath=next;return true}
async function confirmDomain(){const domain=$('domain').value.trim();if(!/^[A-Za-z0-9_.-]+$/.test(domain)){setStatus('数据名称只能包含字母、数字、点、下划线或短横线');return false}const reviewPathMarker=workflowAutoPaths.reviewPath;syncWorkflowPaths(domain);workflowAutoPaths.reviewPath=reviewPathMarker;const reviewAutomatic=syncReviewPath(domain);confirmedDomain=domain;if(reviewAutomatic){setStatus(`正在加载 ${domain} 的审核记录…`);await postJson('/api/load-review',{path:$('reviewPath').value});await refreshCatalog(state.sourceId)}setStatus(`数据名称已确认：${domain}，相关默认路径已同步`);return true}
function ensureDomainConfirmed(){const domain=$('domain').value.trim();if(domain!==confirmedDomain){setStatus('本次数据名称已修改，请先点击“确认”');return false}return true}
function updateViewerControls(viewer){const running=Boolean(viewer&&viewer.running);$('startDatasetViewer').disabled=running;$('openDatasetViewer').disabled=!running;$('stopDatasetViewer').disabled=!running;if(!viewer||!viewer.dataset_root){$('viewerStatus').textContent='结果查看器未启动';return}const stateText=running?'运行中':viewer.returncode===0?'已关闭':`已结束（返回码 ${viewer.returncode}）`,root=viewer.dataset_root?` · 数据目录：${viewer.dataset_root}`:'',log=!running&&viewer.returncode!==null&&viewer.returncode!==0&&viewer.log_path?` · 日志：${viewer.log_path}`:'';$('viewerStatus').textContent=`结果查看器${stateText}${root}${log}`}
async function refreshViewer(){try{const viewer=await responseJson(await fetch('/api/viewer'));updateViewerControls(viewer);if(viewer.running)window.setTimeout(refreshViewer,1000)}catch(e){$('viewerStatus').textContent='读取查看器状态失败：'+e.message}}
async function startDatasetViewer(){const datasetDir=$('datasetDir').value.trim(),canonicalDir=$('canonicalDir').value.trim(),domain=$('domain').value.trim();if(!datasetDir){setStatus('请先设置最终数据文件夹');return}setStatus('正在启动结果查看器…');const viewer=await postJson('/api/start-viewer',{dataset_dir:datasetDir,canonical_dir:canonicalDir,domain});updateViewerControls(viewer);if(viewer.running)setStatus('结果查看器已启动，请点击“打开结果查看器”');refreshViewer()}
function openDatasetViewer(){const url=CONFIG.dataset_viewer_url||'#';if(url==='#'){setStatus('没有配置结果查看器地址');return}window.open(url,'_blank','noopener')}
async function stopDatasetViewer(){setStatus('正在关闭结果查看器…');const viewer=await postJson('/api/stop-viewer',{});updateViewerControls(viewer);setStatus('结果查看器已关闭');refreshViewer()}
const pickerState={target:'',mode:'directory',filters:[],path:'',parent:null};
function pickerStartPath(target){return $(target).value.trim()||CONFIG.workspace_root||'.'}
function closePicker(){$('pathPicker').hidden=true}
async function browsePicker(path){const payload=await responseJson(await fetch('/api/browse?path='+encodeURIComponent(path)));pickerState.path=payload.path;pickerState.parent=payload.parent;$('pickerPath').value=payload.requested_path===payload.path?payload.path:path;$('pickerCurrent').textContent=payload.path;$('pickerUp').disabled=!payload.parent;const host=$('pickerList');host.replaceChildren();for(const directory of payload.directories){const button=document.createElement('button');button.className='picker-item';button.textContent='目录 · '+directory.name;button.addEventListener('click',()=>browsePicker(directory.path).catch(e=>setStatus('浏览目录失败：'+e.message)));host.appendChild(button)}const files=payload.files.filter(file=>pickerState.mode==='file'&&(!pickerState.filters.length||pickerState.filters.some(extension=>file.name.toLowerCase().endsWith(extension))));for(const file of files){const button=document.createElement('button');button.className='picker-item file';button.textContent='文件 · '+file.name;button.addEventListener('click',()=>{$(pickerState.target).value=file.path;closePicker()});host.appendChild(button)}if(!host.children.length){const empty=document.createElement('div');empty.className='history-empty';empty.textContent=pickerState.mode==='file'?'当前目录没有符合条件的文件':'当前目录没有子目录';host.appendChild(empty)}}
async function openPicker(target,mode,filters=[]){pickerState.target=target;pickerState.mode=mode;pickerState.filters=filters;$('pickerTitle').textContent=mode==='file'?'选择标注记录文件':target==='loadPath'?'选择已有地图文件夹':'选择保存文件夹';$('usePickerDirectory').hidden=mode==='file';$('pathPicker').hidden=false;await browsePicker(pickerStartPath(target))}
function choosePickerDirectory(){if(pickerState.mode!=='directory')return;const target=pickerState.target;$(target).value=pickerState.path;if(target==='canonicalDir')$('manifestPath').value=`${pickerState.path}/approved_canonical_manifest.json`;closePicker();if(target==='loadPath')loadRoot().catch(e=>setStatus('加载已有地图失败：'+e.message))}
function choosePickerPath(){if(pickerState.mode!=='directory')return;const target=pickerState.target,path=$('pickerPath').value.trim();if(!path)return;$(target).value=path;if(target==='canonicalDir')$('manifestPath').value=`${path}/approved_canonical_manifest.json`;closePicker();if(target==='loadPath')loadRoot().catch(e=>setStatus('加载已有地图失败：'+e.message))}
async function refreshCatalog(preferredId=''){const payload=await responseJson(await fetch('/api/index'));SOURCES=payload.sources;CONFIG.review_file=payload.review_file;CONFIG.dataset_viewer_url=payload.dataset_viewer_url;updateWorkflowLinks(payload);const target=renderSourceOptions(preferredId);if(target)loadSource(target).catch(e=>setStatus('刷新后加载失败：'+e.message))}
	function jobText(job){if(!job)return'当前没有运行任务';const names={canonical_export:'训练用地图生成',dataset_generation:'最终训练数据生成'};const stateText=job.running?'运行中':job.returncode===0?'已完成':`已结束（返回码 ${job.returncode}）`;const log=job.returncode&&job.returncode!==0?`\n详细日志：${job.log_path}`:'';return`${names[job.kind]||'后台生成'} · ${stateText}${log}\n\n${job.tail||'等待日志…'}`}
async function refreshJobs(){try{const jobs=await responseJson(await fetch('/api/jobs')),job=jobs.length?jobs[jobs.length-1]:null;$('jobStatus').textContent=jobText(job);$('stopJob').disabled=!job||!job.running;if(job&&job.running)window.setTimeout(refreshJobs,1000)}catch(e){$('jobStatus').textContent='读取任务状态失败：'+e.message}}
async function importMaps(){const files=$('mapFiles').files;if(!files.length)return;const form=new FormData();form.append('domain',$('domain').value);form.append('site','imported');for(const file of files)form.append('map_files',file,file.name);setStatus('正在导入原始地图…');const payload=await responseJson(await fetch('/api/import',{method:'POST',body:form}));$('loadPath').value=payload.review_input;await refreshCatalog(payload.source_ids?.[0]||'');setStatus(`已导入 ${payload.files.length} 个原始地图文件`);$('mapFiles').value=''}
async function importReview(){const file=$('reviewFile').files[0];if(!file)return;const form=new FormData();form.append('domain',$('domain').value);form.append('review_file',file,file.name);setStatus('正在导入标注备份…');const payload=await responseJson(await fetch('/api/import-review',{method:'POST',body:form}));$('reviewPath').value=payload.review_file;await refreshCatalog();setStatus('标注备份已导入并加载');$('reviewFile').value=''}
async function loadRoot(){const path=$('loadPath').value.trim();if(!path){setStatus('请先选择一个已有地图文件夹');return}setStatus('正在读取已有地图…');const payload=await postJson('/api/load-root',{path});await refreshCatalog(payload.source_ids?.[0]||'');setStatus('已有地图已加载')}
async function switchReview(){const path=$('reviewPath').value.trim();if(!path){setStatus('请先选择标注记录文件');return}setStatus('正在加载标注记录…');await postJson('/api/load-review',{path});await refreshCatalog();setStatus('标注记录已加载')}
function exportBody(){const domain=$('domain').value.trim()||'terrain';return{output_dir:$('canonicalDir').value.trim(),domain,train_site_id:$('trainSiteId').value.trim()||`${domain}_train`,val_site_id:$('valSiteId').value.trim()||`${domain}_val`,train_source_profile:$('trainProfile').value,val_source_profile:$('valProfile').value,overwrite:$('overwriteCanonical').checked}}
async function exportCanonical(){const body=exportBody();if(!body.output_dir){setStatus('请在高级设置中选择训练地图保存位置');return}setStatus('正在生成训练用地图…');const job=await postJson('/api/export-canonical',body);$('manifestPath').value=`${body.output_dir}/approved_canonical_manifest.json`;$('jobStatus').textContent=jobText(job);refreshJobs()}
async function generateDataset(){const manifest=$('manifestPath').value.trim()||`${$('canonicalDir').value.trim()}/approved_canonical_manifest.json`,body={manifest_path:manifest,output_dir:$('datasetDir').value.trim(),train_paths:+$('trainPaths').value,val_paths:+$('valPaths').value,workers:+$('workers').value,ros_port:+$('rosPort').value};if(!body.manifest_path||!body.output_dir){setStatus('请先在高级设置中选择最终数据保存位置');return}setStatus('正在生成最终训练数据…');const job=await postJson('/api/generate-dataset',body);$('jobStatus').textContent=jobText(job);refreshJobs()}
const exportCanonicalWithDomainCheck=exportCanonical;exportCanonical=async function(){if(!ensureDomainConfirmed())return;return exportCanonicalWithDomainCheck()};const generateDatasetWithDomainCheck=generateDataset;generateDataset=async function(){if(!ensureDomainConfirmed())return;return generateDatasetWithDomainCheck()}
function renderSource(){const s=currentSummary();if(!s)return;$('sourceMeta').innerHTML=`<div>目录：${s.source_category||'未分类'}</div><div>来源：${sourceLabel(s)}</div><div>${s.point_count.toLocaleString()} 个点 · ${s.has_rgb?'含 RGB 原色':'无 RGB 字段'}</div><div>分类：${s.has_classification?'有':'无'} · 强度：${s.has_intensity?'有':'无'}</div>`;const counts=state.source.review_counts||{approve:0,reject:0};$('approved').textContent=counts.approve;$('rejected').textContent=counts.reject;$('colorMode').value=s.has_rgb?'auto':'classification';$('colorMode').querySelector('option[value="rgb"]').disabled=!s.has_rgb;$('legend').innerHTML=s.has_rgb?'<span><i style="background:#fff"></i>原始 RGB</span>':'<span><i style="background:#76b852"></i>class 2</span><span><i style="background:#f5a623"></i>class 1</span><span>颜色为 LAS 分类回退</span>';renderOpenSlots();renderReviewHistory();fit()}
const classNames={0:'从未分类',1:'未分类',2:'地面',3:'低植被',4:'中植被',5:'高植被',6:'建筑物',7:'低点/噪声',8:'模型关键点',9:'水体',10:'铁路',11:'道路表面',12:'重叠点',18:'高噪声'};
function classTable(counts,ranges,total,allRange){const entries=Object.entries(counts||{}).sort((a,b)=>+a[0]-+b[0]);if(!entries.length)return`<table class="class-table"><thead><tr><th>类别</th><th>点数</th><th>高差</th></tr></thead><tbody><tr><td>全部点（无 LAS 分类）</td><td>${Number(total||0).toLocaleString()}</td><td>${rangeText(allRange)}</td></tr></tbody></table>`;return'<table class="class-table"><thead><tr><th>类别</th><th>点数</th><th>高差</th></tr></thead><tbody>'+entries.map(([k,v])=>`<tr><td>class ${k} · ${classNames[k]||'其他 LAS 类别'}</td><td>${Number(v).toLocaleString()}</td><td>${rangeText(ranges?.[k])}</td></tr>`).join('')+'</tbody></table>'}
function rangeText(value){return value&&Number.isFinite(value.z_range)?value.z_range.toFixed(2)+' m':'—'}
function updateReviewControls(){const button=$('updateReviewSplit'),status=$('reviewEditStatus'),record=state.reviewRecord,split=$('split').value;if(!record){button.disabled=true;status.textContent='未加载历史记录';return}if(record.decision==='return'){button.disabled=true;status.textContent='这是已打回的地图空位，请在“补充空位”中选择它';return}if(record.decision!=='approve'){button.disabled=true;status.textContent='当前是拒绝记录，不参与训练集/验证集地图生成';return}if(record.fills_slot){button.disabled=true;status.textContent='这是已补位记录，不能单独修改数据集归属';return}if(!record.timestamp_utc){button.disabled=true;status.textContent='这条旧记录缺少时间戳，无法精确更新';return}button.disabled=split==='unassigned'||split===record.split;status.textContent=`当前历史记录：${reviewSplitLabel(record)}${split!==record.split&&split!=='unassigned'?' · 选择后可更新':''}`}
function updateFitButton(){const button=$('runFit');if(!state.crop){button.disabled=true;button.textContent='拟合地形';return}button.disabled=false;button.textContent=!state.fit?'拟合地形':state.displayMode==='fit'?'显示原点云':'显示拟合结果'}
function renderCropStats(){if(!state.crop){$('cropStats').innerHTML='';$('decisionStatus').textContent='未选择';updateFitButton();updateReviewControls();return}const s=state.crop.stats,sel=state.selection,byClass=s.z_range_by_class||{};$('decisionStatus').textContent=`中心 ${sel.center_xy[0].toFixed(2)}, ${sel.center_xy[1].toFixed(2)}`;updateFitButton();updateReviewControls();$('cropStats').innerHTML=`<div class="stats"><div class="stat"><div class="k">原始/显示点数</div><div class="v">${s.point_count.toLocaleString()} / ${s.display_point_count.toLocaleString()}</div></div><div class="stat"><div class="k">1m 网格占用</div><div class="v">${(s.occupied_1m_fraction*100).toFixed(1)}%</div></div>${classTable(s.class_counts,byClass,s.point_count,s.z_range)}${fitSummaryHtml()}`}
async function loadSource(id){state.sourceId=id;const sourceToken=++state.sourceToken;state.source=null;state.crop=null;state.fit=null;state.displayMode='raw';state.selection=null;state.reviewRecord=null;state.fillSlotKey='';state.selectedReviewTimestamps.clear();$('split').value='train';renderCropStats();renderCrop3d();setStatus('正在读取母图预览…');const r=await fetch('/api/source?id='+encodeURIComponent(id));if(!r.ok)throw new Error(await r.text());const payload=await r.json();if(sourceToken!==state.sourceToken||state.sourceId!==id)return;state.source=payload;OPEN_SLOTS=payload.open_slots||OPEN_SLOTS;state.source.pointData=decode(payload.points);$('source').value=id;renderOpenSlots();renderSource();setStatus('可框选候选区域');draw()}
function selectionKey(selection){return selection?[selection.center_xy[0],selection.center_xy[1],selection.size_m,selection.yaw_deg].join('|'):''}
async function requestCrop(){if(!state.selection)return;const s=state.selection;const sourceId=state.sourceId;const sourceToken=state.sourceToken;const requestedSelection=selectionKey(s);setStatus('正在读取原始点云框内数据…');const q=new URLSearchParams({source:sourceId,cx:s.center_xy[0],cy:s.center_xy[1],size:s.size_m,yaw:s.yaw_deg});const r=await fetch('/api/crop?'+q);if(!r.ok)throw new Error(await r.text());const payload=await r.json();if(sourceToken!==state.sourceToken||sourceId!==state.sourceId||requestedSelection!==selectionKey(state.selection))return;state.crop=payload;state.crop.points=decode(state.crop.points);state.fit=null;state.displayMode='raw';renderCropStats();renderCrop3d();setStatus('候选已加载，可人工审批');draw()}
function setSelection(center,yaw=null,size=null,historicalRecord=null){const selectedSize=Math.max(1,Number(size??$('size').value)||20),angle=yaw===null?(+$('yaw').value||0):yaw;state.reviewRecord=historicalRecord;if(historicalRecord)$('split').value=historicalRecord.split==='val'?'val':historicalRecord.split==='train'?'train':'unassigned';else if($('split').value==='unassigned')$('split').value='train';$('size').value=selectedSize;$('yaw').value=Number(angle).toFixed(1);state.selection={center_xy:[+center[0],+center[1]],size_m:selectedSize,yaw_deg:+angle};state.crop=null;state.fit=null;state.displayMode='raw';renderReviewHistory();renderCropStats();renderCrop3d();requestCrop().catch(e=>setStatus('加载候选失败：'+e.message));draw()}
async function randomCandidate(){const sourceId=state.sourceId;const sourceToken=state.sourceToken;setStatus('正在随机选择母图位置…');const q=new URLSearchParams({source:sourceId,size:+$('size').value||20,random_yaw:$('randomYaw').checked?'1':'0'});const r=await fetch('/api/random?'+q);if(!r.ok)throw new Error(await r.text());const c=await r.json();if(sourceToken!==state.sourceToken||sourceId!==state.sourceId)return;setSelection(c.center_xy,c.yaw_deg)}
async function runFit(){if(!state.crop||!state.selection)return;if(state.fit){state.displayMode=state.displayMode==='fit'?'raw':'fit';renderCropStats();renderCrop3d();setStatus(state.displayMode==='fit'?'已显示拟合地形':'已显示原点云');return}const requestedSelection=selectionKey(state.selection),sourceToken=state.sourceToken,s=state.selection,button=$('runFit');button.disabled=true;setStatus('正在拟合地形…');try{const r=await fetch('/api/fit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_id:state.sourceId,center_xy:s.center_xy,size_m:s.size_m,yaw_deg:s.yaw_deg})});if(!r.ok)throw new Error(await r.text());const payload=await r.json();if(sourceToken!==state.sourceToken||requestedSelection!==selectionKey(state.selection))return;state.fit=payload;state.displayMode='fit';renderCropStats();renderCrop3d();setStatus('拟合完成')}finally{updateFitButton()}}
function applyReviewResponse(response){if(response.sources)SOURCES=response.sources;state.source.review_counts=response.review_counts||state.source.review_counts;state.source.review_split_counts=response.review_split_counts||state.source.review_split_counts;state.source.review_records=response.review_records||reviewRecords();OPEN_SLOTS=response.open_slots||OPEN_SLOTS;state.source.open_slots=OPEN_SLOTS;renderOpenSlots();if(state.reviewRecord?.timestamp_utc){const current=state.source.review_records.find(record=>record.timestamp_utc===state.reviewRecord.timestamp_utc);state.reviewRecord=current||null;if(current)$('split').value=current.split==='val'?'val':current.split==='train'?'train':'unassigned'}const sourceIndex=SOURCES.findIndex(source=>source.id===state.sourceId);if(sourceIndex>=0){SOURCES[sourceIndex].review_counts=state.source.review_counts;SOURCES[sourceIndex].review_split_counts=state.source.review_split_counts}}
async function review(decision,thenNext=false){if(!state.crop||!state.selection)return;if($('split').value==='unassigned'){setStatus('新审核记录必须选择训练集或验证集');return}const s=state.selection,slot=OPEN_SLOTS.find(item=>item.key===state.fillSlotKey);if(decision==='approve'&&state.fillSlotKey&&!slot){setStatus('这个待补位已被其他操作处理，请重新读取');return}const body={source_id:state.sourceId,decision,split:slot?slot.split:$('split').value,center_xy:s.center_xy,size_m:s.size_m,yaw_deg:s.yaw_deg,note:$('note').value,stats:state.crop.stats,fit:state.fit?{method:state.fit.method,stats:state.fit.stats}:null};if(slot)body.fills_slot={split:slot.split,map_index:slot.map_index};const r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok)throw new Error(await r.text());const response=await r.json();applyReviewResponse(response);state.reviewRecord=null;state.fillSlotKey='';renderOpenSlots();renderSource();updateReviewControls();updateWorkflowLinks({sources:SOURCES,open_slots:OPEN_SLOTS,review_file:CONFIG.review_file,default_domain:CONFIG.default_domain,dataset_viewer_url:CONFIG.dataset_viewer_url,workspace_root:CONFIG.workspace_root});$('note').value='';setStatus(decision==='approve'?(slot?`已补充 ${slot.key}`:'已记录通过'):'已记录拒绝');if(thenNext)await randomCandidate()}
async function updateReviewSplit(){const record=state.reviewRecord,split=$('split').value;if(!record||record.decision!=='approve'||!record.timestamp_utc)return;if(split==='unassigned'){setStatus('请选择训练集或验证集后再更新');return}setStatus('正在更新历史记录归属…');const response=await postJson('/api/update-review',{source_id:state.sourceId,timestamp_utc:record.timestamp_utc,split});applyReviewResponse(response);renderReviewHistory();updateReviewControls();updateWorkflowLinks({sources:SOURCES,review_file:CONFIG.review_file,default_domain:CONFIG.default_domain,dataset_viewer_url:CONFIG.dataset_viewer_url,workspace_root:CONFIG.workspace_root});draw();setStatus(`历史记录已调整为${reviewSplitLabel(response.record)}`)}
async function updateSelectedReviewSplits(){const timestamps=[...state.selectedReviewTimestamps],split=$('bulkSplit').value;if(!timestamps.length)return;setStatus(`正在批量更新 ${timestamps.length} 条历史记录…`);const response=await postJson('/api/update-review-batch',{source_id:state.sourceId,timestamps_utc:timestamps,split});applyReviewResponse(response);state.selectedReviewTimestamps.clear();renderReviewHistory();updateReviewControls();updateWorkflowLinks({sources:SOURCES,review_file:CONFIG.review_file,default_domain:CONFIG.default_domain,dataset_viewer_url:CONFIG.dataset_viewer_url,workspace_root:CONFIG.workspace_root});draw();setStatus(`已将 ${response.updated_count} 条历史记录调整为${split==='val'?'验证集':'训练集'}`)}
function setLeftTab(tab){if(!['map','review','generation'].includes(tab))return;state.leftTab=tab;document.querySelectorAll('.left-tab').forEach(button=>{const selected=button.dataset.tab===tab;button.setAttribute('aria-selected',selected?'true':'false')});document.querySelectorAll('.left-tab-panel').forEach(panel=>{panel.hidden=panel.dataset.panel!==tab})}
function updateRightButton(){const collapsed=$('app').classList.contains('audit-collapsed');$('toggleRight').textContent=collapsed?'打开审核面板':'收起审核面板';$('toggleRight').setAttribute('aria-expanded',collapsed?'false':'true')}
function pointer(e){const r=map.getBoundingClientRect();return {x:e.clientX-r.left,y:e.clientY-r.top}}
map.addEventListener('pointerdown',e=>{const p=pointer(e);state.drag={start:p,last:p,pan:e.shiftKey||e.button!==0};map.setPointerCapture(e.pointerId)});
map.addEventListener('pointermove',e=>{if(!state.drag)return;const p=pointer(e);if(state.drag.pan){state.panX+=p.x-state.drag.last.x;state.panY+=p.y-state.drag.last.y;state.drag.last=p;draw()}else{state.drag.current=p;drawDrag(p)}});
map.addEventListener('pointerup',e=>{if(!state.drag)return;const d=state.drag,p=pointer(e);state.drag=null;if(d.pan){draw();return}if(Math.hypot(p.x-d.start.x,p.y-d.start.y)<5){const historical=reviewAt(screenToWorld(p.x,p.y));if(historical){loadReview(historical);draw();return}}const a=screenToWorld(d.start.x,d.start.y),b=screenToWorld(p.x,p.y);setSelection([(a.x+b.x)/2,(a.y+b.y)/2])});
function drawDrag(p){draw();if(!state.drag||state.drag.pan)return;const a=state.drag.start,b=p;mctx.save();mctx.strokeStyle='#e8b85e';mctx.setLineDash([6,4]);mctx.strokeRect(Math.min(a.x,b.x),Math.min(a.y,b.y),Math.abs(a.x-b.x),Math.abs(a.y-b.y));mctx.restore()}
map.addEventListener('wheel',e=>{e.preventDefault();const before=screenToWorld(...Object.values(pointer(e)));const factor=Math.exp(-e.deltaY*.001);state.scale=Math.max(.01,Math.min(1e6,state.scale*factor));const after=screenToWorld(...Object.values(pointer(e)));state.panX+=(after.x-before.x)*state.scale;state.panY-=(after.y-before.y)*state.scale;draw()},{passive:false});
$('sourceCategory').addEventListener('change',e=>{state.sourceCategory=e.target.value;renderSourceOptions();const id=$('source').value;if(id)loadSource(id).catch(x=>setStatus('加载失败：'+x.message))});
$('source').addEventListener('change',e=>loadSource(e.target.value).catch(x=>setStatus('加载失败：'+x.message)));
$('random').addEventListener('click',()=>randomCandidate().catch(x=>setStatus('随机候选失败：'+x.message)));
 $('clear').addEventListener('click',()=>{state.selection=null;state.crop=null;state.fit=null;state.displayMode='raw';state.reviewRecord=null;renderReviewHistory();renderCropStats();renderCrop3d();draw()});
 $('fit').addEventListener('click',fit);$('size').addEventListener('change',()=>{if(state.selection){state.reviewRecord=null;state.selection.size_m=+$('size').value;state.fit=null;renderReviewHistory();updateReviewControls();requestCrop().catch(x=>setStatus('加载失败：'+x.message))}});$('yaw').addEventListener('change',()=>{if(state.selection){state.reviewRecord=null;state.selection.yaw_deg=+$('yaw').value;state.fit=null;renderReviewHistory();updateReviewControls();requestCrop().catch(x=>setStatus('加载失败：'+x.message))}});$('split').addEventListener('change',updateReviewControls);$('fillSlot').addEventListener('change',e=>{state.fillSlotKey=e.target.value;const slot=OPEN_SLOTS.find(item=>item.key===state.fillSlotKey);if(slot)$('split').value=slot.split;updateReviewControls()});$('colorMode').addEventListener('change',()=>{draw();renderCrop3d()});$('pointSize').addEventListener('change',()=>{draw();renderCrop3d()});window.addEventListener('resize',()=>{draw();if(state.three)state.three.render()});
$('runFit').addEventListener('click',()=>runFit().catch(x=>setStatus('拟合失败：'+x.message)));
document.querySelectorAll('.left-tab').forEach(button=>button.addEventListener('click',()=>setLeftTab(button.dataset.tab)));
$('toggleRight').addEventListener('click',()=>{$('app').classList.toggle('audit-collapsed');updateRightButton()});
 $('approve').addEventListener('click',()=>review('approve').catch(x=>setStatus('保存失败：'+x.message)));$('reject').addEventListener('click',()=>review('reject').catch(x=>setStatus('保存失败：'+x.message)));$('approveNext').addEventListener('click',()=>review('approve',true).catch(x=>setStatus('保存失败：'+x.message)));$('rejectNext').addEventListener('click',()=>review('reject',true).catch(x=>setStatus('保存失败：'+x.message)));$('updateReviewSplit').addEventListener('click',()=>updateReviewSplit().catch(x=>setStatus('更新归属失败：'+x.message)));
$('showAllReviews').addEventListener('change',()=>{renderReviewHistory();draw()});
 $('selectAllReviews').addEventListener('change',e=>{for(const {record} of visibleSelectableReviews()){if(e.target.checked)state.selectedReviewTimestamps.add(record.timestamp_utc);else state.selectedReviewTimestamps.delete(record.timestamp_utc)};syncBulkReviewControls();renderReviewHistory()});$('reviewHistory').addEventListener('change',e=>{const checkbox=e.target.closest('.history-select');if(!checkbox||!state.source)return;const record=state.source.review_records[Number(checkbox.dataset.reviewIndex)];if(!record||!reviewIsSelectable(record))return;if(checkbox.checked)state.selectedReviewTimestamps.add(record.timestamp_utc);else state.selectedReviewTimestamps.delete(record.timestamp_utc);syncBulkReviewControls()});$('reviewHistory').addEventListener('click',e=>{if(e.target.closest('.history-select'))return;const button=e.target.closest('.history-open');if(!button||!state.source)return;loadReview(state.source.review_records[Number(button.dataset.reviewIndex)])});$('bulkUpdateReview').addEventListener('click',()=>updateSelectedReviewSplits().catch(x=>setStatus('批量更新归属失败：'+x.message)));
 $('confirmDomain').addEventListener('click',()=>confirmDomain().catch(x=>setStatus('切换审核记录失败：'+x.message)));$('startDatasetViewer').addEventListener('click',()=>startDatasetViewer().catch(x=>setStatus('启动结果查看器失败：'+x.message)));$('openDatasetViewer').addEventListener('click',openDatasetViewer);$('stopDatasetViewer').addEventListener('click',()=>stopDatasetViewer().catch(x=>setStatus('关闭结果查看器失败：'+x.message)));
 $('importMap').addEventListener('click',()=>$('mapFiles').click());$('mapFiles').addEventListener('change',()=>importMaps().catch(e=>setStatus('导入原始地图失败：'+e.message)));$('chooseReview').addEventListener('click',()=>$('reviewFile').click());$('reviewFile').addEventListener('change',()=>importReview().catch(e=>setStatus('导入标注备份失败：'+e.message)));$('loadRoot').addEventListener('click',()=>openPicker('loadPath','directory').catch(e=>setStatus('选择地图文件夹失败：'+e.message)));$('refreshCatalog').addEventListener('click',()=>refreshCatalog(state.sourceId).catch(e=>setStatus('重新读取失败：'+e.message)));$('switchReview').addEventListener('click',()=>switchReview().catch(e=>setStatus('加载标注记录失败：'+e.message)));$('exportCanonical').addEventListener('click',()=>exportCanonical().catch(e=>setStatus('生成训练用地图失败：'+e.message)));$('generateDataset').addEventListener('click',()=>generateDataset().catch(e=>setStatus('生成最终训练数据失败：'+e.message)));$('stopJob').addEventListener('click',()=>postJson('/api/stop-job',{}).then(refreshJobs).catch(e=>setStatus('停止生成失败：'+e.message)));
 $('browseReviewPath').addEventListener('click',()=>openPicker('reviewPath','file',['.jsonl','.ndjson']).catch(e=>setStatus('选择标注记录失败：'+e.message)));$('browseCanonicalDir').addEventListener('click',()=>openPicker('canonicalDir','directory').catch(e=>setStatus('选择保存文件夹失败：'+e.message)));$('browseManifest').addEventListener('click',()=>openPicker('manifestPath','file',['.json']).catch(e=>setStatus('选择程序清单失败：'+e.message)));$('browseDatasetDir').addEventListener('click',()=>openPicker('datasetDir','directory').catch(e=>setStatus('选择保存文件夹失败：'+e.message)));$('closePicker').addEventListener('click',closePicker);$('pickerUp').addEventListener('click',()=>{if(pickerState.parent)browsePicker(pickerState.parent).catch(e=>setStatus('打开文件夹失败：'+e.message))});$('pickerGo').addEventListener('click',()=>browsePicker($('pickerPath').value).catch(e=>setStatus('打开文件夹失败：'+e.message)));$('usePickerPath').addEventListener('click',choosePickerPath);$('usePickerDirectory').addEventListener('click',choosePickerDirectory);$('pathPicker').addEventListener('click',e=>{if(e.target===$('pathPicker'))closePicker()});
const drawCurrentSelection=drawSelection;drawSelection=()=>{drawReviewMarkers();drawCurrentSelection()};
function initialiseWorkflow(){const domain=CONFIG.default_domain||'terrain';$('domain').value=domain;$('site').value='imported';$('trainSiteId').value=`${domain}_train`;$('valSiteId').value=`${domain}_val`;setLeftTab('map');updateRightButton();updateReviewControls();updateWorkflowLinks({sources:SOURCES,review_file:CONFIG.review_file,default_domain:domain,dataset_viewer_url:CONFIG.dataset_viewer_url,workspace_root:CONFIG.workspace_root});renderSourceOptions();refreshJobs();refreshCatalog().catch(e=>setStatus('初始化失败：'+e.message))}
  const baseInitialiseWorkflow=initialiseWorkflow;initialiseWorkflow=()=>{baseInitialiseWorkflow();rememberWorkflowPaths();confirmedDomain=$('domain').value.trim();refreshViewer()};initialiseWorkflow();
</script>
</body></html>'''


def read_json_request(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def make_handler(catalog, html, workspace, default_domain,
                 dataset_viewer_url, jobs, viewer):
    class Handler(BaseHTTPRequestHandler):
        def send_payload(self, payload, content_type, status=200, headers=None):
            body = payload if isinstance(payload, bytes) else payload.encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # Browsers can cancel a large preview while changing maps.
                return

        def send_json(self, value, status=200):
            self.send_payload(json.dumps(value, ensure_ascii=False,
                                         separators=(",", ":")),
                             "application/json; charset=utf-8", status)

        def workflow_index(self):
            result = catalog.index()
            result.update({
                "workspace_root": str(workspace),
                "default_domain": default_domain,
                "dataset_viewer_url": dataset_viewer_url,
                "jobs": jobs.snapshots(),
                "viewer": viewer.status(),
            })
            return result

        def start_canonical_export(self, body):
            domain = safe_component(body.get("domain", default_domain), "domain")
            output_dir = resolve_workspace_path(body.get("output_dir", ""), workspace)
            train_site_id = safe_component(
                body.get("train_site_id", f"{domain}_train"), "train_site_id")
            val_site_id = safe_component(
                body.get("val_site_id", f"{domain}_val"), "val_site_id")
            train_profile = body.get("train_source_profile", "als")
            val_profile = body.get("val_source_profile", "als")
            if train_profile not in {"als", "uls"} or val_profile not in {"als", "uls"}:
                raise ValueError("source profile must be als or uls")
            if not str(body.get("output_dir", "")).strip():
                raise ValueError("output_dir is required")
            command = [
                sys.executable,
                str(workspace / "src/uneven_planner/plan_manager/scripts"
                    / "build_approved_canonical_maps.py"),
                str(catalog.review_store.path),
                str(output_dir),
                "--domain", domain,
                "--train-site-id", train_site_id,
                "--val-site-id", val_site_id,
                "--train-source-profile", train_profile,
                "--val-source-profile", val_profile,
            ]
            if bool(body.get("overwrite", False)):
                command.append("--overwrite")
            log_path = Path(str(output_dir) + ".logs") / "web-canonical-export.log"
            return jobs.start("canonical_export", command, workspace, log_path)

        def start_dataset_generation(self, body):
            manifest = resolve_workspace_path(body.get("manifest_path", ""), workspace)
            output_dir = resolve_workspace_path(body.get("output_dir", ""), workspace)
            if not manifest.is_file():
                raise FileNotFoundError(f"canonical manifest not found: {manifest}")
            if not str(body.get("output_dir", "")).strip():
                raise ValueError("output_dir is required")
            values = {
                "train_paths": int(body.get("train_paths", 100)),
                "val_paths": int(body.get("val_paths", 10)),
                "workers": int(body.get("workers", 4)),
                "ros_port": int(body.get("ros_port", 11412)),
            }
            if any(value <= 0 for value in values.values()):
                raise ValueError("generation counts, workers and ROS port must be positive")
            command = [
                "bash",
                str(workspace / "src/uneven_planner/plan_manager/scripts"
                    / "generate_approved_canonical_dataset.sh"),
                str(manifest),
                str(output_dir),
                str(values["train_paths"]),
                str(values["val_paths"]),
                str(values["workers"]),
                str(values["ros_port"]),
            ]
            log_path = Path(str(output_dir) + ".logs") / "web-generation.log"
            return jobs.start("dataset_generation", command, workspace, log_path)

        def start_dataset_viewer(self, body):
            dataset_root = resolve_workspace_path(
                body.get("dataset_dir", ""), workspace)
            if not str(body.get("dataset_dir", "")).strip():
                raise ValueError("dataset_dir is required")
            if dataset_root.exists() and not dataset_root.is_dir():
                raise ValueError(f"dataset path is not a directory: {dataset_root}")
            canonical_value = str(body.get("canonical_dir", "")).strip()
            canonical_root = (
                resolve_workspace_path(canonical_value, workspace)
                if canonical_value else None)
            if canonical_root and canonical_root.exists() and not canonical_root.is_dir():
                raise ValueError(
                    f"canonical path is not a directory: {canonical_root}")
            domain = safe_component(body.get("domain", default_domain), "domain")
            parsed = urlparse(dataset_viewer_url)
            if parsed.scheme != "http" or not parsed.hostname:
                raise ValueError(
                    "dataset viewer URL must be an http URL with a host")
            try:
                port = parsed.port or 80
            except ValueError as exc:
                raise ValueError("invalid dataset viewer port") from exc
            command = [
                sys.executable,
                str(workspace / "src/uneven_planner/plan_manager/scripts"
                    / "serve_terrain_dataset_viewer.py"),
                str(dataset_root),
                "--host", parsed.hostname,
                "--port", str(port),
            ]
            if canonical_root:
                command.extend(["--quick-canonical-root", str(canonical_root)])
            command.extend(["--quick-final-root", str(dataset_root),
                            "--domain", domain])
            log_path = Path(str(dataset_root) + ".logs") / "web-viewer.log"
            return viewer.start(command, workspace, dataset_root, log_path)

        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                query = parse_qs(parsed.query)
                if parsed.path == "/":
                    self.send_payload(html, "text/html; charset=utf-8")
                elif parsed.path == "/api/index":
                    self.send_json(self.workflow_index())
                elif parsed.path == "/api/jobs":
                    self.send_json(jobs.snapshots())
                elif parsed.path == "/api/viewer":
                    self.send_json(viewer.status())
                elif parsed.path == "/api/browse":
                    self.send_json(browse_directory(
                        query.get("path", [""])[0], workspace))
                elif parsed.path == "/api/review-file":
                    data = (catalog.review_store.path.read_bytes()
                            if catalog.review_store.path.is_file() else b"")
                    self.send_payload(
                        data, "application/x-ndjson; charset=utf-8",
                        headers={
                            "Content-Disposition": (
                                f'attachment; filename="{catalog.review_store.path.name}"')
                        })
                elif parsed.path == "/api/source":
                    self.send_json(catalog.source_record(query.get("id", [""])[0]))
                elif parsed.path == "/api/random":
                    self.send_json(catalog.random_candidate(
                        query.get("source", [""])[0],
                        float(query.get("size", [20])[0]),
                        query.get("random_yaw", ["0"])[0] == "1"))
                elif parsed.path == "/api/crop":
                    self.send_json(catalog.source(query.get("source", [""])[0]).crop(
                        float(query.get("cx", [0])[0]),
                        float(query.get("cy", [0])[0]),
                        float(query.get("size", [20])[0]),
                        float(query.get("yaw", [0])[0]),
                        catalog.max_crop_points))
                else:
                    self.send_json({"error": "not found"}, 404)
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, 404)
            except (OSError, ValueError, KeyError, RuntimeError,
                    laspy.errors.LaspyException) as exc:
                self.send_json({"error": str(exc)}, 500)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path in {"/api/import", "/api/import-review"}:
                try:
                    content_type = self.headers.get("Content-Type", "")
                    if not content_type.startswith("multipart/form-data"):
                        raise ValueError("file upload requires multipart/form-data")
                    form = cgi.FieldStorage(
                        fp=self.rfile,
                        headers=self.headers,
                        environ={
                            "REQUEST_METHOD": "POST",
                            "CONTENT_TYPE": content_type,
                            "CONTENT_LENGTH": self.headers.get(
                                "Content-Length", "0"),
                        },
                        keep_blank_values=True,
                    )
                    if parsed.path == "/api/import-review":
                        result = import_uploaded_review(form, workspace)
                        catalog.review_store.switch_path(
                            Path(result["review_file"]))
                    else:
                        result = register_uploaded_maps(form, workspace)
                        result["source_ids"] = catalog.add_input_path(
                            result["review_input"])
                    self.send_json(result)
                except (OSError, ValueError, KeyError) as exc:
                    self.send_json({"error": str(exc)}, 400)
                return
            if parsed.path in {"/api/load-root", "/api/load-review", "/api/reload",
                               "/api/export-canonical", "/api/generate-dataset",
                               "/api/stop-job", "/api/start-viewer", "/api/stop-viewer",
                               "/api/review", "/api/update-review",
                               "/api/update-review-batch", "/api/fit"}:
                pass
            else:
                self.send_json({"error": "not found"}, 404)
                return
            try:
                if parsed.path == "/api/reload":
                    catalog.review_store.reload()
                    catalog.reload()
                    self.send_json(self.workflow_index())
                    return
                if parsed.path == "/api/stop-job":
                    self.send_json(jobs.stop() or {"running": False})
                    return
                if parsed.path == "/api/stop-viewer":
                    self.send_json(viewer.stop())
                    return
                record = read_json_request(self)
                if parsed.path == "/api/load-root":
                    if not str(record.get("path", "")).strip():
                        raise ValueError("path is required")
                    path = resolve_workspace_path(record.get("path", ""), workspace)
                    source_ids = catalog.add_input_path(path)
                    self.send_json({
                        "root": str(path),
                        "source_ids": source_ids,
                    })
                    return
                if parsed.path == "/api/load-review":
                    path = resolve_workspace_path(record.get("path", ""), workspace)
                    if path.exists() and not path.is_file():
                        raise ValueError(f"review path is not a file: {path}")
                    catalog.review_store.switch_path(path)
                    self.send_json(self.workflow_index())
                    return
                if parsed.path == "/api/export-canonical":
                    self.send_json(self.start_canonical_export(record))
                    return
                if parsed.path == "/api/generate-dataset":
                    self.send_json(self.start_dataset_generation(record))
                    return
                if parsed.path == "/api/start-viewer":
                    self.send_json(self.start_dataset_viewer(record))
                    return
                source_id = record.get("source_id", "")
                if source_id not in catalog.sources:
                    raise ValueError(f"Unknown source: {source_id}")
                if parsed.path == "/api/update-review":
                    timestamp_utc = record.get("timestamp_utc")
                    if not isinstance(timestamp_utc, str) or not timestamp_utc:
                        raise ValueError("timestamp_utc is required")
                    split = record.get("split")
                    if split not in {"train", "val"}:
                        raise ValueError("split must be train or val")
                    self.send_json(catalog.review_store.update_split(
                        source_id, timestamp_utc, split))
                    return
                if parsed.path == "/api/update-review-batch":
                    timestamps_utc = record.get("timestamps_utc")
                    if (not isinstance(timestamps_utc, list)
                            or not timestamps_utc
                            or any(not isinstance(value, str) or not value
                                   for value in timestamps_utc)):
                        raise ValueError("timestamps_utc must be a non-empty list")
                    split = record.get("split")
                    if split not in {"train", "val"}:
                        raise ValueError("split must be train or val")
                    self.send_json(catalog.review_store.update_splits(
                        source_id, timestamps_utc, split))
                    return
                if parsed.path == "/api/fit":
                    center = record.get("center_xy")
                    if not isinstance(center, list) or len(center) != 2:
                        raise ValueError("center_xy must contain two values")
                    self.send_json(catalog.source(source_id).fit(
                        float(center[0]), float(center[1]),
                        float(record.get("size_m", 20.0)),
                        float(record.get("yaw_deg", 0.0))))
                    return
                if record.get("decision") not in {"approve", "reject"}:
                    raise ValueError("decision must be approve or reject")
                split = record.get("split", "train")
                if split not in {"train", "val"}:
                    raise ValueError("split must be train or val")
                slot = fill_slot(record)
                if slot is not None:
                    if record["decision"] != "approve":
                        raise ValueError("only an approval can fill a map slot")
                    if slot not in {
                            (item["split"], item["map_index"])
                            for item in catalog.review_store.open_slots()}:
                        raise ValueError("selected map slot is no longer open")
                    if split != slot[0]:
                        raise ValueError(
                            "approval split must match the selected map slot")
                    record["fills_slot"] = {
                        "split": slot[0], "map_index": slot[1]}
                record["split"] = split
                record["timestamp_utc"] = datetime.datetime.now(
                    datetime.timezone.utc).isoformat()
                record["source_path"] = str(catalog.source(source_id).path)
                counts = catalog.review_store.append(record)
                catalog_summary = catalog.index()
                self.send_json({
                    "review_counts": counts,
                    "review_split_counts": catalog.review_store.split_counts(source_id),
                    "review_records": catalog.review_store.records_for(source_id),
                    "open_slots": catalog_summary["open_slots"],
                    "sources": catalog_summary["sources"],
                })
            except (OSError, ValueError, KeyError, ImportError, RuntimeError,
                    json.JSONDecodeError, laspy.errors.LaspyException) as exc:
                self.send_json({"error": str(exc)}, 400)

        def log_message(self, format_string, *args):
            return

    return Handler


def main():
    args = parse_args()
    root = workspace_root()
    missing_paths = [path for path in args.input_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(missing_paths[0])
    if args.max_display_points <= 0 or args.max_crop_points <= 0:
        raise ValueError("point limits must be positive")
    review_store = ReviewStore(resolve_workspace_path(args.review_file, root))
    catalog = Catalog(
        args.input_paths, args.max_display_points, args.max_crop_points,
        review_store)
    sources = catalog.index()["sources"]
    config = {
        "workspace_root": str(root),
        "review_file": str(review_store.path),
        "default_domain": args.domain,
        "dataset_viewer_url": args.dataset_viewer_url,
    }
    html = (HTML_TEMPLATE.replace("__TITLE__", args.title)
            .replace("__SOURCES__", json.dumps(sources, ensure_ascii=False))
            .replace("__CONFIG__", json.dumps(config, ensure_ascii=False)))
    jobs = JobManager()
    viewer = ViewerManager(args.dataset_viewer_url)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(
            catalog, html, root, args.domain, args.dataset_viewer_url, jobs,
            viewer))
    print(json.dumps({
        "url": f"http://{args.host}:{args.port}/",
        "sources": len(sources),
        "review_file": str(review_store.path),
    }, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.stop()
        server.server_close()


if __name__ == "__main__":
    main()
