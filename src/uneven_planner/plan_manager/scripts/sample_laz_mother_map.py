#!/usr/bin/env python3
"""Sample and quality-gate metric 20 m scenes directly from a LAS/LAZ mother map.

The source is loaded once, but no fixed crop list is required. Candidate
centres and yaw angles are sampled deterministically from raw source support.
The continuous 100 x 100 grid is fitted from a geometry-only lower envelope,
while the retained planner PCD contains that fitted surface together with all
finite raw XYZ returns in the crop. No LAS class filter or obstacle layer is
written. Only quality-passing scenes are retained; every attempted location
and rejection reason is recorded in a manifest.
"""

import argparse
import datetime
import glob
import json
import math
import os
import sys

import laspy
import numpy as np
from prepare_laz_terrain_map import (
    fit_grid,
    fit_yrf_ground_grid,
    geometry_ground_candidates,
    measure_above_surface_coverage,
    measure_classified_above_surface_coverage,
    retain_raw_returns,
    resample_completed_surface,
    save_preview,
    write_binary_pcd,
)
from terrain_map_quality import evaluate


def file_record(path):
    """Record enough provenance to identify a local input without hashing it."""
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_laz", help="Raw LAS/LAZ mother map")
    parser.add_argument("output_dir", help="Retained scene and manifest directory")
    parser.add_argument("--accepted", type=int, default=5,
                        help="Number of quality-passing scenes to retain")
    parser.add_argument("--max-attempts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue an existing sampling manifest in output_dir")
    parser.add_argument("--size", type=float, default=20.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--planner-surface-resolution", type=float, default=0.05,
                        help="Dense fitted-surface spacing used by UnevenMap")
    parser.add_argument("--raw-below-surface-tolerance", type=float, default=1.0,
                        help="Discard raw returns more than this far below the fitted surface (m)")
    parser.add_argument("--raw-above-surface-tolerance", type=float, default=50.0,
                        help="Discard raw returns more than this far above the fitted surface (m)")
    parser.add_argument(
        "--source-profile", choices=("uls", "als"), default="uls",
        help="Acquisition-density profile; controls explicit fitting defaults")
    parser.add_argument("--fit-radius", type=float, default=None,
                        help="Local-plane radius; defaults to 0.35 m for ULS and 0.9 m for ALS")
    parser.add_argument("--surface-cell-size", type=float, default=None,
                        help="Coarse XY cell for geometry-only lower-envelope ground extraction")
    parser.add_argument("--ground-band-below", type=float, default=0.25,
                        help="Ground candidate tolerance below the lower envelope (m)")
    parser.add_argument("--ground-band-above", type=float, default=0.35,
                        help="Ground candidate tolerance above the lower envelope (m)")
    parser.add_argument("--envelope-outlier", type=float, default=0.75,
                        help="Local lower-envelope outlier threshold (m)")
    parser.add_argument("--direct-fit-min-points", type=int, default=None,
                        help="Minimum points for a directly observed local fit")
    parser.add_argument("--min-density", type=float, default=None)
    parser.add_argument("--coverage-resolution", type=float, default=None,
                        help="Cell size for checking source support before fitting")
    parser.add_argument("--min-coverage", type=float, default=0.97)
    parser.add_argument("--min-center-separation", type=float, default=20.0)
    parser.add_argument(
        "--center-sampling", choices=("uniform_support", "surface_point"),
        default="uniform_support",
        help="Sample uniformly from occupied source-area cells (default) or density-weighted from a surface point")
    parser.add_argument(
        "--center-sampling-resolution", type=float, default=1.0,
        help="Cell size used to sample source support area without point-density bias")
    parser.add_argument(
        "--accepted-grades", default="easy,medium,hard",
        help="Comma-separated geometry grades retained; extreme is excluded by default")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--license", default="")
    parser.add_argument("--crs", default="")
    parser.add_argument("--site-id", default="",
                        help="Stable survey/site identifier used for split-leakage audits")
    parser.add_argument("--domain", default="",
                        help="Source-domain label such as desert, forest, hill, snow, or volcano")
    args = parser.parse_args()
    profile_defaults = {
        "uls": {
            "fit_radius": 0.35,
            "direct_fit_min_points": 8,
            "min_density": 40.0,
            "coverage_resolution": 0.2,
            "surface_cell_size": 0.5,
        },
        "als": {
            "fit_radius": 0.9,
            "direct_fit_min_points": 5,
            "min_density": 4.0,
            "coverage_resolution": 1.0,
            "surface_cell_size": 1.0,
        },
    }[args.source_profile]
    for name, value in profile_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.accepted_grades = [
        value.strip() for value in args.accepted_grades.split(",")
        if value.strip()
    ]
    allowed_grades = {"easy", "medium", "hard", "extreme"}
    unknown_grades = sorted(set(args.accepted_grades) - allowed_grades)
    if not args.accepted_grades or unknown_grades:
        raise ValueError(
            "accepted-grades must contain easy, medium, hard, or extreme; "
            f"unknown={unknown_grades}")
    return args


def load_surface_points(path, return_classification=False):
    """Load all finite XYZ returns and optionally aligned LAS classes.

    Runtime ROS clouds do not carry LAS classification, so the release sampler
    must see the same raw geometry. Classification counts are kept in the
    manifest for source description and offline diagnostics, never for
    selecting the retained raw returns. When requested, the per-return labels
    are returned in the same order as the finite XYZ array for crop-level
    diagnostics.
    """
    chunks = []
    classification_chunks = []
    class_counts = np.zeros(256, dtype=np.int64)
    with laspy.open(path) as reader:
        source_count = int(reader.header.point_count)
        source_bounds = {
            "minimum": reader.header.mins.astype(float).tolist(),
            "maximum": reader.header.maxs.astype(float).tolist(),
        }
        for points in reader.chunk_iterator(2_000_000):
            # LAS classification is an unsigned byte.  Keep the aligned
            # source label array compact; crop-level metrics widen only the
            # small selected slice when they need integer operations.
            classification = np.asarray(points.classification, dtype=np.uint8)
            class_counts += np.bincount(classification, minlength=256)
            xyz = np.column_stack((
                np.asarray(points.x),
                np.asarray(points.y),
                np.asarray(points.z),
            ))
            finite = np.isfinite(xyz).all(axis=1)
            if np.any(finite):
                chunks.append(xyz[finite])
                classification_chunks.append(classification[finite])
    if not chunks:
        raise RuntimeError("Mother map contains no finite XYZ points")
    points = np.concatenate(chunks)
    metadata = (
        points, source_count, source_bounds,
        {str(i): int(value) for i, value in enumerate(class_counts) if value})
    if return_classification:
        return metadata + (np.concatenate(classification_chunks),)
    return metadata


def select_local_points(all_ground, center, yaw, size, margin):
    """Return exact and padded crop points in the rotated local frame."""
    selection_half = 0.5 * size + margin
    source_extent = selection_half * (abs(math.cos(yaw)) + abs(math.sin(yaw)))
    axis_mask = (
        (np.abs(all_ground[:, 0] - center[0]) < source_extent)
        & (np.abs(all_ground[:, 1] - center[1]) < source_extent)
    )
    nearby = all_ground[axis_mask]
    if len(nearby) == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    dx = nearby[:, 0] - center[0]
    dy = nearby[:, 1] - center[1]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    local = nearby.copy()
    local[:, 0] = cos_yaw * dx + sin_yaw * dy
    local[:, 1] = -sin_yaw * dx + cos_yaw * dy
    padded = ((np.abs(local[:, 0]) < selection_half)
              & (np.abs(local[:, 1]) < selection_half))
    exact = ((np.abs(local[:, 0]) < 0.5 * size)
             & (np.abs(local[:, 1]) < 0.5 * size))
    return local[exact], local[padded]


def grid_coverage(points, size, resolution):
    cells = int(round(size / resolution))
    half = 0.5 * size
    ix = np.floor((points[:, 0] + half) / resolution).astype(np.int64)
    iy = np.floor((points[:, 1] + half) / resolution).astype(np.int64)
    inside = (ix >= 0) & (ix < cells) & (iy >= 0) & (iy < cells)
    occupied = np.unique(iy[inside] * cells + ix[inside])
    return len(occupied) / float(cells * cells)


def build_surface(exact, padded, args):
    fit_method = getattr(args, "fit_method", "local_plane")
    if fit_method == "yrf_ground":
        (local_x, local_y, elevation, normals, valid, observed, neighbors,
         split_diagnostics) = fit_yrf_ground_grid(
            padded,
            args.size,
            args.resolution,
            coarse_resolution=args.yrf_coarse_resolution,
            voxel_size=args.yrf_voxel_size,
            ground_below=args.ground_band_below,
            ground_above=args.ground_band_above,
            envelope_outlier=args.envelope_outlier,
            lower_envelope_filter_size=args.yrf_lower_envelope_filter_size,
        )
        ground_candidate_count = split_diagnostics["ground_candidate_count"]
    else:
        ground_points, _, _, split_diagnostics = geometry_ground_candidates(
            padded,
            args.size,
            args.fit_radius,
            cell_size=args.surface_cell_size,
            ground_below=args.ground_band_below,
            ground_above=args.ground_band_above,
            envelope_outlier=args.envelope_outlier,
        )
        local_x, local_y, elevation, normals, valid, observed, neighbors = fit_grid(
            ground_points, 0.0, 0.0, args.size, args.resolution,
            args.fit_radius, args.direct_fit_min_points)
        ground_candidate_count = len(ground_points)

    # UnevenMap estimates a local covariance directly from the PCD. Keep a
    # dense fitted surface for stable support, and append every raw return in
    # the crop so above-ground geometry is not discarded before runtime mask
    # handling exists.
    dense_xyz, dense_normals = resample_completed_surface(
        elevation, normals, valid, args.size, args.resolution,
        args.planner_surface_resolution)

    raw_xyz = retain_raw_returns(
        np.asarray(exact, dtype=np.float64), elevation, args.size,
        args.resolution, args.raw_below_surface_tolerance,
        args.raw_above_surface_tolerance)
    raw_normals = np.zeros_like(raw_xyz)
    vertical_origin = float(min(
        np.min(elevation[valid]), np.min(dense_xyz[:, 2])))
    elevation -= vertical_origin
    dense_xyz[:, 2] -= vertical_origin
    raw_xyz[:, 2] -= vertical_origin
    planner_xyz = np.vstack((dense_xyz, raw_xyz))
    planner_normals = np.vstack((dense_normals, raw_normals))
    return {
        "local_x": local_x,
        "local_y": local_y,
        "elevation": elevation,
        "normals": normals,
        "valid": valid,
        "observed": observed,
        "neighbors": neighbors,
        "dense_xyz": dense_xyz,
        "dense_normals": dense_normals,
        "raw_xyz": raw_xyz,
        "raw_normals": raw_normals,
        "planner_xyz": planner_xyz,
        "planner_normals": planner_normals,
        "vertical_origin": vertical_origin,
        "consistent_fraction": 1.0,
        "ground_candidate_count": int(ground_candidate_count),
        "split_diagnostics": split_diagnostics,
        "fit_method": fit_method,
    }


def metadata_for(surface, exact_count, support_coverage, center, yaw,
                 source_info, args):
    valid = surface["valid"]
    normals = surface["normals"]
    slope = np.degrees(np.arccos(np.clip(normals[:, :, 2], -1.0, 1.0)))
    fit_method = surface.get("fit_method", "local_plane")
    processing = {
        "source_profile": args.source_profile,
        "surface_source": "all finite XYZ returns; no LAS class filtering; gross below-surface outliers removed",
        "sampling": (
            "uniform occupied source-area cell with within-cell jitter and random yaw"
            if args.center_sampling == "uniform_support"
            else "random selected-surface point centre and random yaw"),
        "center_sampling_resolution_m": args.center_sampling_resolution,
        "surface_cell_size_m": args.surface_cell_size,
        "ground_band_below_m": args.ground_band_below,
        "ground_band_above_m": args.ground_band_above,
        "envelope_outlier_m": args.envelope_outlier,
        "fit_radius_m": args.fit_radius,
        "direct_fit_minimum_points": args.direct_fit_min_points,
        "planner_surface_resolution_m": args.planner_surface_resolution,
        "raw_below_surface_tolerance_m": args.raw_below_surface_tolerance,
        "raw_above_surface_tolerance_m": args.raw_above_surface_tolerance,
        "axis_scaling": "none",
        "xy_output": "translated and rotated only",
        "z_output": "source elevation minus retained patch minimum",
        "planner_pcd": "dense completed fitted surface plus supported raw XYZ returns",
    }
    if fit_method == "yrf_ground":
        processing.update({
            "ground_fit_support": (
                "YRF 0.2m voxel centroids + 0.4m lower envelope + 3x3 "
                "upper-median reference + ground-band median"),
            "fit": "YRF-compatible coarse ground reference and cubic elevation interpolation",
            "yrf_voxel_size_m": args.yrf_voxel_size,
            "yrf_coarse_resolution_m": args.yrf_coarse_resolution,
            "yrf_lower_envelope_filter_size": (
                args.yrf_lower_envelope_filter_size),
            "gap_completion": (
                "coarse unsupported-cell completion followed by cubic elevation interpolation"),
        })
    else:
        processing.update({
            "ground_fit_support": "coarse lower envelope + local median outlier removal + height band",
            "fit": "distance-weighted local plane with non-rejecting MAD refinement on geometry-only ground candidates",
            "gap_completion": (
                "harmonic elevation fill followed by finite-difference normals"),
        })
    return {
        "domain": args.domain,
        "site_id": args.site_id,
        "source_file": os.path.abspath(args.input_laz),
        "source_url": args.source_url,
        "license": args.license,
        "source_crs": args.crs,
        "source_point_count": source_info[0],
        "source_bounds": source_info[1],
        "source_class_counts": source_info[2],
        "source_center_xy": [float(center[0]), float(center[1])],
        "crop_yaw_deg": float(math.degrees(yaw)),
        "patch_size_m": args.size,
        "resolution_m": args.resolution,
        "grid_shape": list(surface["elevation"].shape),
        "source_surface": "all_finite_XYZ_no_LAS_class_filter",
        "surface_fit_method": fit_method,
        "source_raw_points_in_patch": int(exact_count),
        "raw_returns_retained_in_planner_pcd": int(len(surface["raw_xyz"])),
        "source_raw_density_points_per_m2": float(
            exact_count / (args.size ** 2)),
        "ground_candidate_points_in_patch": int(
            surface["ground_candidate_count"]),
        "ground_candidate_density_points_per_m2": float(
            surface["ground_candidate_count"] / (args.size ** 2)),
        "source_support_coverage": float(support_coverage),
        "source_support_resolution_m": float(args.coverage_resolution),
        "planner_point_count": int(len(surface["planner_xyz"])),
        "planner_point_density_points_per_m2": float(
            len(surface["planner_xyz"]) / (args.size ** 2)),
        "planner_surface_consistent_fraction": float(surface["consistent_fraction"]),
        "valid_fraction": float(valid.mean()),
        "observed_fraction": float(surface["observed"].mean()),
        "vertical_origin_m": surface["vertical_origin"],
        "elevation_range_m": [
            float(np.min(surface["elevation"])),
            float(np.max(surface["elevation"])),
        ],
        "slope_degrees": {
            "median": float(np.median(slope[valid])),
            "p95": float(np.quantile(slope[valid], 0.95)),
            "maximum": float(np.max(slope[valid])),
        },
        "processing": processing,
    }


def save_npz(path, surface, args, center, yaw):
    normals = surface["normals"]
    np.savez_compressed(
        path,
        elevation=surface["elevation"].astype(np.float32),
        normal_x=normals[:, :, 0].astype(np.float32),
        normal_y=normals[:, :, 1].astype(np.float32),
        normal_z=normals[:, :, 2].astype(np.float32),
        valid_mask=surface["valid"],
        observed_mask=surface["observed"],
        fit_neighbors=surface["neighbors"],
        resolution=np.float32(args.resolution),
        size=np.float32(args.size),
        source_center=np.asarray(center, dtype=np.float64),
        crop_yaw_rad=np.float64(yaw),
        vertical_origin=np.float64(surface["vertical_origin"]),
    )


def write_sampling_manifest(output_dir, input_path, source_info, args,
                            records, accepted_centers):
    """Checkpoint accepted sampling progress so long runs can resume."""
    manifest = {
        "policy": "terrain20m-v4-base4-rawpcd",
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "source_file": os.path.abspath(input_path),
        "source": file_record(input_path),
        "dependency_versions": {
            "laspy": laspy.__version__,
            "numpy": np.__version__,
        },
        "seed": args.seed,
        "requested_accepted_scenes": args.accepted,
        "accepted_scenes": len(accepted_centers),
        "attempts_used": len(records),
        "parameters": vars(args),
        "records": records,
    }
    manifest_path = os.path.join(output_dir, "sampling_manifest.json")
    temporary_path = manifest_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary_path, manifest_path)


def main():
    args = parse_args()
    if args.accepted <= 0 or args.max_attempts <= 0:
        raise ValueError("accepted and max-attempts must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading raw XYZ returns from mother map...", flush=True)
    raw_points, source_count, source_bounds, class_counts = load_surface_points(
        args.input_laz)
    print(
        f"Loaded {len(raw_points)} finite raw XYZ points; LAS classes are not used",
        flush=True)
    source_info = (source_count, source_bounds, class_counts)
    rng = np.random.default_rng(args.seed)
    accepted_centers = []
    records = []
    first_attempt = 1
    if args.resume:
        manifest_path = os.path.join(args.output_dir, "sampling_manifest.json")
        current_source = os.path.abspath(args.input_laz)
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as stream:
                previous_manifest = json.load(stream)
            previous_source = os.path.abspath(
                previous_manifest.get("source_file", ""))
            if previous_source != current_source:
                raise ValueError(
                    "Resume source does not match the existing manifest: "
                    f"{previous_source} != {current_source}")
            records = list(previous_manifest.get("records", []))
            accepted_records = [
                record for record in records
                if record.get("status") == "accepted"
            ]
            accepted_centers = [
                np.asarray(record["center_xy"], dtype=np.float64)
                for record in accepted_records
            ]
            scene_files = glob.glob(os.path.join(args.output_dir, "scene_*.json"))
            if len(scene_files) != len(accepted_centers):
                raise ValueError(
                    "Resume scene files do not match accepted manifest records: "
                    f"{len(scene_files)} != {len(accepted_centers)}")
            if records:
                first_attempt = max(
                    int(record["attempt"]) for record in records) + 1
            print(
                f"Resuming {len(accepted_centers)} accepted scenes at attempt "
                f"{first_attempt}", flush=True)
        else:
            # A run interrupted before the first checkpoint can still reuse
            # its completed scene files. Start a fresh deterministic candidate
            # stream and keep the existing centres for separation checks.
            scene_files = sorted(
                glob.glob(os.path.join(args.output_dir, "scene_*.json")))
            if not scene_files:
                raise FileNotFoundError(
                    f"Cannot resume without a manifest or scene files: "
                    f"{manifest_path}")
            for scene_path in scene_files:
                with open(scene_path, encoding="utf-8") as stream:
                    scene_metadata = json.load(stream)
                center_xy = scene_metadata.get("source_center_xy")
                if not isinstance(center_xy, list) or len(center_xy) != 2:
                    raise ValueError(
                        f"Existing scene is missing source_center_xy: {scene_path}")
                records.append({
                    "attempt": 0,
                    "center_xy": [float(center_xy[0]), float(center_xy[1])],
                    "yaw_deg": float(scene_metadata.get("crop_yaw_deg", 0.0)),
                    "status": "accepted",
                    "scene": os.path.splitext(
                        os.path.basename(scene_path))[0],
                    "reasons": [],
                    "quality": scene_metadata.get("quality", {}),
                })
                accepted_centers.append(
                    np.asarray(center_xy, dtype=np.float64))
            print(
                f"Resuming {len(accepted_centers)} existing scenes without a "
                "manifest; starting a fresh candidate stream at attempt 1",
                flush=True)
        if len(accepted_centers) >= args.accepted:
            print(
                "Resume already satisfies the requested accepted-scene count; "
                "no new candidates will be sampled.",
                flush=True,
            )
            return 0

    # A rotated square reaches half_size*sqrt(2) from its centre. Use occupied
    # source-area cells inside that envelope as the sampling frame. Uniform
    # selection over cells avoids weighting centres by local point density;
    # the crop-level coverage gate still handles internal holes.
    surface_xy_min = np.min(raw_points[:, :2], axis=0)
    surface_xy_max = np.max(raw_points[:, :2], axis=0)
    center_margin = 0.5 * args.size * math.sqrt(2.0) + args.fit_radius
    support_cells = None
    support_origin = None
    if args.center_sampling == "uniform_support":
        if args.center_sampling_resolution <= 0.0:
            raise ValueError("center-sampling-resolution must be positive")
        sampling_resolution = args.center_sampling_resolution
        support_origin = np.floor(surface_xy_min / sampling_resolution) * sampling_resolution
        support_indices = np.floor(
            (raw_points[:, :2] - support_origin) / sampling_resolution
        ).astype(np.int64)
        width = int(support_indices[:, 0].max()) + 1
        linear = np.unique(
            support_indices[:, 1] * width + support_indices[:, 0])
        support_cells = np.column_stack((linear % width, linear // width))
        cell_centres = support_origin + (
            support_cells.astype(np.float64) + 0.5) * sampling_resolution
        jitter_margin = center_margin + 0.5 * sampling_resolution
        eligible = np.all(
            (cell_centres >= surface_xy_min + jitter_margin)
            & (cell_centres <= surface_xy_max - jitter_margin), axis=1)
        support_cells = support_cells[eligible]
        if len(support_cells) == 0:
            raise RuntimeError(
                "Mother map support cannot contain a rotated patch with the "
                f"required fitting margin: min={surface_xy_min.tolist()}, "
                f"max={surface_xy_max.tolist()}, margin={center_margin:.3f}m")
        print(
            f"Prepared {len(support_cells)} occupied "
            f"{sampling_resolution:g}m centre-sampling cells",
            flush=True)

    # Recreate the RNG state at the first unrecorded attempt. Each historical
    # attempt consumes one centre draw and one yaw draw in this loop.
    for _ in range(first_attempt - 1):
        if args.center_sampling == "uniform_support":
            rng.integers(0, len(support_cells))
            rng.random(2)
        else:
            rng.integers(0, len(raw_points))
        rng.uniform(-math.pi, math.pi)

    for attempt in range(first_attempt, args.max_attempts + 1):
        if args.center_sampling == "uniform_support":
            cell = support_cells[int(rng.integers(0, len(support_cells)))]
            center = support_origin + (
                cell.astype(np.float64) + rng.random(2)) \
                * args.center_sampling_resolution
        else:
            center = raw_points[int(rng.integers(0, len(raw_points))), :2]
        yaw = float(rng.uniform(-math.pi, math.pi))
        record = {
            "attempt": attempt,
            "center_xy": center.astype(float).tolist(),
            "yaw_deg": math.degrees(yaw),
        }
        if accepted_centers:
            separation = min(np.linalg.norm(center - old) for old in accepted_centers)
            record["nearest_accepted_center_m"] = float(separation)
            if separation < args.min_center_separation:
                record.update(status="prefilter_reject", reasons=["center_separation"])
                records.append(record)
                continue

        exact, padded = select_local_points(
            raw_points, center, yaw, args.size, args.fit_radius)
        density = len(exact) / (args.size ** 2)
        coverage = grid_coverage(
            exact, args.size, args.coverage_resolution) if len(exact) else 0.0
        record.update(
            raw_points=int(len(exact)),
            density_points_per_m2=float(density),
            raw_grid_coverage=float(coverage),
        )
        reasons = []
        if density < args.min_density:
            reasons.append("density")
        if coverage < args.min_coverage:
            reasons.append("raw_grid_coverage")
        if reasons:
            record.update(status="prefilter_reject", reasons=reasons)
            records.append(record)
            print(f"attempt {attempt}: prefilter reject {reasons}", flush=True)
            continue

        try:
            surface = build_surface(exact, padded, args)
        except Exception as exc:
            record.update(status="fit_reject", reasons=[str(exc)])
            records.append(record)
            print(f"attempt {attempt}: fit reject {exc}", flush=True)
            continue

        temporary_stem = os.path.join(args.output_dir, ".quality_candidate")
        temporary_npz = temporary_stem + ".npz"
        temporary_json = temporary_stem + ".json"
        metadata = metadata_for(
            surface, len(exact), coverage, center, yaw, source_info, args)
        save_npz(temporary_npz, surface, args, center, yaw)
        with open(temporary_json, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        quality = evaluate(temporary_npz)
        quality_record = dict(quality)
        quality_record.pop("path", None)
        record["quality"] = quality_record
        metadata["quality"] = quality_record
        with open(temporary_json, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        if quality["quality"] != "pass":
            record.update(status="quality_reject", reasons=quality["reasons"])
            records.append(record)
            os.remove(temporary_npz)
            os.remove(temporary_json)
            print(f"attempt {attempt}: quality reject {quality['reasons']}", flush=True)
            continue
        if quality["grade"] not in args.accepted_grades:
            record.update(
                status="grade_reject",
                reasons=[f"grade_not_requested:{quality['grade']}"],
            )
            records.append(record)
            os.remove(temporary_npz)
            os.remove(temporary_json)
            print(
                f"attempt {attempt}: grade reject {quality['grade']}",
                flush=True)
            continue
        scene_index = len(accepted_centers)
        stem = os.path.join(args.output_dir, f"scene_{scene_index:03d}")
        os.replace(temporary_npz, stem + ".npz")
        os.replace(temporary_json, stem + ".json")
        write_binary_pcd(
            stem + ".pcd", surface["planner_xyz"], surface["planner_normals"])
        grid_xyz = np.column_stack((
            surface["local_x"].ravel(), surface["local_y"].ravel(),
            surface["elevation"].ravel()))
        write_binary_pcd(
            stem + "_grid.pcd", grid_xyz, surface["normals"].reshape(-1, 3))
        save_preview(
            stem + "_preview.png", surface["elevation"], surface["normals"],
            surface["neighbors"], args.size)
        record.update(
            status="accepted",
            scene=os.path.basename(stem),
            reasons=[],
        )
        records.append(record)
        accepted_centers.append(center.copy())
        print(
            f"attempt {attempt}: accepted {record['scene']} "
            f"grade={quality['grade']} score={quality['geometry_score']:.1f}",
            flush=True)
        write_sampling_manifest(
            args.output_dir, args.input_laz, source_info, args,
            records, accepted_centers)
        if len(accepted_centers) >= args.accepted:
            break

    write_sampling_manifest(
        args.output_dir, args.input_laz, source_info, args,
        records, accepted_centers)
    manifest_path = os.path.join(args.output_dir, "sampling_manifest.json")
    print(json.dumps({
        "manifest": manifest_path,
        "accepted": len(accepted_centers),
        "attempts": len(records),
    }, ensure_ascii=False))
    return 0 if len(accepted_centers) >= args.accepted else 2


if __name__ == "__main__":
    sys.exit(main())
