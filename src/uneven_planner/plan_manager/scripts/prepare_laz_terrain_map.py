#!/usr/bin/env python3
"""Extract a metric terrain patch from raw LAS/LAZ XYZ returns.

The geometry-only path does not use LAS classification. It estimates a lower
ground envelope for the continuous terrain grid, while the planner PCD keeps
all supported raw XYZ returns in the crop. Only gross returns more than the
configured distance below that fitted surface are discarded. No obstacle
layer is created here; runtime mask extraction belongs to the
deployment/training pipeline.
"""

import argparse
import json
import math
import os
import struct

import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    label as connected_component_label,
    map_coordinates,
    median_filter,
)
from scipy.spatial import cKDTree


YRF_DEFAULT_COARSE_RESOLUTION = 0.4
YRF_DEFAULT_VOXEL_SIZE = 0.2
YRF_DEFAULT_GROUND_BAND_BELOW = 0.25
YRF_DEFAULT_GROUND_BAND_ABOVE = 0.35
YRF_DEFAULT_ENVELOPE_OUTLIER = 0.75
YRF_DEFAULT_LOWER_ENVELOPE_FILTER_SIZE = 7


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_laz", help="Input LAS/LAZ; all finite XYZ returns are used")
    parser.add_argument("output_dir", help="Directory for PCD, NPZ, PNG, and JSON outputs")
    parser.add_argument("--name", default="terrain_patch", help="Output file stem")
    parser.add_argument("--center-x", type=float, required=True, help="Patch center in source CRS")
    parser.add_argument("--center-y", type=float, required=True, help="Patch center in source CRS")
    parser.add_argument("--size", type=float, default=20.0, help="Square patch size in metres")
    parser.add_argument("--resolution", type=float, default=0.2, help="Output grid resolution in metres")
    parser.add_argument("--fit-radius", type=float, default=0.35,
                        help="Local plane fitting radius in metres (current calibrated candidate)")
    parser.add_argument("--direct-fit-min-points", type=int, default=5,
                        help="Minimum points for a directly observed local fit")
    parser.add_argument("--planner-surface-resolution", type=float, default=0.05,
                        help="Spacing of the dense PCD supplied to UnevenMap")
    parser.add_argument("--raw-below-surface-tolerance", type=float, default=1.0,
                        help="Discard raw returns more than this far below the fitted surface (m)")
    parser.add_argument("--raw-above-surface-tolerance", type=float, default=50.0,
                        help="Discard raw returns more than this far above the fitted surface (m)")
    parser.add_argument("--surface-cell-size", type=float, default=1.0,
                        help="Coarse XY cell for geometry-only lower-envelope extraction")
    parser.add_argument("--ground-band-below", type=float, default=0.25)
    parser.add_argument("--ground-band-above", type=float, default=0.35)
    parser.add_argument("--envelope-outlier", type=float, default=0.75)
    parser.add_argument("--source-url", default="", help="Dataset landing page for provenance")
    parser.add_argument("--license", default="", help="Dataset licence identifier")
    parser.add_argument("--crs", default="", help="Source CRS, for example EPSG:25832")
    return parser.parse_args()


def load_raw_points(path, center_x, center_y, size, margin):
    half = 0.5 * size + margin
    xs = []
    ys = []
    zs = []
    class_counts = np.zeros(256, dtype=np.int64)

    with laspy.open(path) as reader:
        point_count = int(reader.header.point_count)
        for points in reader.chunk_iterator(2_000_000):
            classification = np.asarray(points.classification, dtype=np.int64)
            class_counts += np.bincount(classification, minlength=256)
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            z = np.asarray(points.z)
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            inside = finite & (
                (x >= center_x - half)
                & (x <= center_x + half)
                & (y >= center_y - half)
                & (y <= center_y + half)
            )
            if not np.any(inside):
                continue
            xs.append(x[inside])
            ys.append(y[inside])
            zs.append(z[inside])

    if not xs:
        raise RuntimeError("No finite XYZ points found in the requested patch")

    xyz = np.column_stack((np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)))
    return xyz.astype(np.float64, copy=False), point_count, class_counts


def weighted_plane_fit(points, query_x, query_y, radius, min_points,
                       refine_outliers=True):
    """Fit local z=ax+by+c without turning non-planarity into missing data."""
    if len(points) < min_points:
        return None
    dx = points[:, 0] - query_x
    dy = points[:, 1] - query_y
    design = np.column_stack((dx, dy, np.ones(len(points))))
    sigma = max(radius * 0.5, 1e-3)
    weights = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    def solve(mask):
        root_w = np.sqrt(weights[mask])
        matrix = design[mask] * root_w[:, None]
        target = points[mask, 2] * root_w
        coeff, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        return coeff, rank

    all_points = np.ones(len(points), dtype=bool)
    coeff, rank = solve(all_points)
    if rank < 3 or not np.isfinite(coeff).all():
        return None

    if not refine_outliers:
        return float(coeff[2]), int(len(points))

    # Refine the estimate when a few returns are clear local
    # outliers.  This changes the fitted value, never the cell's validity: if
    # fewer than three points survive, the all-point fit above remains in use.
    residual = points[:, 2] - design.dot(coeff)
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    threshold = max(0.03, 3.0 * 1.4826 * mad)
    retained = np.abs(residual - median) <= threshold
    if int(retained.sum()) >= min_points:
        refined, refined_rank = solve(retained)
        if refined_rank == 3 and np.isfinite(refined).all():
            coeff = refined

    return float(coeff[2]), int(len(points))


def complete_elevation_surface(elevation, observed):
    """Harmonically fill unsupported grid cells while holding observations fixed."""
    if not np.any(observed):
        raise RuntimeError("No grid cell has enough point-cloud support")
    nearest = distance_transform_edt(
        ~observed, return_distances=False, return_indices=True)
    filled = elevation[tuple(nearest)].copy()
    missing = ~observed
    if not np.any(missing):
        return filled

    # Nearest fill supplies finite boundary values immediately.  Relax only the
    # missing cells so internal gaps become a continuous surface without moving
    # any directly fitted elevation.
    for _ in range(max(elevation.shape) * 2):
        padded = np.pad(filled, 1, mode="edge")
        neighbour_mean = 0.25 * (
            padded[:-2, 1:-1] + padded[2:, 1:-1]
            + padded[1:-1, :-2] + padded[1:-1, 2:])
        delta = np.max(np.abs(neighbour_mean[missing] - filled[missing]))
        filled[missing] = neighbour_mean[missing]
        if delta < 1e-6:
            break
    return filled


def normals_from_elevation(elevation, resolution):
    slope_y, slope_x = np.gradient(elevation, resolution, resolution)
    normals = np.stack((-slope_x, -slope_y, np.ones_like(elevation)), axis=-1)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    return normals


def geometry_ground_candidates(xyz, size, fit_radius, cell_size=1.0,
                               ground_below=0.25, ground_above=0.35,
                               envelope_outlier=0.75):
    """Split raw XYZ returns into a geometry-only ground candidate set.

    The estimator deliberately does not inspect LAS classification.  It builds
    a coarse lower elevation envelope, removes isolated low envelope spikes
    with a local median, and keeps raw returns close to that envelope.
    If ``ground_above`` is ``None``, every return on or above the lower
    envelope is retained. This is the all-class surface mode: high returns
    such as trees remain in the fitted surface while only low noise below the
    envelope is removed. The caller retains the original XYZ crop for the
    planner PCD.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("xyz must be a non-empty N x 3 array")
    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive")
    if ground_below < 0.0 or (
            ground_above is not None and ground_above < 0.0):
        raise ValueError("ground bands must be non-negative")
    if envelope_outlier <= 0.0:
        raise ValueError("envelope_outlier must be positive")

    support_half = 0.5 * size + fit_radius
    cells = int(math.ceil(2.0 * support_half / cell_size))
    origin = -support_half
    indices = np.floor((xyz[:, :2] - origin) / cell_size).astype(np.int64)
    inside = (
        (indices[:, 0] >= 0) & (indices[:, 0] < cells)
        & (indices[:, 1] >= 0) & (indices[:, 1] < cells))
    if not np.any(inside):
        raise ValueError("raw point cloud does not overlap the fitting window")

    coarse = np.full((cells, cells), np.inf, dtype=np.float64)
    selected_indices = indices[inside]
    np.minimum.at(
        coarse,
        (selected_indices[:, 1], selected_indices[:, 0]),
        np.asarray(xyz[:, 2][inside], dtype=np.float64),
    )
    observed = np.isfinite(coarse)
    if not np.any(observed):
        raise ValueError("raw point cloud has no finite coarse support")

    # Nearest fill is only an intermediate lookup for the local median.  It is
    # not the final terrain surface and does not create planner support points.
    nearest_indices = distance_transform_edt(
        ~observed, return_distances=False, return_indices=True)
    envelope = coarse[tuple(nearest_indices)]
    envelope = median_filter(envelope, size=3, mode="nearest")
    neighbourhood = median_filter(envelope, size=5, mode="nearest")
    # Only suppress implausibly low cells.  A high local envelope can be a
    # genuine rising terrain edge; the 3x3 median above already removes an
    # isolated high return without flattening a real slope.
    outlier = envelope < neighbourhood - envelope_outlier
    envelope[outlier] = neighbourhood[outlier]

    point_x = indices[:, 0]
    point_y = indices[:, 1]
    valid_point = (
        (point_x >= 0) & (point_x < cells)
        & (point_y >= 0) & (point_y < cells))
    ground_z = np.full(len(xyz), np.nan, dtype=np.float64)
    ground_z[valid_point] = envelope[point_y[valid_point], point_x[valid_point]]
    residual = np.asarray(xyz[:, 2], dtype=np.float64) - ground_z
    ground_mask = (
        np.isfinite(residual)
        & (residual >= -ground_below))
    if ground_above is not None:
        ground_mask &= residual <= ground_above
    ground_points = np.asarray(xyz[ground_mask], dtype=np.float64)
    if len(ground_points) < 3:
        raise ValueError(
            "geometry-only split retained fewer than three ground points")

    diagnostics = {
        "surface_cell_size_m": float(cell_size),
        "ground_band_below_m": float(ground_below),
        "ground_band_above_m": (
            None if ground_above is None else float(ground_above)),
        "fit_point_scope": (
            "all finite XYZ above lower envelope"
            if ground_above is None else "geometry ground candidates"),
        "envelope_outlier_m": float(envelope_outlier),
        "coarse_cells": int(cells),
        "coarse_observed_fraction": float(observed.mean()),
        "ground_candidate_count": int(len(ground_points)),
    }
    return ground_points, ground_z, residual, diagnostics


def resample_completed_surface(elevation, normals, valid, size, resolution,
                               output_resolution):
    """Densely sample one finite surface for UnevenMap's local covariance fit."""
    cells_float = size / output_resolution
    cells = int(round(cells_float))
    if not math.isclose(cells_float, cells, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "size must be an integer multiple of planner-surface-resolution")

    axis = -0.5 * size + (np.arange(cells) + 0.5) * output_resolution
    sample_x, sample_y = np.meshgrid(axis, axis)
    first_center = -0.5 * size + 0.5 * resolution
    columns = (sample_x.ravel() - first_center) / resolution
    rows = (sample_y.ravel() - first_center) / resolution
    coordinates = np.vstack((rows, columns))
    sampled_valid = map_coordinates(
        valid.astype(np.uint8), coordinates, order=0,
        mode="nearest").astype(bool)
    coordinates = coordinates[:, sampled_valid]
    xyz = np.column_stack((
        sample_x.ravel()[sampled_valid],
        sample_y.ravel()[sampled_valid],
        map_coordinates(elevation, coordinates, order=1, mode="nearest"),
    ))
    sampled_normals = np.column_stack([
        map_coordinates(normals[:, :, component], coordinates,
                        order=1, mode="nearest")
        for component in range(3)
    ])
    sampled_normals /= np.maximum(
        np.linalg.norm(sampled_normals, axis=1, keepdims=True), 1e-12)
    return xyz, sampled_normals


def retain_raw_returns(xyz, elevation, size, resolution,
                       below_surface_tolerance=1.0,
                       above_surface_tolerance=50.0):
    """Keep supported raw returns without turning gross outliers into terrain."""
    if below_surface_tolerance < 0.0:
        raise ValueError("below_surface_tolerance must be non-negative")
    if above_surface_tolerance <= 0.0:
        raise ValueError("above_surface_tolerance must be positive")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be an N x 3 array")
    half = 0.5 * size
    columns = (xyz[:, 0] + half) / resolution - 0.5
    rows = (xyz[:, 1] + half) / resolution - 0.5
    ground = map_coordinates(
        np.asarray(elevation, dtype=np.float64),
        np.vstack((rows, columns)), order=1, mode="nearest")
    residual = xyz[:, 2] - ground
    keep = np.isfinite(xyz).all(axis=1) & np.isfinite(residual)
    keep &= ((residual >= -below_surface_tolerance)
             & (residual <= above_surface_tolerance))
    return np.asarray(xyz[keep], dtype=np.float64)


def measure_above_surface_coverage(surface, size, resolution,
                                   height_threshold=2.0,
                                   cell_size=1.0):
    """Measure horizontal occupancy of returns above the fitted terrain.

    This is a geometry-only proxy for canopy/obstacle density.  It deliberately
    uses the same raw XYZ returns that reach the planner rather than LAS
    classification, because runtime point clouds do not carry LAS classes.
    The metric is used to reject unsuitable source crops; it does not remove
    any returns from the accepted planner PCD.
    """
    if size <= 0.0 or resolution <= 0.0:
        raise ValueError("size and resolution must be positive")
    if height_threshold < 0.0:
        raise ValueError("height_threshold must be non-negative")
    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive")

    raw_xyz = np.asarray(surface.get('raw_xyz'), dtype=np.float64)
    elevation = np.asarray(surface.get('elevation'), dtype=np.float64)
    if raw_xyz.ndim != 2 or raw_xyz.shape[1] != 3:
        raise ValueError("surface raw_xyz must be an N x 3 array")
    if elevation.ndim != 2 or len(raw_xyz) == 0:
        raise ValueError("surface must contain raw points and a 2D elevation")

    half = 0.5 * size
    columns = (raw_xyz[:, 0] + half) / resolution - 0.5
    rows = (raw_xyz[:, 1] + half) / resolution - 0.5
    fitted_z = map_coordinates(
        elevation, np.vstack((rows, columns)), order=1, mode='nearest')
    residual = raw_xyz[:, 2] - fitted_z
    finite = np.isfinite(residual)
    above = finite & (residual > height_threshold)

    cells = int(round(size / cell_size))
    if not math.isclose(size / cell_size, cells, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("size must be an integer multiple of cell_size")
    occupied = np.zeros(cells * cells, dtype=bool)
    if np.any(above):
        ix = np.floor((raw_xyz[above, 0] + half) / cell_size).astype(np.int64)
        iy = np.floor((raw_xyz[above, 1] + half) / cell_size).astype(np.int64)
        inside = ((ix >= 0) & (ix < cells) &
                  (iy >= 0) & (iy < cells))
        occupied[iy[inside] * cells + ix[inside]] = True

    occupied_grid = occupied.reshape((cells, cells))
    components, component_count = connected_component_label(
        occupied_grid, structure=np.ones((3, 3), dtype=np.uint8))
    component_sizes = np.bincount(components.ravel())[1:]
    largest_component = int(component_sizes.max()) if len(component_sizes) else 0

    finite_residual = residual[finite]
    return {
        'height_threshold_m': float(height_threshold),
        'cell_size_m': float(cell_size),
        'above_surface_point_count': int(np.count_nonzero(above)),
        'raw_point_count': int(len(raw_xyz)),
        'above_surface_point_fraction': float(
            np.count_nonzero(above) / max(len(raw_xyz), 1)),
        'above_surface_occupied_cells': int(np.count_nonzero(occupied)),
        'above_surface_total_cells': int(cells * cells),
        'above_surface_coverage_fraction': float(
            np.mean(occupied)),
        'above_surface_component_count': int(component_count),
        'above_surface_largest_component_cells': largest_component,
        'above_surface_residual_p95_m': float(
            np.percentile(finite_residual, 95)) if len(finite_residual) else 0.0,
        'above_surface_residual_p99_m': float(
            np.percentile(finite_residual, 99)) if len(finite_residual) else 0.0,
    }


def measure_classified_above_surface_coverage(
        surface, xyz, classifications, size, resolution,
        height_threshold=2.0, cell_size=1.0,
        tree_class_values=(1,), ground_class_values=(2,)):
    """Measure above-ground occupancy for selected LAS classes.

    ``classification`` is kept alongside the source returns only for
    offline crop selection and provenance.  The accepted planner PCD still
    contains the raw XYZ returns without class filtering.  The current
    sources label most non-ground forest returns as class 1 (unclassified),
    so this reports *tree candidates*, not exact individual tree instances.
    Connected components are horizontal occupied regions after the height
    test; one component is therefore an approximate object count.
    """
    if size <= 0.0 or resolution <= 0.0:
        raise ValueError("size and resolution must be positive")
    if height_threshold < 0.0:
        raise ValueError("height_threshold must be non-negative")
    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive")

    points = np.asarray(xyz, dtype=np.float64)
    labels = np.asarray(classifications, dtype=np.int64)
    elevation = np.asarray(surface.get('elevation'), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must be an N x 3 array")
    if labels.ndim != 1 or len(labels) != len(points):
        raise ValueError("classifications must align with xyz")
    if elevation.ndim != 2 or len(points) == 0:
        raise ValueError("surface must contain raw points and a 2D elevation")

    tree_class_values = tuple(int(value) for value in tree_class_values)
    ground_class_values = tuple(int(value) for value in ground_class_values)
    if not tree_class_values:
        raise ValueError("tree_class_values must not be empty")
    if not ground_class_values:
        raise ValueError("ground_class_values must not be empty")

    half = 0.5 * size
    columns = (points[:, 0] + half) / resolution - 0.5
    rows = (points[:, 1] + half) / resolution - 0.5
    fitted_z = map_coordinates(
        elevation, np.vstack((rows, columns)), order=1, mode='nearest')
    # build_surface shifts both the fitted surface and retained raw points by
    # vertical_origin.  The exact crop passed here is still in the pre-shift
    # local frame, so restore the fitted surface's original Z origin.
    fitted_z += float(surface.get('vertical_origin', 0.0))
    finite = np.isfinite(points).all(axis=1) & np.isfinite(fitted_z)
    tree_class = np.isin(labels, tree_class_values)
    ground_class = np.isin(labels, ground_class_values)
    tree_above = finite & tree_class & (points[:, 2] - fitted_z > height_threshold)

    cells_float = size / cell_size
    cells = int(round(cells_float))
    if not math.isclose(cells_float, cells, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("size must be an integer multiple of cell_size")
    occupied = np.zeros((cells, cells), dtype=bool)
    if np.any(tree_above):
        ix = np.floor((points[tree_above, 0] + half) / cell_size).astype(np.int64)
        iy = np.floor((points[tree_above, 1] + half) / cell_size).astype(np.int64)
        inside = ((ix >= 0) & (ix < cells) &
                  (iy >= 0) & (iy < cells))
        occupied[iy[inside], ix[inside]] = True

    components, component_count = connected_component_label(
        occupied, structure=np.ones((3, 3), dtype=np.uint8))
    component_sizes = np.bincount(components.ravel())[1:]
    largest_component = int(component_sizes.max()) if len(component_sizes) else 0
    unique_labels, label_counts = np.unique(labels[finite], return_counts=True)
    class_counts = {
        str(int(label)): int(count)
        for label, count in zip(unique_labels, label_counts)
    }
    tree_residual = points[tree_above, 2] - fitted_z[tree_above]
    return {
        'classification_field': 'LAS classification',
        'classification_values_in_crop': class_counts,
        'tree_class_values': list(tree_class_values),
        'ground_class_values': list(ground_class_values),
        'ground_class_point_count': int(np.count_nonzero(finite & ground_class)),
        'tree_class_point_count': int(np.count_nonzero(finite & tree_class)),
        'tree_candidate_point_count': int(np.count_nonzero(tree_above)),
        'tree_candidate_point_fraction': float(
            np.count_nonzero(tree_above) / max(len(points), 1)),
        'tree_candidate_occupied_cells': int(np.count_nonzero(occupied)),
        'tree_candidate_total_cells': int(cells * cells),
        'tree_candidate_coverage_fraction': float(np.mean(occupied)),
        'tree_candidate_component_count': int(component_count),
        'tree_candidate_largest_component_cells': largest_component,
        'tree_candidate_residual_p95_m': float(
            np.percentile(tree_residual, 95)) if len(tree_residual) else 0.0,
        'tree_candidate_residual_p99_m': float(
            np.percentile(tree_residual, 99)) if len(tree_residual) else 0.0,
        'height_threshold_m': float(height_threshold),
        'cell_size_m': float(cell_size),
        'point_count': int(len(points)),
    }


def fit_grid(xyz, center_x, center_y, size, resolution, radius,
             direct_fit_min_points=5, robust_refinement=True):
    cells_float = size / resolution
    cells = int(round(cells_float))
    if not math.isclose(cells_float, cells, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("size must be an integer multiple of resolution")

    local_axis = -0.5 * size + (np.arange(cells) + 0.5) * resolution
    local_x, local_y = np.meshgrid(local_axis, local_axis)
    global_xy = np.column_stack((local_x.ravel() + center_x, local_y.ravel() + center_y))
    tree = cKDTree(xyz[:, :2])
    neighborhoods = tree.query_ball_point(global_xy, radius)

    elevation = np.full(cells * cells, np.nan, dtype=np.float64)
    neighbors = np.zeros(cells * cells, dtype=np.int32)

    for index, point_indices in enumerate(neighborhoods):
        if len(point_indices) < direct_fit_min_points:
            continue
        fit = weighted_plane_fit(
            xyz[np.asarray(point_indices, dtype=np.int64)], global_xy[index, 0],
            global_xy[index, 1], radius, direct_fit_min_points,
            refine_outliers=robust_refinement)
        if fit is None:
            continue
        elevation[index], neighbors[index] = fit

    elevation = elevation.reshape(cells, cells)
    neighbors = neighbors.reshape(cells, cells)
    observed = np.isfinite(elevation)
    filled_elevation = complete_elevation_surface(elevation, observed)
    normals = normals_from_elevation(filled_elevation, resolution)
    valid = np.ones_like(observed, dtype=bool)
    return (local_x, local_y, filled_elevation, normals, valid, observed,
            neighbors)


def fit_yrf_ground_grid(
        xyz, size, resolution,
        coarse_resolution=YRF_DEFAULT_COARSE_RESOLUTION,
        voxel_size=YRF_DEFAULT_VOXEL_SIZE,
        ground_below=YRF_DEFAULT_GROUND_BAND_BELOW,
        ground_above=YRF_DEFAULT_GROUND_BAND_ABOVE,
        envelope_outlier=YRF_DEFAULT_ENVELOPE_OUTLIER,
        lower_envelope_filter_size=YRF_DEFAULT_LOWER_ENVELOPE_FILTER_SIZE):
    """Build the ground grid used by the YRF elevation-generator path.

    YRF first voxelizes the XYZ cloud and takes the minimum return in each
    coarse cell. A geometry-only low-envelope cleanup removes isolated returns
    far below their local surface; no LAS class is used. The cleaned minima
    feed the upper median of the surrounding 3x3 cells, followed by the upper
    median of returns in the ground band. The coarse elevation is then
    cubic-interpolated to the requested grid.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError("xyz must be a non-empty N x 3 array")
    if size <= 0.0 or resolution <= 0.0 or coarse_resolution <= 0.0:
        raise ValueError("size and grid resolutions must be positive")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    if ground_below < 0.0 or ground_above < 0.0:
        raise ValueError("ground bands must be non-negative")
    if envelope_outlier <= 0.0:
        raise ValueError("envelope_outlier must be positive")
    if (lower_envelope_filter_size < 1
            or lower_envelope_filter_size % 2 == 0):
        raise ValueError(
            "lower_envelope_filter_size must be a positive odd integer")
    coarse_cells_float = size / coarse_resolution
    coarse_cells = int(round(coarse_cells_float))
    if not math.isclose(coarse_cells_float, coarse_cells,
                        rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("size must be an integer multiple of coarse resolution")
    fine_cells_float = size / resolution
    fine_cells = int(round(fine_cells_float))
    if not math.isclose(fine_cells_float, fine_cells,
                        rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("size must be an integer multiple of resolution")

    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if not len(xyz):
        raise ValueError("xyz contains no finite points")

    # PCL VoxelGrid uses voxel centroids.  The local minimum is equivalent to
    # the voxel origin for this crop and keeps the operation independent of the
    # source CRS magnitude.
    voxel_origin = np.min(xyz, axis=0)
    voxel_indices = np.floor((xyz - voxel_origin) / voxel_size).astype(np.int64)
    unique_voxels, inverse = np.unique(
        voxel_indices, axis=0, return_inverse=True)
    voxel_sums = np.zeros((len(unique_voxels), 3), dtype=np.float64)
    np.add.at(voxel_sums, inverse, xyz)
    voxel_counts = np.bincount(inverse, minlength=len(unique_voxels))
    processed = voxel_sums / voxel_counts[:, None]

    half = size * 0.5
    origin = -half
    indices = np.floor((processed[:, :2] - origin) / coarse_resolution)
    indices = indices.astype(np.int64)
    inside = (
        (indices[:, 0] >= 0) & (indices[:, 0] < coarse_cells)
        & (indices[:, 1] >= 0) & (indices[:, 1] < coarse_cells))
    if not np.any(inside):
        raise ValueError("xyz does not overlap the requested map bounds")

    cell_min = np.full((coarse_cells, coarse_cells), np.inf, dtype=np.float64)
    selected = indices[inside]
    np.minimum.at(
        cell_min,
        (selected[:, 1], selected[:, 0]),
        processed[:, 2][inside],
    )
    cell_observed = np.isfinite(cell_min)
    if not np.any(cell_observed):
        raise ValueError("xyz has no coarse grid support")

    # Keep the YRF lower-envelope construction, but remove gross low returns
    # present in these forest LAS tiles before deriving the 3x3 reference.
    # This is geometric cleanup, not a LAS-class filter.
    nearest_indices = distance_transform_edt(
        ~cell_observed, return_distances=False, return_indices=True)
    envelope = cell_min[tuple(nearest_indices)]
    envelope = median_filter(envelope, size=3, mode="nearest")
    neighbourhood = median_filter(
        envelope, size=lower_envelope_filter_size, mode="nearest")
    low_outlier = envelope < neighbourhood - envelope_outlier
    envelope[low_outlier] = neighbourhood[low_outlier]

    def upper_median(values):
        values = np.asarray(values, dtype=np.float64)
        return float(np.partition(values, len(values) // 2)[len(values) // 2])

    ground_reference = np.full_like(cell_min, np.nan)
    for row in range(coarse_cells):
        for column in range(coarse_cells):
            neighbours = envelope[
                max(0, row - 1):min(coarse_cells, row + 2),
                max(0, column - 1):min(coarse_cells, column + 2)]
            neighbours = neighbours[np.isfinite(neighbours)]
            if len(neighbours):
                ground_reference[row, column] = upper_median(neighbours)

    elevation_coarse = np.full_like(ground_reference, np.nan)
    ground_counts = np.zeros_like(cell_min, dtype=np.int32)
    for row in range(coarse_cells):
        for column in range(coarse_cells):
            if not np.isfinite(ground_reference[row, column]):
                continue
            cell = (
                inside
                & (indices[:, 0] == column)
                & (indices[:, 1] == row))
            if not np.any(cell):
                continue
            residual = processed[cell, 2] - ground_reference[row, column]
            ground = processed[cell, 2][
                (residual >= -ground_below) & (residual <= ground_above)]
            ground_counts[row, column] = len(ground)
            elevation_coarse[row, column] = (
                upper_median(ground)
                if len(ground) else ground_reference[row, column])

    coarse_valid = np.isfinite(elevation_coarse)
    if not np.any(coarse_valid):
        raise ValueError("YRF ground split produced no finite coarse cells")
    # The simulator leaves cells without returns empty.  Canonical maps need a
    # complete 20 m sidecar, so only those unsupported coarse cells are filled
    # before applying the same cubic elevation interpolation.
    elevation_coarse = complete_elevation_surface(
        elevation_coarse, coarse_valid)

    fine_axis = origin + (np.arange(fine_cells) + 0.5) * resolution
    coarse_axis = origin + (np.arange(coarse_cells) + 0.5) * coarse_resolution
    fine_x, fine_y = np.meshgrid(fine_axis, fine_axis)
    columns = (fine_x.ravel() - coarse_axis[0]) / coarse_resolution
    rows = (fine_y.ravel() - coarse_axis[0]) / coarse_resolution
    coordinates = np.vstack((rows, columns))
    elevation = map_coordinates(
        elevation_coarse, coordinates, order=3, mode="nearest",
    ).reshape(fine_cells, fine_cells)
    observed = map_coordinates(
        (ground_counts > 0).astype(np.uint8), coordinates, order=0,
        mode="nearest",
    ).reshape(fine_cells, fine_cells).astype(bool)
    neighbors = map_coordinates(
        ground_counts.astype(np.float64), coordinates, order=0,
        mode="nearest",
    ).reshape(fine_cells, fine_cells).astype(np.int32)
    normals = normals_from_elevation(elevation, resolution)
    valid = np.ones_like(observed, dtype=bool)
    diagnostics = {
        "fit_method": "yrf_ground_reference",
        "fit_point_scope": "finite XYZ after 0.2m voxel centroids",
        "voxel_size_m": float(voxel_size),
        "coarse_resolution_m": float(coarse_resolution),
        "ground_reference": "3x3 upper median of coarse-cell minima",
        "ground_height": "upper median of returns in [-0.25m, +0.35m]",
        "lower_envelope_cleanup": (
            "3-cell median followed by "
            f"{lower_envelope_filter_size}x{lower_envelope_filter_size} "
            "local lower-outlier replacement"),
        "lower_envelope_outlier_m": float(envelope_outlier),
        "lower_envelope_outlier_cells": int(np.count_nonzero(low_outlier)),
        "ground_band_below_m": float(ground_below),
        "ground_band_above_m": float(ground_above),
        "voxel_point_count": int(len(processed)),
        "coarse_observed_fraction": float(cell_observed.mean()),
        "coarse_ground_fraction": float(np.mean(ground_counts > 0)),
        "ground_candidate_count": int(np.sum(ground_counts)),
        "interpolation": "cubic elevation; nearest support/count mask",
    }
    return (fine_x, fine_y, elevation, normals, valid, observed, neighbors,
            diagnostics)


def write_binary_pcd(path, xyz, normals):
    fields = np.column_stack((xyz, normals)).astype("<f4", copy=False)
    count = len(fields)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z normal_x normal_y normal_z\n"
        "SIZE 4 4 4 4 4 4\n"
        "TYPE F F F F F F\n"
        "COUNT 1 1 1 1 1 1\n"
        "WIDTH {}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS {}\n"
        "DATA binary\n"
    ).format(count, count)
    with open(path, "wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(fields.tobytes(order="C"))


def save_preview(path, elevation, normals, neighbors, size):
    slope = np.degrees(np.arccos(np.clip(normals[:, :, 2], -1.0, 1.0)))
    extent = [-0.5 * size, 0.5 * size, -0.5 * size, 0.5 * size]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    images = [
        axes[0].imshow(elevation, origin="lower", extent=extent, cmap="terrain"),
        axes[1].imshow(slope, origin="lower", extent=extent, cmap="magma", vmin=0.0),
        axes[2].imshow(neighbors, origin="lower", extent=extent, cmap="viridis"),
    ]
    titles = ["Elevation relative to patch minimum (m)", "Local slope (deg)", "Direct-fit neighbour count"]
    for axis, image, title in zip(axes, images, titles):
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    xyz, source_point_count, class_counts = load_raw_points(
        args.input_laz,
        args.center_x,
        args.center_y,
        args.size,
        args.fit_radius,
    )
    local_xyz = xyz.copy()
    local_xyz[:, 0] -= args.center_x
    local_xyz[:, 1] -= args.center_y
    ground_points, _, _, split_diagnostics = geometry_ground_candidates(
        local_xyz,
        args.size,
        args.fit_radius,
        cell_size=args.surface_cell_size,
        ground_below=args.ground_band_below,
        ground_above=args.ground_band_above,
        envelope_outlier=args.envelope_outlier,
    )
    local_x, local_y, elevation, normals, valid, observed, neighbors = fit_grid(
        ground_points,
        0.0,
        0.0,
        args.size,
        args.resolution,
        args.fit_radius,
        args.direct_fit_min_points,
    )

    core = (
        (local_xyz[:, 0] >= -0.5 * args.size)
        & (local_xyz[:, 0] < 0.5 * args.size)
        & (local_xyz[:, 1] >= -0.5 * args.size)
        & (local_xyz[:, 1] < 0.5 * args.size)
    )
    dense_xyz, dense_normals = resample_completed_surface(
        elevation, normals, valid, args.size, args.resolution,
        args.planner_surface_resolution)

    raw_xyz = retain_raw_returns(
        local_xyz[core], elevation, args.size, args.resolution,
        args.raw_below_surface_tolerance,
        args.raw_above_surface_tolerance)
    raw_normals = np.zeros_like(raw_xyz)

    # Translate the lowest retained source/fitted point to zero. This is an
    # offset only; the raw PCD may contain returns well above the fitted grid.
    vertical_origin = float(min(
        np.min(elevation[valid]), np.min(dense_xyz[:, 2])))
    elevation -= vertical_origin
    dense_xyz[:, 2] -= vertical_origin
    raw_xyz[:, 2] -= vertical_origin
    planner_xyz = np.vstack((dense_xyz, raw_xyz))
    planner_normals = np.vstack((dense_normals, raw_normals))
    stem = os.path.join(args.output_dir, args.name)
    write_binary_pcd(stem + ".pcd", planner_xyz, planner_normals)
    grid_xyz = np.column_stack(
        (local_x.ravel(), local_y.ravel(), elevation.ravel()))
    write_binary_pcd(stem + "_grid.pcd", grid_xyz, normals.reshape(-1, 3))
    np.savez_compressed(
        stem + ".npz",
        elevation=elevation.astype(np.float32),
        normal_x=normals[:, :, 0].astype(np.float32),
        normal_y=normals[:, :, 1].astype(np.float32),
        normal_z=normals[:, :, 2].astype(np.float32),
        valid_mask=valid,
        observed_mask=observed,
        fit_neighbors=neighbors,
        resolution=np.float32(args.resolution),
        size=np.float32(args.size),
        source_center=np.array([args.center_x, args.center_y]),
        vertical_origin=np.float64(vertical_origin),
    )
    save_preview(stem + "_preview.png", elevation, normals, neighbors, args.size)

    slope = np.degrees(np.arccos(np.clip(normals[:, :, 2], -1.0, 1.0)))
    metadata = {
        "source_file": os.path.abspath(args.input_laz),
        "source_url": args.source_url,
        "license": args.license,
        "source_crs": args.crs,
        "source_point_count": source_point_count,
        "source_class_counts": {str(i): int(value) for i, value in enumerate(class_counts) if value},
        "source_center_xy": [args.center_x, args.center_y],
        "patch_size_m": args.size,
        "resolution_m": args.resolution,
        "grid_shape": list(elevation.shape),
        "source_raw_points_in_patch": int(core.sum()),
        "raw_returns_retained_in_planner_pcd": int(len(raw_xyz)),
        "ground_candidate_points_in_patch": int(len(ground_points)),
        "ground_candidate_density_points_per_m2": float(
            len(ground_points) / (args.size * args.size)),
        "planner_point_count": int(len(planner_xyz)),
        "planner_point_density_points_per_m2": float(
            len(planner_xyz) / (args.size * args.size)),
        "planner_surface_consistent_fraction": 1.0,
        "valid_fraction": float(valid.mean()),
        "observed_fraction": float(observed.mean()),
        "vertical_origin_m": vertical_origin,
        "elevation_range_m": [float(np.min(elevation)), float(np.max(elevation))],
        "slope_degrees": {
            "median": float(np.median(slope[valid])),
            "p95": float(np.quantile(slope[valid], 0.95)),
            "maximum": float(np.max(slope[valid])),
        },
        "processing": {
            "surface_source": "all finite XYZ returns; no LAS class filtering; gross below-surface outliers removed",
            "ground_fit_support": "coarse lower envelope + local median outlier removal + height band",
            "split_diagnostics": split_diagnostics,
            "surface_cell_size_m": args.surface_cell_size,
            "ground_band_below_m": args.ground_band_below,
            "ground_band_above_m": args.ground_band_above,
            "envelope_outlier_m": args.envelope_outlier,
            "fit": "distance-weighted local z=ax+by+c with non-rejecting MAD refinement on geometry-only ground candidates",
            "fit_radius_m": args.fit_radius,
            "direct_fit_minimum_points": args.direct_fit_min_points,
            "gap_completion": "harmonic elevation fill followed by finite-difference normals",
            "planner_surface_resolution_m": args.planner_surface_resolution,
            "raw_below_surface_tolerance_m": args.raw_below_surface_tolerance,
            "raw_above_surface_tolerance_m": args.raw_above_surface_tolerance,
            "axis_scaling": "none",
            "xy_output": "recentered only",
            "z_output": "source elevation minus patch minimum",
            "planner_pcd": "dense completed fitted surface plus supported raw XYZ returns",
            "network_grid_pcd": os.path.basename(stem) + "_grid.pcd",
        },
    }
    with open(stem + ".json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
