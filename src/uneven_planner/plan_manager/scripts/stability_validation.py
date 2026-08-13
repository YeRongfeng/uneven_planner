#!/usr/bin/env python3
"""ROS-independent stability-map construction and trajectory validation.

The contract mirrors MPT's active privileged stability path:

* maps use ``(row=y, column=x, yaw)`` order;
* signed ESDF distance is periodic in yaw;
* XY samples live at cell centers; and
* continuous queries use trilinear interpolation with border padding.
"""

import math

import numpy as np
from scipy.ndimage import distance_transform_edt


DEFAULT_D_SAFE_METERS = 0.15
DEFAULT_YAW_ESDF_WEIGHT = 1.4


def build_periodic_signed_stability_esdf(
        yaw_stability,
        voxel_size_xy,
        yaw_weight=DEFAULT_YAW_ESDF_WEIGHT):
    """Convert a binary ``(H,W,Y)`` stability map to MPT's signed ESDF."""
    stability = np.asarray(yaw_stability)
    if stability.ndim != 3:
        raise ValueError(
            "yaw_stability must have shape (H,W,Y), got "
            f"{stability.shape}")
    if min(stability.shape) < 2:
        raise ValueError(
            "yaw_stability dimensions must all be at least two, got "
            f"{stability.shape}")
    if not np.all(np.isfinite(stability)):
        raise ValueError("yaw_stability contains NaN/Inf")
    voxel_size_xy = float(voxel_size_xy)
    yaw_weight = float(yaw_weight)
    if voxel_size_xy <= 0.0 or yaw_weight <= 0.0:
        raise ValueError("voxel_size_xy and yaw_weight must be positive")

    yaw_bins = stability.shape[2]
    occupied = stability <= 0.5
    occupied_tiled = np.concatenate(
        [occupied, occupied, occupied], axis=2)
    sampling = (
        voxel_size_xy,
        voxel_size_xy,
        yaw_weight * (2.0 * math.pi / float(yaw_bins)),
    )
    distance_to_occupied = distance_transform_edt(
        ~occupied_tiled, sampling=sampling)
    distance_to_free = distance_transform_edt(
        occupied_tiled, sampling=sampling)
    signed_tiled = distance_to_occupied - distance_to_free
    return signed_tiled[:, :, yaw_bins:2 * yaw_bins].astype(
        np.float32, copy=False)


def sample_periodic_stability_esdf(
        points_xy_yaw,
        stability_esdf,
        map_bounds,
        resolution):
    """Sample an ``(H,W,Y)`` ESDF at continuous ``(...,3)`` poses.

    ``map_bounds`` is ``(min_x,max_x,min_y,max_y)``. The lower bounds are
    cell edges, so index zero is centered half a cell above each bound.
    Out-of-map values are border-clamped exactly like MPT's grid sampler;
    callers performing hard validation should reject them separately.
    """
    points = np.asarray(points_xy_yaw, dtype=np.float64)
    esdf = np.asarray(stability_esdf, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(
            "points_xy_yaw must have shape (...,3), got "
            f"{points.shape}")
    if esdf.ndim != 3 or min(esdf.shape) < 2:
        raise ValueError(
            "stability_esdf must have shape (H,W,Y), with dimensions >=2, "
            f"got {esdf.shape}")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(esdf)):
        raise ValueError("points_xy_yaw/stability_esdf contains NaN/Inf")
    if len(map_bounds) != 4:
        raise ValueError("map_bounds must be (min_x,max_x,min_y,max_y)")

    min_x, max_x, min_y, max_y = map(float, map_bounds)
    resolution = float(resolution)
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    height, width, yaw_bins = esdf.shape
    if not np.isclose(max_x - min_x, width * resolution, atol=1e-6):
        raise ValueError("map x extent does not match W * resolution")
    if not np.isclose(max_y - min_y, height * resolution, atol=1e-6):
        raise ValueError("map y extent does not match H * resolution")

    flat = points.reshape(-1, 3)
    x_index = np.clip(
        (flat[:, 0] - min_x) / resolution - 0.5,
        0.0,
        float(width - 1),
    )
    y_index = np.clip(
        (flat[:, 1] - min_y) / resolution - 0.5,
        0.0,
        float(height - 1),
    )
    yaw_index = (
        np.mod(flat[:, 2] + math.pi, 2.0 * math.pi)
        / (2.0 * math.pi / float(yaw_bins))
    )

    x0 = np.floor(x_index).astype(np.int64)
    y0 = np.floor(y_index).astype(np.int64)
    yaw0 = np.floor(yaw_index).astype(np.int64) % yaw_bins
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    yaw1 = (yaw0 + 1) % yaw_bins
    wx = x_index - x0
    wy = y_index - y0
    wyaw = yaw_index - np.floor(yaw_index)

    c000 = esdf[y0, x0, yaw0]
    c001 = esdf[y0, x0, yaw1]
    c010 = esdf[y0, x1, yaw0]
    c011 = esdf[y0, x1, yaw1]
    c100 = esdf[y1, x0, yaw0]
    c101 = esdf[y1, x0, yaw1]
    c110 = esdf[y1, x1, yaw0]
    c111 = esdf[y1, x1, yaw1]

    c00 = c000 * (1.0 - wyaw) + c001 * wyaw
    c01 = c010 * (1.0 - wyaw) + c011 * wyaw
    c10 = c100 * (1.0 - wyaw) + c101 * wyaw
    c11 = c110 * (1.0 - wyaw) + c111 * wyaw
    c0 = c00 * (1.0 - wx) + c01 * wx
    c1 = c10 * (1.0 - wx) + c11 * wx
    sampled = c0 * (1.0 - wy) + c1 * wy
    return sampled.astype(np.float32).reshape(points.shape[:-1])


def validate_trajectory_stability(
        trajectory_xy_yaw,
        stability_esdf,
        map_bounds,
        resolution,
        d_safe=DEFAULT_D_SAFE_METERS):
    """Apply the continuous MPT stability hard gate to one trajectory."""
    trajectory = np.asarray(trajectory_xy_yaw, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError(
            "trajectory_xy_yaw must have shape (N,3), got "
            f"{trajectory.shape}")
    if len(trajectory) == 0:
        raise ValueError("trajectory_xy_yaw must not be empty")
    d_safe = float(d_safe)
    if d_safe < 0.0:
        raise ValueError("d_safe must be non-negative")

    min_x, max_x, min_y, max_y = map(float, map_bounds)
    finite_pose = np.all(np.isfinite(trajectory), axis=1)
    in_bounds = (
        finite_pose
        & (trajectory[:, 0] >= min_x)
        & (trajectory[:, 0] < max_x)
        & (trajectory[:, 1] >= min_y)
        & (trajectory[:, 1] < max_y)
    )
    margins = np.full(len(trajectory), np.nan, dtype=np.float32)
    if np.any(in_bounds):
        margins[in_bounds] = sample_periodic_stability_esdf(
            trajectory[in_bounds],
            stability_esdf,
            map_bounds,
            resolution,
        )
    finite_margin = np.isfinite(margins)
    stable = in_bounds & finite_margin & (margins >= d_safe)
    invalid_indices = np.flatnonzero(~stable)
    first_invalid_index = (
        None if len(invalid_indices) == 0 else int(invalid_indices[0]))

    reason = None
    if first_invalid_index is not None:
        if not finite_pose[first_invalid_index]:
            reason = "non_finite_pose"
        elif not in_bounds[first_invalid_index]:
            reason = "out_of_bounds"
        elif not finite_margin[first_invalid_index]:
            reason = "non_finite_margin"
        else:
            reason = "below_d_safe"

    finite_values = margins[finite_margin]
    minimum_margin = (
        float(np.min(finite_values)) if len(finite_values) else float("nan"))
    return {
        "valid": bool(np.all(stable)),
        "margins": margins,
        "stable": stable,
        "invalid_count": int(np.count_nonzero(~stable)),
        "below_threshold_count": int(np.count_nonzero(
            in_bounds & finite_margin & (margins < d_safe))),
        "first_invalid_index": first_invalid_index,
        "first_invalid_reason": reason,
        "minimum_margin": minimum_margin,
        "required_margin": d_safe,
    }
