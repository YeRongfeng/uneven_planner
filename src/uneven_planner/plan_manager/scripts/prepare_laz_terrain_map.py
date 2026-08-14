#!/usr/bin/env python3
"""Extract a metric terrain patch from a classified LAS/LAZ point cloud.

The script keeps only LAS class 2 (ground), fits a robust local plane at every
output cell, and writes a binary PCD with x/y/z and upward-facing normals.  It
does not scale any coordinate axis.  Invalid cells are filled only so the PCD
remains readable; their original validity is preserved in the NPZ mask.
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
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_laz", help="Input LAS/LAZ with class 2 ground points")
    parser.add_argument("output_dir", help="Directory for PCD, NPZ, PNG, and JSON outputs")
    parser.add_argument("--name", default="terrain_patch", help="Output file stem")
    parser.add_argument("--center-x", type=float, required=True, help="Patch center in source CRS")
    parser.add_argument("--center-y", type=float, required=True, help="Patch center in source CRS")
    parser.add_argument("--size", type=float, default=20.0, help="Square patch size in metres")
    parser.add_argument("--resolution", type=float, default=0.2, help="Output grid resolution in metres")
    parser.add_argument("--fit-radius", type=float, default=0.35,
                        help="Local plane fitting radius in metres (current calibrated candidate)")
    parser.add_argument("--min-neighbors", type=int, default=8, help="Minimum ground points per local fit")
    parser.add_argument("--max-rmse", type=float, default=0.12, help="Maximum robust plane RMSE in metres")
    parser.add_argument("--source-url", default="", help="Dataset landing page for provenance")
    parser.add_argument("--license", default="", help="Dataset licence identifier")
    parser.add_argument("--crs", default="", help="Source CRS, for example EPSG:25832")
    return parser.parse_args()


def load_ground_points(path, center_x, center_y, size, margin):
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
            ground = classification == 2
            if not np.any(ground):
                continue
            x = np.asarray(points.x)[ground]
            y = np.asarray(points.y)[ground]
            inside = (
                (x >= center_x - half)
                & (x <= center_x + half)
                & (y >= center_y - half)
                & (y <= center_y + half)
            )
            if not np.any(inside):
                continue
            xs.append(x[inside])
            ys.append(y[inside])
            zs.append(np.asarray(points.z)[ground][inside])

    if not xs:
        raise RuntimeError("No class 2 ground points found in the requested patch")

    xyz = np.column_stack((np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)))
    return xyz.astype(np.float64, copy=False), point_count, class_counts


def weighted_plane_fit(points, query_x, query_y, radius, max_rmse, min_neighbors):
    dx = points[:, 0] - query_x
    dy = points[:, 1] - query_y
    design = np.column_stack((dx, dy, np.ones(len(points))))
    sigma = max(radius * 0.5, 1e-3)
    weights = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))

    def solve(mask):
        root_w = np.sqrt(weights[mask])
        matrix = design[mask] * root_w[:, None]
        target = points[mask, 2] * root_w
        return np.linalg.lstsq(matrix, target, rcond=None)[0]

    keep = np.ones(len(points), dtype=bool)
    coeff = solve(keep)
    residual = points[:, 2] - design.dot(coeff)
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    threshold = max(0.03, 3.0 * 1.4826 * mad)
    keep = np.abs(residual - median) <= threshold
    if int(keep.sum()) < min_neighbors:
        return None

    coeff = solve(keep)
    residual = points[keep, 2] - design[keep].dot(coeff)
    rmse = float(np.sqrt(np.average(residual * residual, weights=weights[keep])))
    if not np.isfinite(rmse) or rmse > max_rmse:
        return None

    slope_x, slope_y, elevation = coeff
    normal = np.array([-slope_x, -slope_y, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return float(elevation), normal, rmse, int(keep.sum())


def fit_grid(xyz, center_x, center_y, size, resolution, radius, min_neighbors, max_rmse):
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
    normals = np.full((cells * cells, 3), np.nan, dtype=np.float64)
    rmse = np.full(cells * cells, np.nan, dtype=np.float64)
    neighbors = np.zeros(cells * cells, dtype=np.int32)

    for index, point_indices in enumerate(neighborhoods):
        if len(point_indices) < min_neighbors:
            continue
        fit = weighted_plane_fit(
            xyz[np.asarray(point_indices)],
            global_xy[index, 0],
            global_xy[index, 1],
            radius,
            max_rmse,
            min_neighbors,
        )
        if fit is None:
            continue
        elevation[index], normals[index], rmse[index], neighbors[index] = fit

    elevation = elevation.reshape(cells, cells)
    normals = normals.reshape(cells, cells, 3)
    rmse = rmse.reshape(cells, cells)
    neighbors = neighbors.reshape(cells, cells)
    valid = np.isfinite(elevation)
    if not np.any(valid):
        raise RuntimeError("Every local plane fit was invalid")

    # Keep the mask, but nearest-fill invalid samples so consumers that cannot
    # represent NaN still receive a structurally valid PCD.
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled_elevation = elevation[tuple(nearest)]
    filled_normals = normals[tuple(nearest)]
    return local_x, local_y, filled_elevation, filled_normals, valid, rmse, neighbors


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


def save_preview(path, elevation, normals, valid, rmse, size):
    slope = np.degrees(np.arccos(np.clip(normals[:, :, 2], -1.0, 1.0)))
    extent = [-0.5 * size, 0.5 * size, -0.5 * size, 0.5 * size]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    images = [
        axes[0].imshow(elevation, origin="lower", extent=extent, cmap="terrain"),
        axes[1].imshow(slope, origin="lower", extent=extent, cmap="magma", vmin=0.0),
        axes[2].imshow(np.where(valid, rmse * 100.0, np.nan), origin="lower", extent=extent, cmap="viridis"),
    ]
    titles = ["Elevation relative to patch minimum (m)", "Local slope (deg)", "Robust plane RMSE (cm)"]
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
    xyz, source_point_count, class_counts = load_ground_points(
        args.input_laz,
        args.center_x,
        args.center_y,
        args.size,
        args.fit_radius,
    )
    local_x, local_y, elevation, normals, valid, rmse, neighbors = fit_grid(
        xyz,
        args.center_x,
        args.center_y,
        args.size,
        args.resolution,
        args.fit_radius,
        args.min_neighbors,
        args.max_rmse,
    )

    core = (
        (xyz[:, 0] >= args.center_x - 0.5 * args.size)
        & (xyz[:, 0] < args.center_x + 0.5 * args.size)
        & (xyz[:, 1] >= args.center_y - 0.5 * args.size)
        & (xyz[:, 1] < args.center_y + 0.5 * args.size)
    )
    dense_xyz = xyz[core].copy()
    dense_xyz[:, 0] -= args.center_x
    dense_xyz[:, 1] -= args.center_y

    # Keep only raw ground returns that agree with the robust fitted surface.
    # The gridded representation is ideal for the network, while UnevenMap's
    # ellipsoid fit needs multiple source points in each local neighbourhood.
    first_center = -0.5 * args.size + 0.5 * args.resolution
    sample_columns = (dense_xyz[:, 0] - first_center) / args.resolution
    sample_rows = (dense_xyz[:, 1] - first_center) / args.resolution
    coordinates = np.vstack((sample_rows, sample_columns))
    sampled_surface = map_coordinates(
        elevation, coordinates, order=1, mode="nearest")
    valid_rmse = np.isfinite(rmse)
    rmse_nearest = distance_transform_edt(
        ~valid_rmse, return_distances=False, return_indices=True)
    filled_rmse = rmse[tuple(rmse_nearest)]
    sampled_rmse = map_coordinates(
        filled_rmse, coordinates, order=1, mode="nearest")
    surface_threshold = np.maximum(args.max_rmse, 3.0 * sampled_rmse)
    surface_consistent = np.abs(dense_xyz[:, 2] - sampled_surface) <= surface_threshold
    dense_xyz = dense_xyz[surface_consistent]
    coordinates = coordinates[:, surface_consistent]
    dense_normals = np.column_stack(
        [map_coordinates(normals[:, :, component], coordinates,
                         order=1, mode="nearest")
         for component in range(3)])
    dense_normals /= np.linalg.norm(dense_normals, axis=1, keepdims=True)

    # UnevenMap currently clips input PCDs to z in [-0.01, 5.0]. Translate the
    # lowest retained source/fitted point to zero. This is an offset only.
    vertical_origin = float(min(
        np.min(elevation[valid]), np.min(dense_xyz[:, 2])))
    elevation -= vertical_origin
    dense_xyz[:, 2] -= vertical_origin
    stem = os.path.join(args.output_dir, args.name)
    write_binary_pcd(stem + ".pcd", dense_xyz, dense_normals)
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
        fit_rmse=rmse.astype(np.float32),
        fit_neighbors=neighbors,
        resolution=np.float32(args.resolution),
        size=np.float32(args.size),
        source_center=np.array([args.center_x, args.center_y]),
        vertical_origin=np.float64(vertical_origin),
    )
    save_preview(stem + "_preview.png", elevation, normals, valid, rmse, args.size)

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
        "ground_points_in_patch": int(core.sum()),
        "ground_density_points_per_m2": float(core.sum() / (args.size * args.size)),
        "planner_point_count": int(len(dense_xyz)),
        "planner_point_density_points_per_m2": float(
            len(dense_xyz) / (args.size * args.size)),
        "planner_surface_consistent_fraction": float(
            len(dense_xyz) / max(1, int(core.sum()))),
        "valid_fraction": float(valid.mean()),
        "vertical_origin_m": vertical_origin,
        "elevation_range_m": [float(np.min(elevation)), float(np.max(elevation))],
        "slope_degrees": {
            "median": float(np.median(slope[valid])),
            "p95": float(np.quantile(slope[valid], 0.95)),
            "maximum": float(np.max(slope[valid])),
        },
        "fit_rmse_m": {
            "median": float(np.nanmedian(rmse)),
            "p95": float(np.nanquantile(rmse, 0.95)),
            "maximum": float(np.nanmax(rmse)),
        },
        "processing": {
            "ground_class": 2,
            "fit": "distance-weighted local z=ax+by+c with one MAD outlier rejection pass",
            "fit_radius_m": args.fit_radius,
            "min_neighbors": args.min_neighbors,
            "max_rmse_m": args.max_rmse,
            "axis_scaling": "none",
            "xy_output": "recentered only",
            "z_output": "source elevation minus patch minimum",
            "planner_pcd": "surface-consistent classified ground returns",
            "network_grid_pcd": os.path.basename(stem) + "_grid.pcd",
        },
    }
    with open(stem + ".json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
