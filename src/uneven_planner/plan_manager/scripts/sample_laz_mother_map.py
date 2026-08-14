#!/usr/bin/env python3
"""Sample and quality-gate metric 20 m scenes directly from a LAS/LAZ mother map.

The source is loaded once, but no fixed crop list is required.  Candidate
centres and yaw angles are sampled deterministically from classified ground
surface returns.  Cheap density/coverage checks run before the robust 100 x 100 local
plane fit.  Only Stage-A quality-passing scenes are retained; every attempted
location and rejection reason is recorded in a manifest.
"""

import argparse
import datetime
import json
import math
import os
import sys

import laspy
import numpy as np
from scipy.ndimage import map_coordinates

from prepare_laz_terrain_map import (
    fit_grid,
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
    parser.add_argument("input_laz", help="Classified LAS/LAZ mother map")
    parser.add_argument("output_dir", help="Retained scene and manifest directory")
    parser.add_argument("--accepted", type=int, default=5,
                        help="Number of quality-passing scenes to retain")
    parser.add_argument("--max-attempts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--size", type=float, default=20.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--planner-surface-resolution", type=float, default=0.05,
                        help="Dense fitted-surface spacing used by UnevenMap")
    parser.add_argument(
        "--source-profile", choices=("uls", "als"), default="uls",
        help="Acquisition-density profile; controls explicit fitting defaults")
    parser.add_argument(
        "--point-classes", default="2",
        help="Comma-separated LAS classes used as the terrain surface (normally 2; use 1 only when source metadata says the cloud is an unclassified bare surface)")
    parser.add_argument("--fit-radius", type=float, default=None,
                        help="Robust local-plane radius; defaults to 0.35 m for ULS and 0.9 m for ALS")
    parser.add_argument("--min-neighbors", type=int, default=None)
    parser.add_argument("--max-rmse", type=float, default=0.12)
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
            "min_neighbors": 8,
            "min_density": 40.0,
            "coverage_resolution": 0.2,
        },
        "als": {
            "fit_radius": 0.9,
            "min_neighbors": 5,
            "min_density": 4.0,
            "coverage_resolution": 1.0,
        },
    }[args.source_profile]
    for name, value in profile_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    try:
        args.point_classes = sorted({
            int(value.strip()) for value in args.point_classes.split(",")
            if value.strip()
        })
    except ValueError as exc:
        raise ValueError("point-classes must be comma-separated integers") from exc
    if (not args.point_classes
            or any(value < 0 or value > 255 for value in args.point_classes)):
        raise ValueError("point-classes must contain LAS classes from 0 to 255")
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


def load_surface_points(path, point_classes):
    chunks = []
    class_counts = np.zeros(256, dtype=np.int64)
    with laspy.open(path) as reader:
        source_count = int(reader.header.point_count)
        source_bounds = {
            "minimum": reader.header.mins.astype(float).tolist(),
            "maximum": reader.header.maxs.astype(float).tolist(),
        }
        for points in reader.chunk_iterator(2_000_000):
            classification = np.asarray(points.classification, dtype=np.int64)
            class_counts += np.bincount(classification, minlength=256)
            selected = np.isin(classification, point_classes)
            if not np.any(selected):
                continue
            xyz = np.column_stack((
                np.asarray(points.x)[selected],
                np.asarray(points.y)[selected],
                np.asarray(points.z)[selected],
            ))
            finite = np.isfinite(xyz).all(axis=1)
            if np.any(finite):
                chunks.append(xyz[finite])
    if not chunks:
        raise RuntimeError(
            f"Mother map contains no finite points in classes {point_classes}")
    return (np.concatenate(chunks), source_count, source_bounds,
            {str(i): int(value) for i, value in enumerate(class_counts) if value})


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
    local_x, local_y, elevation, normals, valid, rmse, neighbors = fit_grid(
        padded, 0.0, 0.0, args.size, args.resolution,
        args.fit_radius, args.min_neighbors, args.max_rmse)

    # UnevenMap estimates a local covariance directly from the PCD. Feeding it
    # raw classified returns makes its surface differ from the robust grid used
    # by the quality gate and the network. Resample that fitted surface densely
    # instead, so all three consumers see the same geometry while UnevenMap
    # still has enough points in each ellipsoid neighbourhood.
    planner_cells_float = args.size / args.planner_surface_resolution
    planner_cells = int(round(planner_cells_float))
    if not math.isclose(planner_cells_float, planner_cells,
                        rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "size must be an integer multiple of planner-surface-resolution")
    planner_axis = (-0.5 * args.size
                    + (np.arange(planner_cells) + 0.5)
                    * args.planner_surface_resolution)
    planner_x, planner_y = np.meshgrid(planner_axis, planner_axis)
    first_center = -0.5 * args.size + 0.5 * args.resolution
    columns = (planner_x.ravel() - first_center) / args.resolution
    rows = (planner_y.ravel() - first_center) / args.resolution
    coordinates = np.vstack((rows, columns))
    dense_valid = map_coordinates(
        valid.astype(np.uint8), coordinates, order=0,
        mode="nearest").astype(bool)
    coordinates = coordinates[:, dense_valid]
    dense_xyz = np.column_stack((
        planner_x.ravel()[dense_valid],
        planner_y.ravel()[dense_valid],
        map_coordinates(elevation, coordinates, order=1, mode="nearest"),
    ))
    dense_normals = np.column_stack([
        map_coordinates(normals[:, :, component], coordinates,
                        order=1, mode="nearest")
        for component in range(3)
    ])
    dense_normals /= np.maximum(
        np.linalg.norm(dense_normals, axis=1, keepdims=True), 1e-12)

    vertical_origin = float(min(np.min(elevation[valid]), np.min(dense_xyz[:, 2])))
    elevation -= vertical_origin
    dense_xyz[:, 2] -= vertical_origin
    return {
        "local_x": local_x,
        "local_y": local_y,
        "elevation": elevation,
        "normals": normals,
        "valid": valid,
        "rmse": rmse,
        "neighbors": neighbors,
        "dense_xyz": dense_xyz,
        "dense_normals": dense_normals,
        "vertical_origin": vertical_origin,
        "consistent_fraction": 1.0,
    }


def metadata_for(surface, exact_count, support_coverage, center, yaw,
                 source_info, args):
    valid = surface["valid"]
    normals = surface["normals"]
    slope = np.degrees(np.arccos(np.clip(normals[:, :, 2], -1.0, 1.0)))
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
        "source_surface_classes": args.point_classes,
        "source_surface_points_in_patch": int(exact_count),
        "source_surface_density_points_per_m2": float(
            exact_count / (args.size ** 2)),
        "source_support_coverage": float(support_coverage),
        "source_support_resolution_m": float(args.coverage_resolution),
        "planner_point_count": int(len(surface["dense_xyz"])),
        "planner_point_density_points_per_m2": float(
            len(surface["dense_xyz"]) / (args.size ** 2)),
        "planner_surface_consistent_fraction": float(surface["consistent_fraction"]),
        "valid_fraction": float(valid.mean()),
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
        "fit_rmse_m": {
            "median": float(np.nanmedian(surface["rmse"])),
            "p95": float(np.nanquantile(surface["rmse"], 0.95)),
            "maximum": float(np.nanmax(surface["rmse"])),
        },
        "processing": {
            "source_profile": args.source_profile,
            "surface_classes": args.point_classes,
            "sampling": (
                "uniform occupied source-area cell with within-cell jitter and random yaw"
                if args.center_sampling == "uniform_support"
                else "random selected-surface point centre and random yaw"),
            "center_sampling_resolution_m": args.center_sampling_resolution,
            "fit": "distance-weighted local plane with MAD rejection",
            "fit_radius_m": args.fit_radius,
            "min_neighbors": args.min_neighbors,
            "max_rmse_m": args.max_rmse,
            "planner_surface_resolution_m": args.planner_surface_resolution,
            "axis_scaling": "none",
            "xy_output": "translated and rotated only",
            "z_output": "source elevation minus retained patch minimum",
            "planner_pcd": "dense bilinear resampling of robust fitted surface",
        },
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
        fit_rmse=surface["rmse"].astype(np.float32),
        fit_neighbors=surface["neighbors"],
        resolution=np.float32(args.resolution),
        size=np.float32(args.size),
        source_center=np.asarray(center, dtype=np.float64),
        crop_yaw_rad=np.float64(yaw),
        vertical_origin=np.float64(surface["vertical_origin"]),
    )


def main():
    args = parse_args()
    if args.accepted <= 0 or args.max_attempts <= 0:
        raise ValueError("accepted and max-attempts must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading selected terrain-surface returns from mother map...", flush=True)
    ground, source_count, source_bounds, class_counts = load_surface_points(
        args.input_laz, args.point_classes)
    print(
        f"Loaded {len(ground)} points from LAS classes {args.point_classes}",
        flush=True)
    source_info = (source_count, source_bounds, class_counts)
    rng = np.random.default_rng(args.seed)
    accepted_centers = []
    records = []

    # A rotated square reaches half_size*sqrt(2) from its centre. Use occupied
    # source-area cells inside that envelope as the sampling frame. Uniform
    # selection over cells avoids weighting centres by local point density;
    # the crop-level coverage gate still handles internal holes.
    surface_xy_min = np.min(ground[:, :2], axis=0)
    surface_xy_max = np.max(ground[:, :2], axis=0)
    center_margin = 0.5 * args.size * math.sqrt(2.0) + args.fit_radius
    support_cells = None
    support_origin = None
    if args.center_sampling == "uniform_support":
        if args.center_sampling_resolution <= 0.0:
            raise ValueError("center-sampling-resolution must be positive")
        sampling_resolution = args.center_sampling_resolution
        support_origin = np.floor(surface_xy_min / sampling_resolution) * sampling_resolution
        support_indices = np.floor(
            (ground[:, :2] - support_origin) / sampling_resolution
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

    for attempt in range(1, args.max_attempts + 1):
        if args.center_sampling == "uniform_support":
            cell = support_cells[int(rng.integers(0, len(support_cells)))]
            center = support_origin + (
                cell.astype(np.float64) + rng.random(2)) \
                * args.center_sampling_resolution
        else:
            center = ground[int(rng.integers(0, len(ground))), :2]
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
            ground, center, yaw, args.size, args.fit_radius)
        density = len(exact) / (args.size ** 2)
        coverage = grid_coverage(
            exact, args.size, args.coverage_resolution) if len(exact) else 0.0
        record.update(
            ground_points=int(len(exact)),
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
        record["quality"] = quality
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
            stem + ".pcd", surface["dense_xyz"], surface["dense_normals"])
        grid_xyz = np.column_stack((
            surface["local_x"].ravel(), surface["local_y"].ravel(),
            surface["elevation"].ravel()))
        write_binary_pcd(
            stem + "_grid.pcd", grid_xyz, surface["normals"].reshape(-1, 3))
        save_preview(
            stem + "_preview.png", surface["elevation"], surface["normals"],
            surface["valid"], surface["rmse"], args.size)
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
        if len(accepted_centers) >= args.accepted:
            break

    manifest = {
        "policy": "terrain20m-v2",
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "source_file": os.path.abspath(args.input_laz),
        "source": file_record(args.input_laz),
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
    manifest_path = os.path.join(args.output_dir, "sampling_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({
        "manifest": manifest_path,
        "accepted": len(accepted_centers),
        "attempts": len(records),
    }, ensure_ascii=False))
    return 0 if len(accepted_centers) >= args.accepted else 2


if __name__ == "__main__":
    sys.exit(main())
