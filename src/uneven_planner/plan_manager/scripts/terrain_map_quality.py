#!/usr/bin/env python3
"""Objective quality gate and geometric difficulty grading for terrain maps.

Input is the NPZ sidecar produced by prepare_laz_terrain_map.py.  Quality is a
hard gate.  Difficulty is reported only for maps that pass that gate; planner
benchmark results should later calibrate the geometric grade.
"""

import argparse
import glob
import json
import math
import os

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label


POLICY_VERSION = "terrain20m-v2"
MAX_TRAVERSABLE_SLOPE_DEG = math.degrees(math.acos(0.8))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="NPZ files or glob patterns")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    return sorted(dict.fromkeys(paths))


def percentile(values, q):
    return float(np.quantile(values, q))


def evaluate(path):
    data = np.load(path)
    required = {"elevation", "normal_x", "normal_y", "normal_z", "valid_mask", "fit_rmse", "resolution"}
    missing = sorted(required.difference(data.files))
    if missing:
        return {"path": path, "quality": "reject", "grade": None, "reasons": ["missing:" + ",".join(missing)]}

    elevation = data["elevation"].astype(np.float64)
    valid = data["valid_mask"].astype(bool)
    rmse = data["fit_rmse"].astype(np.float64)
    resolution = float(data["resolution"])
    normals = np.stack((data["normal_x"], data["normal_y"], data["normal_z"]), axis=-1).astype(np.float64)
    metadata_path = os.path.splitext(path)[0] + ".json"
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as stream:
            metadata = json.load(stream)

    finite = np.isfinite(elevation) & np.isfinite(normals).all(axis=-1)
    valid &= finite
    if not np.any(valid):
        return {"path": path, "quality": "reject", "grade": None, "reasons": ["no_valid_cells"]}

    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = elevation[tuple(nearest)]
    rows, columns = np.indices(elevation.shape)
    design = np.column_stack((columns[valid] * resolution, rows[valid] * resolution, np.ones(valid.sum())))
    plane = np.linalg.lstsq(design, elevation[valid], rcond=None)[0]
    detrended = filled - (plane[0] * columns * resolution + plane[1] * rows * resolution + plane[2])

    micro = filled - gaussian_filter(filled, 0.4 / resolution)
    local = filled - gaussian_filter(filled, 1.2 / resolution)
    normal_length = np.linalg.norm(normals, axis=-1)
    slope = np.degrees(np.arccos(np.clip(normals[:, :, 2] / np.maximum(normal_length, 1e-12), -1.0, 1.0)))
    traversable = valid & (slope <= MAX_TRAVERSABLE_SLOPE_DEG)
    components, _ = label(traversable)
    counts = np.bincount(components.ravel())
    largest_traversable = float(np.max(counts[1:]) / valid.size) if len(counts) > 1 else 0.0

    horizontal = np.abs(np.diff(filled, axis=1))[valid[:, 1:] & valid[:, :-1]]
    vertical = np.abs(np.diff(filled, axis=0))[valid[1:, :] & valid[:-1, :]]
    steps = np.concatenate((horizontal, vertical))
    source_density = float(metadata.get(
        "source_surface_density_points_per_m2",
        metadata.get("ground_density_points_per_m2", float("nan"))))
    planner_density = float(metadata.get(
        "planner_point_density_points_per_m2", source_density))
    processing = metadata.get("processing", {})
    source_profile = processing.get("source_profile", "uls")
    source_density_floor = {"uls": 40.0, "als": 4.0}.get(source_profile)
    source_support_coverage = float(metadata.get(
        "source_support_coverage", float("nan")))
    relief_98 = percentile(elevation[valid], 0.99) - percentile(elevation[valid], 0.01)
    elevation_min = float(np.min(elevation[valid]))
    elevation_max = float(np.max(elevation[valid]))

    metrics = {
        "valid_fraction": float(valid.mean()),
        "elevation_min_m": elevation_min,
        "elevation_max_m": elevation_max,
        "source_profile": source_profile,
        "source_surface_density_points_per_m2": source_density,
        "planner_point_density_points_per_m2": planner_density,
        "source_support_coverage": source_support_coverage,
        "fit_rmse_p95_m": float(np.nanquantile(rmse, 0.95)),
        "normal_length_error_p99": percentile(np.abs(normal_length[valid] - 1.0), 0.99),
        "relief_p01_p99_m": relief_98,
        "detrended_std_m": float(np.std(detrended[valid])),
        "micro_abs_p90_m": percentile(np.abs(micro[valid]), 0.90),
        "local_abs_p90_m": percentile(np.abs(local[valid]), 0.90),
        "slope_median_deg": percentile(slope[valid], 0.50),
        "slope_p95_deg": percentile(slope[valid], 0.95),
        "slope_over_20_fraction": float(np.mean(slope[valid] > 20.0)),
        "untraversable_slope_fraction": float(np.mean(slope[valid] > MAX_TRAVERSABLE_SLOPE_DEG)),
        "largest_traversable_component_fraction": largest_traversable,
        "neighbor_step_p999_m": percentile(steps, 0.999),
        "neighbor_step_max_m": float(np.max(steps)),
    }

    reasons = []
    if metrics["valid_fraction"] < 0.97:
        reasons.append("valid_fraction<0.97")
    if elevation_min < -0.009 or elevation_max > 5.0:
        reasons.append("outside_runtime_z_contract[-0.01,5.0]")
    if source_density_floor is None:
        reasons.append("unknown_source_profile")
    elif (not math.isfinite(source_density)
          or source_density < source_density_floor):
        reasons.append(
            f"source_surface_density<{source_density_floor:g}/m2")
    if math.isfinite(source_support_coverage) and source_support_coverage < 0.97:
        reasons.append("source_support_coverage<0.97")
    if not math.isfinite(planner_density) or planner_density < 40.0:
        reasons.append("planner_surface_density<40/m2")
    if metrics["fit_rmse_p95_m"] > 0.08:
        reasons.append("fit_rmse_p95>0.08m")
    if metrics["normal_length_error_p99"] > 0.01:
        reasons.append("non_unit_normals")
    if metrics["neighbor_step_max_m"] > 0.50:
        reasons.append("isolated_height_jump>0.50m_per_cell")
    if metrics["largest_traversable_component_fraction"] < 0.70:
        reasons.append("largest_traversable_component<0.70")
    if metrics["untraversable_slope_fraction"] > 0.25:
        reasons.append("untraversable_slope_fraction>0.25")

    # Geometry score excludes total relief by design: a single smooth plane is
    # not difficult merely because its two corners have different elevations.
    if source_profile == "als":
        # Typical ALS point spacing cannot support a 0.4 m roughness claim.
        # Do not let interpolation or flight-line texture masquerade as
        # vehicle-scale microterrain; retain only scales ALS observes.
        score = (
            np.clip(metrics["detrended_std_m"] / 0.45, 0.0, 1.0) * 35.0
            + np.clip(metrics["local_abs_p90_m"] / 0.12, 0.0, 1.0) * 30.0
            + np.clip(metrics["slope_over_20_fraction"] / 0.20, 0.0, 1.0) * 25.0
            + np.clip(metrics["slope_p95_deg"] / 30.0, 0.0, 1.0) * 10.0
        )
        score_profile = "als_observable_scales_v1"
    else:
        score = (
            np.clip(metrics["detrended_std_m"] / 0.45, 0.0, 1.0) * 30.0
            + np.clip(metrics["local_abs_p90_m"] / 0.12, 0.0, 1.0) * 25.0
            + np.clip(metrics["micro_abs_p90_m"] / 0.04, 0.0, 1.0) * 15.0
            + np.clip(metrics["slope_over_20_fraction"] / 0.20, 0.0, 1.0) * 20.0
            + np.clip(metrics["slope_p95_deg"] / 30.0, 0.0, 1.0) * 10.0
        )
        score_profile = "uls_multiscale_v1"
    score = float(score)
    if reasons:
        grade = None
        quality = "reject"
    else:
        quality = "pass"
        if score < 25.0:
            grade = "easy"
        elif score < 55.0:
            grade = "medium"
        elif score < 80.0:
            grade = "hard"
        else:
            grade = "extreme"

    return {
        "path": path,
        "policy_version": POLICY_VERSION,
        "quality": quality,
        "grade": grade,
        "geometry_score": score,
        "geometry_score_profile": score_profile,
        "reasons": reasons,
        "metrics": metrics,
        "planner_calibration_required": True,
    }


def main():
    args = parse_args()
    results = [evaluate(path) for path in expand_inputs(args.inputs)]
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    print("quality grade    score  valid density rough  local slope95 >20%  file / rejection")
    for result in results:
        metrics = result.get("metrics", {})
        print(
            "{:<7} {:<8} {:>5.1f} {:>5.3f} {:>7.1f} {:>5.3f} {:>5.3f} {:>7.1f} {:>5.1f}  {}{}".format(
                result["quality"],
                result["grade"] or "-",
                result.get("geometry_score", float("nan")),
                metrics.get("valid_fraction", float("nan")),
                metrics.get("source_surface_density_points_per_m2", float("nan")),
                metrics.get("detrended_std_m", float("nan")),
                metrics.get("local_abs_p90_m", float("nan")),
                metrics.get("slope_p95_deg", float("nan")),
                100.0 * metrics.get("slope_over_20_fraction", float("nan")),
                os.path.basename(result["path"]),
                " | " + ", ".join(result["reasons"]) if result["reasons"] else "",
            )
        )


if __name__ == "__main__":
    main()
