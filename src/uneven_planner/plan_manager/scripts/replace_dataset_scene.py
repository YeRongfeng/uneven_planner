#!/usr/bin/env python3
"""Replace one canonical slot and, when requested, its generated environment."""

import argparse
import datetime
import json
import math
import os
import shlex
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from prepare_laz_terrain_map import (
    SIM_DEFAULT_COARSE_RESOLUTION,
    SIM_DEFAULT_OUTPUT_RESOLUTION,
    SIM_DEFAULT_VOXEL_SIZE,
    YRF_DEFAULT_ENVELOPE_OUTLIER,
    YRF_DEFAULT_GROUND_BAND_ABOVE,
    YRF_DEFAULT_GROUND_BAND_BELOW,
    YRF_DEFAULT_LOWER_ENVELOPE_FILTER_SIZE,
    write_binary_pcd,
)
from sample_laz_mother_map import (
    build_surface,
    grid_coverage,
    load_surface_points,
    metadata_for,
    save_npz,
    select_local_points,
)
from terrain_map_quality import evaluate


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--map-index", required=True, type=int)
    parser.add_argument("--final-root", type=Path,
                        help="Existing final dataset root to regenerate")
    parser.add_argument("--environment-id", type=int)
    parser.add_argument("--paths-per-env", type=int, default=0)
    parser.add_argument("--replacement-file", type=Path)
    parser.add_argument("--domain", default="terrain")
    parser.add_argument("--reason", default="")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--ros-port", type=int, default=11412)
    return parser.parse_args()


def value_from(metadata, processing, name, default):
    value = processing.get(name)
    return default if value is None else value


def build_processing_args(metadata, source_path, domain, split):
    processing = metadata.get("processing") or {}
    profile = str(processing.get("source_profile", "als")).strip().lower()
    source_profile_defaults = {
        "als": {"fit_radius": 0.9, "surface_cell_size": 1.0,
                "direct_fit_min_points": 5, "coverage_resolution": 1.0},
        "uls": {"fit_radius": 0.35, "surface_cell_size": 0.5,
                "direct_fit_min_points": 8, "coverage_resolution": 0.2},
    }
    defaults = source_profile_defaults.get(profile, source_profile_defaults["als"])
    return SimpleNamespace(
        domain=domain,
        input_laz=str(source_path),
        site_id=metadata.get("site_id", f"{domain}_{split}"),
        source_url=metadata.get("source_url", ""),
        license=metadata.get("license", ""),
        crs=metadata.get("source_crs", ""),
        source_profile=profile,
        size=float(metadata.get("patch_size_m", 20.0)),
        resolution=SIM_DEFAULT_OUTPUT_RESOLUTION,
        fit_radius=float(value_from(
            metadata, processing, "fit_radius_m", defaults["fit_radius"])),
        surface_cell_size=float(value_from(
            metadata, processing, "surface_cell_size_m",
            defaults["surface_cell_size"])),
        ground_band_below=float(value_from(
            metadata, processing, "ground_band_below_m",
            YRF_DEFAULT_GROUND_BAND_BELOW)),
        ground_band_above=float(value_from(
            metadata, processing, "ground_band_above_m",
            YRF_DEFAULT_GROUND_BAND_ABOVE)),
        envelope_outlier=float(value_from(
            metadata, processing, "envelope_outlier_m",
            YRF_DEFAULT_ENVELOPE_OUTLIER)),
        direct_fit_min_points=int(value_from(
            metadata, processing, "direct_fit_minimum_points",
            defaults["direct_fit_min_points"])),
        planner_surface_resolution=float(value_from(
            metadata, processing, "planner_surface_resolution_m", 0.05)),
        raw_below_surface_tolerance=float(value_from(
            metadata, processing, "raw_below_surface_tolerance_m", 1.0)),
        raw_above_surface_tolerance=float(value_from(
            metadata, processing, "raw_above_surface_tolerance_m", 50.0)),
        coverage_resolution=float(value_from(
            metadata, processing, "center_sampling_resolution_m",
            defaults["coverage_resolution"])),
        fit_method="sim_elevation",
        yrf_coarse_resolution=SIM_DEFAULT_COARSE_RESOLUTION,
        yrf_voxel_size=SIM_DEFAULT_VOXEL_SIZE,
        yrf_lower_envelope_filter_size=int(value_from(
            metadata, processing, "yrf_lower_envelope_filter_size",
            YRF_DEFAULT_LOWER_ENVELOPE_FILTER_SIZE)),
        center_sampling="replacement",
        center_sampling_resolution=float(value_from(
            metadata, processing, "center_sampling_resolution_m",
            defaults["coverage_resolution"])),
    )


def eligible_support_cells(points, size, fit_radius, resolution):
    xy = np.asarray(points[:, :2], dtype=np.float64)
    lower = xy.min(axis=0)
    upper = xy.max(axis=0)
    margin = 0.5 * size * math.sqrt(2.0) + fit_radius
    origin = np.floor(lower / resolution) * resolution
    indices = np.floor((xy - origin) / resolution).astype(np.int64)
    width = int(indices[:, 0].max()) + 1
    linear = np.unique(indices[:, 1] * width + indices[:, 0])
    cells = np.column_stack((linear % width, linear // width))
    centers = origin + (cells.astype(np.float64) + 0.5) * resolution
    eligible = np.all(
        (centers >= lower + margin + 0.5 * resolution)
        & (centers <= upper - margin - 0.5 * resolution), axis=1)
    return origin, cells[eligible]


def choose_candidate(points, old_center, args, seed):
    resolution = max(0.2, min(1.0, float(args.coverage_resolution)))
    origin, cells = eligible_support_cells(
        points, args.size, args.fit_radius, resolution)
    if len(cells) == 0:
        raise RuntimeError("mother map has no supported replacement crop")
    rng = np.random.default_rng(seed)
    candidates = []
    old_center = np.asarray(old_center, dtype=np.float64)
    for _ in range(300):
        cell = cells[int(rng.integers(0, len(cells)))]
        center = origin + (cell.astype(np.float64) + rng.random(2)) * resolution
        yaw = float(rng.uniform(-math.pi, math.pi))
        distance = float(np.linalg.norm(center - old_center))
        candidates.append((distance, center, yaw))
    # Prefer a genuinely different crop, but keep a fallback for a small tile
    # whose usable support is narrower than one patch.
    candidates.sort(key=lambda item: item[0], reverse=True)
    for require_separation in (True, False):
        for distance, center, yaw in candidates:
            if require_separation and distance < args.size:
                continue
            exact, padded = select_local_points(
                points, center, yaw, args.size, args.fit_radius)
            if len(exact) < 50 or len(padded) == 0:
                continue
            coverage = grid_coverage(
                exact, args.size, args.coverage_resolution)
            if coverage < 0.80:
                continue
            try:
                surface = build_surface(exact, padded, args)
            except (ValueError, RuntimeError, IndexError):
                continue
            if (not surface["valid"].all() or
                    not np.isfinite(surface["planner_xyz"]).all()):
                continue
            return center, yaw, exact, coverage, surface
    raise RuntimeError("could not build a supported replacement crop")


def write_replacement(canonical_root, split, map_index, metadata, source_path,
                      domain, seed):
    args = build_processing_args(metadata, source_path, domain, split)
    source_points, source_count, source_bounds, class_counts = (
        load_surface_points(str(source_path)))
    old_center = metadata.get("source_center_xy")
    if not isinstance(old_center, list) or len(old_center) != 2:
        raise ValueError("canonical metadata has no source_center_xy")
    center, yaw, exact, coverage, surface = choose_candidate(
        source_points, old_center, args, seed)
    source_info = (source_count, source_bounds, class_counts)
    split_dir = canonical_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    stem = split_dir / f"map_{map_index:03d}"
    temporary = split_dir / f".replacement_map_{map_index:03d}_{os.getpid()}"
    save_npz(str(temporary) + ".npz", surface, args, center, yaw)
    write_binary_pcd(
        str(temporary) + ".pcd", surface["planner_xyz"], surface["planner_normals"])
    replacement_metadata = metadata_for(
        surface, len(exact), coverage, center, yaw, source_info, args)
    temporary_json = Path(str(temporary) + ".json")
    # terrain_map_quality reads the processing profile and density fields from
    # the sidecar, so write the preliminary metadata before evaluating it.
    temporary_json.write_text(
        json.dumps(replacement_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    quality = evaluate(str(temporary) + ".npz")
    quality.pop("path", None)
    replacement_metadata.update({
        "canonical_map": True,
        "split": split,
        "canonical_map_index": map_index,
        "replacement": {
            "replaced_map": f"{split}/map_{map_index:03d}",
            "previous_center_xy": metadata.get("source_center_xy"),
            "previous_yaw_deg": metadata.get("crop_yaw_deg"),
            "selection_policy": "same mother map, new supported random crop",
        },
        "quality": quality,
        "manual_review": {
            "source_id": metadata.get("manual_review", {}).get(
                "source_id", source_path.name),
            "split": split,
            "center_xy": center.astype(float).tolist(),
            "size_m": args.size,
            "yaw_deg": math.degrees(yaw),
            "stats": {
                "source_points_in_crop": int(len(exact)),
                "source_support_coverage": float(coverage),
                "replacement": True,
            },
        },
    })
    temporary_json.write_text(
        json.dumps(replacement_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    for suffix in (".npz", ".pcd", ".json"):
        os.replace(str(temporary) + suffix, str(stem) + suffix)
    return {
        "split": split,
        "map_index": map_index,
        "source_id": replacement_metadata["manual_review"]["source_id"],
        "source_path": str(source_path),
        "center_xy": replacement_metadata["source_center_xy"],
        "yaw_deg": replacement_metadata["crop_yaw_deg"],
        "size_m": args.size,
        "stats": replacement_metadata["manual_review"]["stats"],
        "quality": quality,
        "previous_center_xy": metadata.get("source_center_xy"),
        "previous_yaw_deg": metadata.get("crop_yaw_deg"),
    }


def append_replacement(path, record, reason):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.update({
        "schema": "canonical-scene-replacement-v1",
        "timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "reason": reason,
    })
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def master_is_available(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        return False


def run_final_generation(args, canonical_path, domain):
    if args.final_root is None:
        return
    if args.environment_id is None:
        raise ValueError("environment-id is required when final-root is set")
    if args.paths_per_env <= 0:
        raise ValueError("paths-per-env must be positive for final replacement")
    if args.ros_port <= 0:
        raise ValueError("ros-port must be positive")
    final_root = args.final_root.expanduser().absolute()
    final_root.mkdir(parents=True, exist_ok=True)
    path_count = str(args.paths_per_env)
    launch_values = {
        "parallel_workers": "1",
        "num_environments": "1",
        "train_paths_per_env": path_count,
        "val_paths_per_env": path_count,
        "stop_after_train": "true" if args.split == "train" else "false",
        "start_phase": args.split,
        "terminate_after_generator": "true",
        "dataset_dir": str(final_root),
        "start_env_id": str(args.environment_id),
        "external_map_path": str(canonical_path),
        "external_map_paths": str(canonical_path),
        "train_external_map_paths": str(canonical_path),
        "val_external_map_paths": str(canonical_path),
        "external_map_format": "pcd",
        "external_domain": domain,
        "external_map_is_canonical": "true",
        "canonical_maps_per_environment": "1",
        "canonical_primary_scene_count": "1",
        "canonical_pool_start_env_id": str(args.environment_id),
        "target_map_size": "20.0",
        "target_resolution": "0.1",
        "external_map_physical_size": "0.0",
        "external_map_min_physical_size": "0.0",
        "scale_external_map_z": "false",
        "external_map_fixed_yaw_deg": "0.0",
        "crop_min_coverage": "0.97",
        "crop_max_attempts": "1",
        "generation_random_seed": str(
            args.seed if args.seed >= 0 else int(time.time())),
        "max_path_retries_before_regenerate": "30",
        "enable_rviz": "false",
    }
    command = ["roslaunch", "plan_manager",
               "terrain_dataset_generation_parallel.launch"]
    command.extend(f"{key}:={shlex.quote(value)}"
                   for key, value in launch_values.items())
    shell_command = (
        f"source {shlex.quote(str(WORKSPACE_ROOT / 'devel/setup.bash'))} && "
        f"exec {' '.join(command)}")
    environment = os.environ.copy()
    environment["ROS_MASTER_URI"] = f"http://127.0.0.1:{args.ros_port}"
    own_master = None
    if not master_is_available(args.ros_port):
        own_master = subprocess.Popen(
            ["roscore", "-p", str(args.ros_port)],
            cwd=str(WORKSPACE_ROOT), env=environment)
        deadline = time.monotonic() + 20.0
        while not master_is_available(args.ros_port):
            if own_master.poll() is not None:
                raise RuntimeError("roscore exited before becoming available")
            if time.monotonic() >= deadline:
                own_master.terminate()
                raise RuntimeError("timed out waiting for the replacement ROS master")
            time.sleep(0.2)
    try:
        subprocess.run(
            ["bash", "-lc", shell_command],
            cwd=str(WORKSPACE_ROOT), env=environment, check=True)
    finally:
        if own_master is not None and own_master.poll() is None:
            own_master.terminate()
            try:
                own_master.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                own_master.kill()
    env_dir = final_root / args.split / f"env{args.environment_id:06d}"
    map_path = env_dir / "map.p"
    paths = sorted(env_dir.glob("path_*.p"))
    if not map_path.is_file() or len(paths) != args.paths_per_env:
        raise RuntimeError(
            f"replacement finished without the expected final environment: "
            f"map={map_path.is_file()} paths={len(paths)}/{args.paths_per_env}")


def main():
    args = parse_args()
    if args.map_index < 0:
        raise ValueError("map-index must be non-negative")
    canonical_root = args.canonical_root.expanduser().absolute()
    stem = canonical_root / args.split / f"map_{args.map_index:03d}"
    metadata_path = Path(str(stem) + ".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_value = metadata.get("source_file")
    if not source_value:
        source_value = (metadata.get("manual_review") or {}).get("source_path")
    if not source_value:
        raise ValueError(f"canonical metadata has no source file: {metadata_path}")
    source_path = Path(source_value).expanduser().absolute()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    seed = args.seed if args.seed >= 0 else int(time.time_ns() & 0x7FFFFFFF)
    replacement = write_replacement(
        canonical_root, args.split, args.map_index, metadata, source_path,
        args.domain, seed)
    replacement_file = (
        args.replacement_file.expanduser().absolute()
        if args.replacement_file else
        WORKSPACE_ROOT / "dataset" / "reviews" / args.domain
        / "canonical_replacements.jsonl")
    append_replacement(replacement_file, replacement, args.reason)
    canonical_path = stem.with_suffix(".pcd")
    run_final_generation(args, canonical_path, args.domain)
    print(json.dumps({
        "canonical_map": str(canonical_path),
        "replacement_file": str(replacement_file),
        "final_environment": (
            str(args.final_root / args.split / f"env{args.environment_id:06d}")
            if args.final_root is not None else None),
        "seed": seed,
        "quality": replacement["quality"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
