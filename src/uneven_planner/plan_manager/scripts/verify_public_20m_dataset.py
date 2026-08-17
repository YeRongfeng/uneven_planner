#!/usr/bin/env python3
"""Verify one generated public 20 m terrain-domain dataset."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def fail(errors, message):
    errors.append(message)


def load_pickle(path, errors):
    try:
        with path.open("rb") as stream:
            return pickle.load(stream)
    except Exception as exc:  # Report every bad artifact in one audit pass.
        fail(errors, f"{path}: cannot load pickle: {exc}")
        return None


def verify_split(root, split, scenes, paths_per_scene, expected_site, errors):
    split_root = root / split
    env_dirs = sorted(path for path in split_root.glob("env*" ) if path.is_dir())
    if len(env_dirs) != scenes:
        fail(errors, f"{split}: expected {scenes} environment directories, found {len(env_dirs)}")

    map_count = 0
    path_count = 0
    valid_fractions = []
    observed_sites = set()
    for env_dir in env_dirs:
        map_path = env_dir / "map.p"
        if not map_path.is_file():
            fail(errors, f"{map_path}: missing")
            continue
        map_data = load_pickle(map_path, errors)
        if not isinstance(map_data, dict):
            fail(errors, f"{map_path}: expected dictionary")
            continue
        map_count += 1

        tensor = np.asarray(map_data.get("tensor"))
        if tensor.shape != (100, 100, 4):
            fail(errors, f"{map_path}: tensor shape {tensor.shape}, expected (100, 100, 4)")
        elif not np.isfinite(tensor).all():
            fail(errors, f"{map_path}: tensor contains non-finite values")

        valid_mask = np.asarray(map_data.get("valid_mask"))
        if valid_mask.shape != (100, 100):
            fail(errors, f"{map_path}: valid_mask shape {valid_mask.shape}, expected (100, 100)")
        else:
            valid_fraction = float(np.mean(valid_mask.astype(bool)))
            valid_fractions.append(valid_fraction)
            if valid_fraction < 1.0:
                fail(errors, f"{map_path}: completed surface still contains invalid cells")

        observed_mask = np.asarray(map_data.get("observed_mask"))
        if observed_mask.shape != (100, 100):
            fail(errors, f"{map_path}: observed_mask shape {observed_mask.shape}, expected (100, 100)")

        if "obstacle_mask" in map_data or "obstacle_height" in map_data:
            fail(errors, f"{map_path}: dataset map must not contain an obstacle layer")

        if map_data.get("dataset_phase") != split:
            fail(errors, f"{map_path}: dataset_phase is {map_data.get('dataset_phase')!r}, expected {split!r}")
        sample = (map_data.get("crop") or {}).get("mother_map_sample") or {}
        site_id = sample.get("site_id")
        if site_id:
            observed_sites.add(site_id)
        if site_id != expected_site:
            fail(errors, f"{map_path}: site_id is {site_id!r}, expected {expected_site!r}")

        paths = sorted(env_dir.glob("path_*.p"))
        if len(paths) != paths_per_scene:
            fail(errors, f"{env_dir}: expected {paths_per_scene} paths, found {len(paths)}")
        for path in paths:
            path_data = load_pickle(path, errors)
            if not isinstance(path_data, dict):
                fail(errors, f"{path}: expected dictionary")
                continue
            trajectory = np.asarray(path_data.get("path"))
            if trajectory.shape != (100, 3):
                fail(errors, f"{path}: trajectory shape {trajectory.shape}, expected (100, 3)")
            elif not np.isfinite(trajectory).all():
                fail(errors, f"{path}: trajectory contains non-finite values")
            if path_data.get("planner_map_version") != map_data.get("planner_map_version"):
                fail(errors, f"{path}: planner map version does not match map.p")
            path_count += 1

    expected_path_count = scenes * paths_per_scene
    if path_count != expected_path_count:
        fail(errors, f"{split}: expected {expected_path_count} readable paths, found {path_count}")
    return {
        "maps": map_count,
        "paths": path_count,
        "sites": sorted(observed_sites),
        "minimum_valid_fraction": min(valid_fractions) if valid_fractions else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--scenes", type=int, required=True)
    parser.add_argument("--train-paths", type=int, required=True)
    parser.add_argument("--val-paths", type=int, required=True)
    parser.add_argument("--train-site", required=True)
    parser.add_argument("--val-site", required=True)
    args = parser.parse_args()

    errors = []
    if args.train_site == args.val_site:
        fail(errors, "train and validation site IDs are identical")
    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "train": verify_split(
            args.dataset_root, "train", args.scenes, args.train_paths, args.train_site, errors
        ),
        "val": verify_split(
            args.dataset_root, "val", args.scenes, args.val_paths, args.val_site, errors
        ),
    }

    manifest_paths = sorted(args.dataset_root.glob("experiment_manifest_worker_*.json"))
    if not manifest_paths:
        manifest_paths = [args.dataset_root / "experiment_manifest.json"]
    manifest_statuses = []
    manifest_environment_ids = set()
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text())
            status = manifest.get("status")
            manifest_statuses.append(status)
            if status != "completed":
                fail(errors, f"{manifest_path}: status is {status!r}, expected 'completed'")
            manifest_environment_ids.update(
                int(record["environment_id"])
                for record in manifest.get("map_applications", [])
                if "environment_id" in record
            )
        except Exception as exc:
            fail(errors, f"{manifest_path}: cannot load manifest: {exc}")
    expected_environment_ids = set(range(args.scenes))
    if manifest_environment_ids != expected_environment_ids:
        fail(
            errors,
            "experiment manifests cover environment IDs "
            f"{sorted(manifest_environment_ids)}, expected {sorted(expected_environment_ids)}",
        )
    summary["experiment_manifests"] = {
        "count": len(manifest_paths),
        "statuses": manifest_statuses,
        "environment_ids": sorted(manifest_environment_ids),
    }

    summary["status"] = "passed" if not errors else "failed"
    summary["errors"] = errors
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
