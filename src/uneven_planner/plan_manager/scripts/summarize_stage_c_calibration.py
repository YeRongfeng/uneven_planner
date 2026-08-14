#!/usr/bin/env python3
"""Summarize per-map Stage-C manifests without changing acceptance outcomes."""

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from terrain_map_quality import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_root")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--paths-per-map", type=int, default=3,
                        help="Requested saved trajectories per map")
    parser.add_argument(
        "--groups", default="easy,medium,hard",
        help="Comma-separated calibration result directories to summarize")
    return parser.parse_args()


def summarize_grade(root, grade, paths_per_map):
    manifests = sorted(glob.glob(os.path.join(
        root, grade, "scene_*", "dataset", "experiment_manifest.json")))
    rows = []
    for manifest_path in manifests:
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        source_path = manifest["source_maps"][0]["path"]
        quality = evaluate(os.path.splitext(source_path)[0] + ".npz")
        attempts = [
            attempt for attempt in manifest["attempts"]
            if attempt["phase"] == "train"
        ]
        initiated_path_ids = sorted({attempt["path_id"] for attempt in attempts})
        saved = [attempt for attempt in attempts
                 if attempt["outcome"] == "saved"]
        first_attempt_successes = sum(
            any(attempt["path_id"] == path_id
                and attempt["retry_index"] == 0
                and attempt["outcome"] == "saved"
                for attempt in attempts)
            for path_id in initiated_path_ids
        )
        failure_outcomes = Counter(
            attempt["outcome"] for attempt in attempts
            if attempt["outcome"] != "saved")
        map_application = manifest["map_applications"][-1]
        rows.append({
            "scene": os.path.basename(os.path.dirname(
                os.path.dirname(manifest_path))),
            "source_map": source_path,
            "status": manifest["status"],
            "geometry_score": quality["geometry_score"],
            "saved_paths": len(saved),
            "initiated_path_slots": len(initiated_path_ids),
            "planning_attempts": len(attempts),
            "first_attempt_successes": first_attempt_successes,
            "failure_outcomes": dict(failure_outcomes),
            "occupied_xy": map_application["occupied_xy"],
            "occupied_se2": map_application["occupied_se2"],
            "saved_stability_margins_m": [
                attempt["minimum_stability_margin_m"] for attempt in saved
            ],
        })

    margins = [margin for row in rows
               for margin in row["saved_stability_margins_m"]]
    requested_paths = paths_per_map * len(rows)
    attempts = sum(row["planning_attempts"] for row in rows)
    saved_paths = sum(row["saved_paths"] for row in rows)
    initiated_slots = sum(row["initiated_path_slots"] for row in rows)
    first_successes = sum(row["first_attempt_successes"] for row in rows)
    failures = sum((Counter(row["failure_outcomes"]) for row in rows), Counter())
    occupancy = [row["occupied_xy"] for row in rows]
    return {
        "grade": grade,
        "maps": len(rows),
        "maps_completed": sum(row["status"] == "completed" for row in rows),
        "maps_rejected": sum(row["status"] == "map_rejected" for row in rows),
        "requested_paths": requested_paths,
        "saved_paths": saved_paths,
        "initiated_path_slots": initiated_slots,
        "planning_attempts": attempts,
        "first_attempt_successes": first_successes,
        "map_completion_fraction": (
            sum(row["status"] == "completed" for row in rows) / len(rows)
            if rows else 0.0),
        "requested_path_completion_fraction": (
            saved_paths / requested_paths if requested_paths else 0.0),
        "attempt_yield": saved_paths / attempts if attempts else 0.0,
        "first_attempt_success_fraction": (
            first_successes / initiated_slots if initiated_slots else 0.0),
        "failure_outcomes": dict(failures),
        "geometry_score_mean": float(np.mean(
            [row["geometry_score"] for row in rows])) if rows else None,
        "occupied_xy_median": float(np.median(occupancy)) if occupancy else None,
        "stability_margin_median_m": (
            float(np.median(margins)) if margins else None),
        "stability_margin_min_m": min(margins) if margins else None,
        "scenes": rows,
    }


def main():
    args = parse_args()
    groups = [value.strip() for value in args.groups.split(",")
              if value.strip()]
    if not groups:
        raise ValueError("groups must not be empty")
    result = {
        "protocol": {
            "requested_paths_per_map": args.paths_per_map,
            "consecutive_failure_limit_per_path": 5,
            "note": "Fixed seeds and profile mixture; start/goal pairs differ by map.",
        },
        "grades": [
            summarize_grade(args.calibration_root, grade, args.paths_per_map)
            for grade in groups
        ],
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
