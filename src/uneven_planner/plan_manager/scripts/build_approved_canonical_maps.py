#!/usr/bin/env python3
"""Turn manually approved LAZ crops into canonical 20 m map inputs.

The approval file is the source of truth for crop centres and yaw angles.  This
script does not sample new positions: every output map is built from one
``approve`` record, using the same continuous-surface and raw-return policy as
the normal LAZ sampler.  The resulting PCD/NPZ/JSON triplets are suitable for
the canonical-map mode of ``terrain_dataset_generator.py``.
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sample_laz_mother_map import (
    build_surface,
    grid_coverage,
    load_surface_points,
    metadata_for,
    save_npz,
    select_local_points,
)
from prepare_laz_terrain_map import (
    YRF_DEFAULT_COARSE_RESOLUTION,
    YRF_DEFAULT_ENVELOPE_OUTLIER,
    YRF_DEFAULT_GROUND_BAND_ABOVE,
    YRF_DEFAULT_GROUND_BAND_BELOW,
    YRF_DEFAULT_LOWER_ENVELOPE_FILTER_SIZE,
    YRF_DEFAULT_VOXEL_SIZE,
    write_binary_pcd,
)
from terrain_map_quality import evaluate
from review_slots import active_records, resolve_slots


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--train-source-id", action="append",
        help="Legacy fallback: source_id assigned to the train split")
    parser.add_argument(
        "--val-source-id", action="append",
        help="Legacy fallback: source_id assigned to the val split")
    parser.add_argument("--domain", default="terrain")
    parser.add_argument("--train-site-id", default="terrain_train")
    parser.add_argument("--val-site-id", default="terrain_val")
    parser.add_argument("--train-source-profile", default="als",
                        choices=("als", "uls"))
    parser.add_argument("--val-source-profile", default="als",
                        choices=("als", "uls"))
    parser.add_argument("--size", type=float, default=20.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--fit-radius", type=float, default=0.9)
    parser.add_argument("--surface-cell-size", type=float, default=1.0)
    parser.add_argument("--ground-band-below", type=float,
                        default=YRF_DEFAULT_GROUND_BAND_BELOW)
    parser.add_argument("--ground-band-above", type=float,
                        default=YRF_DEFAULT_GROUND_BAND_ABOVE)
    parser.add_argument("--envelope-outlier", type=float,
                        default=YRF_DEFAULT_ENVELOPE_OUTLIER)
    parser.add_argument("--direct-fit-min-points", type=int, default=5)
    parser.add_argument("--planner-surface-resolution", type=float, default=0.05)
    parser.add_argument("--raw-below-surface-tolerance", type=float, default=1.0)
    parser.add_argument("--raw-above-surface-tolerance", type=float, default=50.0)
    parser.add_argument("--coverage-resolution", type=float, default=1.0)
    parser.add_argument("--yrf-coarse-resolution", type=float,
                        default=YRF_DEFAULT_COARSE_RESOLUTION)
    parser.add_argument("--yrf-voxel-size", type=float,
                        default=YRF_DEFAULT_VOXEL_SIZE)
    parser.add_argument("--yrf-lower-envelope-filter-size", type=int,
                        default=YRF_DEFAULT_LOWER_ENVELOPE_FILTER_SIZE)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Clear the selected non-empty output directory before rebuilding")
    parser.add_argument(
        "--replacement-file", type=Path,
        help="JSONL log of returned canonical slots")
    return parser.parse_args()


def processing_args(source_path, cli, site_id, source_profile):
    return SimpleNamespace(
        domain=cli.domain,
        input_laz=str(source_path),
        site_id=site_id,
        source_url="",
        license="",
        crs="",
        source_profile=source_profile,
        size=cli.size,
        resolution=cli.resolution,
        fit_radius=cli.fit_radius,
        surface_cell_size=cli.surface_cell_size,
        ground_band_below=cli.ground_band_below,
        ground_band_above=cli.ground_band_above,
        envelope_outlier=cli.envelope_outlier,
        direct_fit_min_points=cli.direct_fit_min_points,
        planner_surface_resolution=cli.planner_surface_resolution,
        raw_below_surface_tolerance=cli.raw_below_surface_tolerance,
        raw_above_surface_tolerance=cli.raw_above_surface_tolerance,
        coverage_resolution=cli.coverage_resolution,
        fit_method="yrf_ground",
        yrf_coarse_resolution=cli.yrf_coarse_resolution,
        yrf_voxel_size=cli.yrf_voxel_size,
        yrf_lower_envelope_filter_size=cli.yrf_lower_envelope_filter_size,
        center_sampling="manual_approval",
        center_sampling_resolution=cli.coverage_resolution,
    )


def write_one(record, line_number, split, index, source_points, source_info,
              cli, output_dir, site_id):
    source_path = Path(record["source_path"]).resolve()
    source_profile = (cli.train_source_profile if split == "train"
                      else cli.val_source_profile)
    args = processing_args(source_path, cli, site_id, source_profile)
    center = np.asarray(record["center_xy"], dtype=np.float64)
    yaw = math.radians(float(record["yaw_deg"]))
    exact, padded = select_local_points(
        source_points, center, yaw, cli.size, cli.fit_radius)
    if len(exact) == 0:
        raise RuntimeError(
            f"approval line {line_number} has no source points in its crop")
    coverage = grid_coverage(exact, cli.size, cli.coverage_resolution)
    surface = build_surface(exact, padded, args)
    if not np.isfinite(surface["planner_xyz"]).all():
        raise RuntimeError(
            f"approval line {line_number} produced non-finite planner points")
    if not surface["valid"].all():
        raise RuntimeError(
            f"approval line {line_number} produced an incomplete canonical grid")

    stem = output_dir / split / f"map_{index:03d}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_npz(str(stem) + ".npz", surface, args, center, yaw)
    write_binary_pcd(
        str(stem) + ".pcd", surface["planner_xyz"], surface["planner_normals"])
    metadata = metadata_for(
        surface, len(exact), coverage, center, yaw, source_info, args)
    metadata.update({
        "canonical_map": True,
        "split": split,
        "canonical_map_index": index,
        "approval_record_line": line_number,
        "manual_review": {
            "source_id": record["source_id"],
            "split": record.get("split"),
            "center_xy": record["center_xy"],
            "size_m": record["size_m"],
            "yaw_deg": record["yaw_deg"],
            "stats": record.get("stats", {}),
        },
    })
    if record.get("replacement"):
        metadata["replacement"] = record["replacement"]
    if record.get("fills_slot"):
        metadata["filled_slot"] = record["fills_slot"]
    metadata_path = Path(str(stem) + ".json")
    # terrain_map_quality reads the sidecar next to the NPZ. Write the
    # processing metadata before evaluating so the profile and densities are
    # not mistaken for missing/default values.
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    quality = evaluate(str(stem) + ".npz")
    quality.pop("path", None)
    metadata["quality"] = quality
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return {
        "line": line_number,
        "split": split,
        "source_id": record["source_id"],
        "source_path": str(source_path),
        "map_path": str(stem.with_suffix(".pcd").resolve()),
        "center_xy": record["center_xy"],
        "yaw_deg": record["yaw_deg"],
        "source_points_in_crop": int(len(exact)),
        "source_support_coverage": float(coverage),
        "quality": quality,
    }


def main():
    cli = parse_args()
    review_file = cli.review_file.resolve()
    output_dir = cli.output_dir.resolve()

    train_ids = set(cli.train_source_id or ())
    val_ids = set(cli.val_source_id or ())
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(
            "A source_id cannot be both train and val: "
            + ", ".join(sorted(overlap)))

    replacement_file = (
        cli.replacement_file.resolve()
        if cli.replacement_file else
        review_file.parent / "canonical_replacements.jsonl")
    source_splits = {
        source_id: "train" for source_id in train_ids}
    source_splits.update({source_id: "val" for source_id in val_ids})
    slot_state = resolve_slots(review_file, replacement_file, source_splits)
    if slot_state["open_slots"]:
        labels = ", ".join(slot["key"] for slot in slot_state["open_slots"])
        raise RuntimeError(
            "review file contains map slots waiting for a human replacement: "
            + labels)
    active = active_records(slot_state)
    if not active:
        raise RuntimeError("The review file contains no approved regions")

    ordered_records = []
    for (split, index), line_number, record in active:
        if not record.get("source_path"):
            raise ValueError(
                f"Approval line {line_number} has no source_path")
        center = record.get("center_xy")
        if (not isinstance(center, list) or len(center) != 2 or
                not all(np.isfinite(float(value)) for value in center)):
            raise ValueError(
                f"Approval line {line_number} has no finite center_xy")
        effective = dict(record)
        effective["split"] = split
        site_id = (cli.train_site_id if split == "train"
                   else cli.val_site_id)
        ordered_records.append((
            line_number, effective, split, site_id, index))

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not cli.overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_indices = {
        split: sum(1 for _, _, record_split, _, _ in ordered_records
                   if record_split == split)
        for split in ("train", "val")}

    results = []
    source_cache = {}
    for line_number, record, split, site_id, index in ordered_records:
        source_path = Path(record["source_path"]).resolve()
        if source_path not in source_cache:
            print(f"Loading {source_path}", flush=True)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            source_points, source_count, source_bounds, class_counts = (
                load_surface_points(str(source_path)))
            source_cache[source_path] = (
                source_points, (source_count, source_bounds, class_counts))
        source_points, source_info = source_cache[source_path]
        result = write_one(
            record, line_number, split, index, source_points, source_info,
            cli, output_dir, site_id)
        results.append(result)
        quality = result["quality"]
        print(
            f"{split} map_{index:03d}: line={line_number}, "
            f"points={result['source_points_in_crop']}, "
            f"coverage={result['source_support_coverage']:.3f}, "
            f"quality={quality.get('quality')} grade={quality.get('grade')}",
            flush=True)

    manifest = {
        "policy": "terrain20m-v4-manual-approved-canonical",
        "domain": cli.domain,
        "train_site_id": cli.train_site_id,
        "val_site_id": cli.val_site_id,
        "train_source_profile": cli.train_source_profile,
        "val_source_profile": cli.val_source_profile,
        "review_file": str(review_file),
        "replacement_file": str(replacement_file),
        "replacement_count": len(slot_state["replacement_records"]),
        "parameters": {
            "size_m": cli.size,
            "resolution_m": cli.resolution,
            "fit_radius_m": cli.fit_radius,
            "surface_cell_size_m": cli.surface_cell_size,
            "ground_band_m": [cli.ground_band_below, cli.ground_band_above],
            "envelope_outlier_m": cli.envelope_outlier,
            "direct_fit_min_points": cli.direct_fit_min_points,
            "planner_surface_resolution_m": cli.planner_surface_resolution,
            "raw_below_surface_tolerance_m": cli.raw_below_surface_tolerance,
            "raw_above_surface_tolerance_m": cli.raw_above_surface_tolerance,
            "coverage_resolution_m": cli.coverage_resolution,
            "fit_method": "yrf_ground_reference",
            "yrf_coarse_resolution_m": cli.yrf_coarse_resolution,
            "yrf_voxel_size_m": cli.yrf_voxel_size,
            "yrf_lower_envelope_filter_size": (
                cli.yrf_lower_envelope_filter_size),
            "surface_source": "all finite XYZ returns; no LAS class filter",
        },
        "approved_count": len(results),
        "split_counts": {
            "train": global_indices["train"],
            "val": global_indices["val"],
        },
        "train_maps": [
            result["map_path"] for result in results if result["split"] == "train"],
        "val_maps": [
            result["map_path"] for result in results if result["split"] == "val"],
        "records": results,
    }
    (output_dir / "approved_canonical_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "approved_count": len(results),
        "train_maps": global_indices["train"],
        "val_maps": global_indices["val"],
        "manifest": str(output_dir / "approved_canonical_manifest.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
