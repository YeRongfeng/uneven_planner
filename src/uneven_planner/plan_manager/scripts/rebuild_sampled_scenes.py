#!/usr/bin/env python3
"""Rebuild existing sampled scenes with the continuous-surface policy."""

import argparse
import json
import math
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
from prepare_laz_terrain_map import save_preview, write_binary_pcd
from terrain_map_quality import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing scene_*.json")
    parser.add_argument("output_dir", type=Path, help="New directory for rebuilt scenes")
    parser.add_argument(
        "--scenes", default="",
        help="Comma-separated scene numbers, such as 000,001,008; empty rebuilds all")
    return parser.parse_args()


def selected_metadata(input_dir, selected):
    paths = sorted(input_dir.glob("scene_*.json"))
    if selected:
        wanted = {f"scene_{value.strip()}" for value in selected.split(",") if value.strip()}
        paths = [path for path in paths if path.stem in wanted]
        missing = wanted.difference(path.stem for path in paths)
        if missing:
            raise FileNotFoundError("Missing scenes: " + ", ".join(sorted(missing)))
    if not paths:
        raise RuntimeError("No scene metadata selected")
    loaded = []
    for path in paths:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if selected:
                raise RuntimeError(f"Cannot read {path}: {exc}") from exc
            print(f"Skipping incomplete metadata {path.name}: {exc}", flush=True)
            continue
        loaded.append((path, metadata))
    if not loaded:
        raise RuntimeError("No complete scene metadata selected")
    return loaded


def processing_args(metadata):
    processing = metadata["processing"]
    return SimpleNamespace(
        domain=metadata.get("domain", ""),
        input_laz=metadata["source_file"],
        site_id=metadata.get("site_id", ""),
        source_url=metadata.get("source_url", ""),
        license=metadata.get("license", ""),
        crs=metadata.get("source_crs", ""),
        source_profile=processing.get("source_profile", "als"),
        size=float(metadata["patch_size_m"]),
        resolution=float(metadata["resolution_m"]),
        fit_radius=float(processing["fit_radius_m"]),
        surface_cell_size=float(processing.get("surface_cell_size_m", 1.0)),
        ground_band_below=float(processing.get("ground_band_below_m", 0.25)),
        ground_band_above=float(processing.get("ground_band_above_m", 0.35)),
        envelope_outlier=float(processing.get("envelope_outlier_m", 0.75)),
        direct_fit_min_points=int(
            processing.get("direct_fit_minimum_points",
                           processing.get("min_neighbors", 5))),
        planner_surface_resolution=float(
            processing.get("planner_surface_resolution_m", 0.05)),
        raw_below_surface_tolerance=float(
            processing.get("raw_below_surface_tolerance_m", 1.0)),
        raw_above_surface_tolerance=float(
            processing.get("raw_above_surface_tolerance_m", 50.0)),
        coverage_resolution=float(
            metadata.get("source_support_resolution_m", 1.0)),
        center_sampling="uniform_support",
        center_sampling_resolution=float(
            processing.get("center_sampling_resolution_m", 1.0)),
    )


def write_scene(output_stem, surface, metadata, args, center, yaw):
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_npz(str(output_stem) + ".npz", surface, args, center, yaw)
    (output_stem.with_suffix(".json")).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_binary_pcd(
        str(output_stem) + ".pcd", surface["planner_xyz"], surface["planner_normals"])
    grid_xyz = np.column_stack((
        surface["local_x"].ravel(), surface["local_y"].ravel(),
        surface["elevation"].ravel()))
    write_binary_pcd(
        str(output_stem) + "_grid.pcd", grid_xyz,
        surface["normals"].reshape(-1, 3))
    save_preview(
        str(output_stem) + "_preview.png", surface["elevation"],
        surface["normals"], surface["neighbors"], args.size)


def main():
    cli = parse_args()
    scenes = selected_metadata(cli.input_dir, cli.scenes)
    source_files = {metadata["source_file"] for _, metadata in scenes}
    if len(source_files) != 1:
        raise RuntimeError("Selected scenes must come from one mother map")
    mother_map = next(iter(source_files))
    print(f"Loading mother map {mother_map}", flush=True)
    points, source_count, source_bounds, class_counts = load_surface_points(
        mother_map)
    source_info = (source_count, source_bounds, class_counts)

    records = []
    for index, (path, old_metadata) in enumerate(scenes, start=1):
        args = processing_args(old_metadata)
        center = np.asarray(old_metadata["source_center_xy"], dtype=np.float64)
        yaw = math.radians(float(old_metadata["crop_yaw_deg"]))
        exact, padded = select_local_points(
            points, center, yaw, args.size, args.fit_radius)
        surface = build_surface(exact, padded, args)
        support_resolution = float(
            old_metadata.get("source_support_resolution_m", 1.0))
        coverage = grid_coverage(exact, args.size, support_resolution)
        metadata = metadata_for(
            surface, len(exact), coverage, center, yaw, source_info, args)
        output_stem = cli.output_dir / path.stem
        write_scene(output_stem, surface, metadata, args, center, yaw)
        quality = evaluate(str(output_stem) + ".npz")
        record = {
            "scene": path.stem,
            "quality": quality["quality"],
            "grade": quality["grade"],
            "valid_fraction": quality["metrics"]["valid_fraction"],
            "observed_fraction": quality["metrics"]["observed_fraction"],
            "source_support_coverage": coverage,
        }
        records.append(record)
        print(
            f"[{index}/{len(scenes)}] {path.stem}: "
            f"valid={record['valid_fraction']:.4f}, "
            f"observed={record['observed_fraction']:.4f}", flush=True)

    manifest = {
        "policy": "terrain20m-v4-base4-rawpcd",
        "source_scene_dir": str(cli.input_dir.resolve()),
        "mother_map": mother_map,
        "scenes": records,
    }
    manifest_path = cli.output_dir / "rebuild_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(cli.output_dir.resolve()),
        "scene_count": len(records),
        "all_finite": all(record["valid_fraction"] == 1.0 for record in records),
        "manifest": str(manifest_path.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
