#!/usr/bin/env python3
"""Finalize a terrain domain as ``domain/train`` and ``domain/val``.

The older generator kept canonical scene sidecars under ``sampled_scenes`` and
put the generated environments below ``trajectories``.  This finalizer copies
the two pieces of sidecar information that the viewer needs into each
``map.p`` before moving the environments to the domain root.
"""

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np


def clean_quality(value):
    if not isinstance(value, dict) or not value.get("quality"):
        return None
    result = dict(value)
    result.pop("path", None)
    return result


def load_quality_records(domain_root, split):
    manifest_path = (
        domain_root / "sampled_scenes" / split / "sampling_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing sampling manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {}
    for record in manifest.get("records", []):
        if record.get("status") != "accepted":
            continue
        scene = record.get("scene")
        quality = clean_quality(record.get("quality"))
        if scene and quality:
            records[scene] = quality
    return records


def source_sidecar(domain_root, split, map_data, scene_name):
    source_value = map_data.get("original_map_path")
    if source_value:
        candidate = Path(source_value).with_suffix(".npz")
        if candidate.is_file():
            return candidate
    candidate = (
        domain_root / "sampled_scenes" / split / f"{scene_name}.npz")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Cannot find canonical sidecar for {split}/{scene_name}")


def update_map(map_path, domain_root, split, quality_records):
    with map_path.open("rb") as stream:
        map_data = pickle.load(stream)
    if not isinstance(map_data, dict):
        raise ValueError(f"{map_path}: expected a dictionary")

    tensor = np.asarray(map_data.get("tensor"))
    if tensor.ndim != 3 or tensor.shape[2] < 4:
        raise ValueError(f"{map_path}: unexpected tensor shape {tensor.shape}")
    valid_mask = np.asarray(map_data.get("valid_mask"), dtype=bool)
    if valid_mask.shape != tensor.shape[:2]:
        raise ValueError(
            f"{map_path}: valid_mask {valid_mask.shape} does not match "
            f"tensor {tensor.shape[:2]}")

    scene_name = Path(map_data.get("original_map_path", map_path)).stem
    sidecar = source_sidecar(domain_root, split, map_data, scene_name)
    with np.load(sidecar) as source:
        if "observed_mask" not in source.files:
            raise ValueError(f"{sidecar}: missing observed_mask")
        observed_mask = np.asarray(source["observed_mask"], dtype=bool)
    if observed_mask.shape != tensor.shape[:2]:
        raise ValueError(
            f"{map_path}: observed_mask {observed_mask.shape} does not match "
            f"tensor {tensor.shape[:2]}")

    quality = clean_quality(map_data.get("quality"))
    if quality is None:
        quality = quality_records.get(scene_name)
    if quality is None:
        raise ValueError(
            f"{map_path}: no accepted quality record for {scene_name}")

    map_data["observed_mask"] = observed_mask
    map_data["quality"] = quality
    map_data["source_map"] = {
        "scene": scene_name,
        "split": split,
    }
    # The canonical source cache is removed after finalization.  Keep the
    # scene identity above, but do not leave a path that points at a deleted
    # intermediate file.
    map_data.pop("original_map_path", None)

    temporary_path = map_path.with_name(map_path.name + ".tmp")
    with temporary_path.open("wb") as stream:
        pickle.dump(map_data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, map_path)


def finalize_split(domain_root, split):
    old_root = domain_root / "trajectories" / split
    new_root = domain_root / split
    if not old_root.is_dir():
        raise FileNotFoundError(f"Missing generated split: {old_root}")
    if new_root.exists():
        raise FileExistsError(f"Destination already exists: {new_root}")

    map_paths = sorted(old_root.glob("env*/map.p"))
    if not map_paths:
        raise FileNotFoundError(f"No map.p files found under {old_root}")
    quality_records = load_quality_records(domain_root, split)
    for map_path in map_paths:
        update_map(map_path, domain_root, split, quality_records)
    shutil.move(str(old_root), str(new_root))
    return len(map_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain_root", type=Path)
    parser.add_argument(
        "--remove-intermediate", action="store_true",
        help="Remove sampled_scenes after both splits are self-contained")
    args = parser.parse_args()
    domain_root = args.domain_root.resolve()

    if (domain_root / "trajectories").is_dir() is False:
        raise FileNotFoundError(f"Missing trajectories directory: {domain_root / 'trajectories'}")
    counts = {
        split: finalize_split(domain_root, split)
        for split in ("train", "val")
    }

    trajectories_root = domain_root / "trajectories"
    for manifest in sorted(trajectories_root.glob("experiment_manifest*.json")):
        target = domain_root / manifest.name
        if target.exists():
            raise FileExistsError(f"Manifest destination already exists: {target}")
        shutil.move(str(manifest), str(target))
    if trajectories_root.exists():
        trajectories_root.rmdir()

    matplotlib_root = domain_root / "matplotlib"
    if matplotlib_root.exists():
        shutil.rmtree(matplotlib_root)

    if args.remove_intermediate:
        sampled_root = domain_root / "sampled_scenes"
        if sampled_root.exists():
            shutil.rmtree(sampled_root)

    for split, count in counts.items():
        final_maps = sorted((domain_root / split).glob("env*/map.p"))
        if len(final_maps) != count:
            raise RuntimeError(
                f"{split}: moved {count} maps but found {len(final_maps)}")
        for map_path in final_maps:
            with map_path.open("rb") as stream:
                map_data = pickle.load(stream)
            if "observed_mask" not in map_data or "quality" not in map_data:
                raise RuntimeError(f"{map_path}: final metadata is incomplete")

    print(json.dumps({"domain": str(domain_root), "maps": counts}, indent=2))


if __name__ == "__main__":
    main()
