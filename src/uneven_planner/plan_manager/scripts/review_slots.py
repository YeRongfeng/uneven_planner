#!/usr/bin/env python3
"""Resolve canonical map slots from the append-only review records."""

import datetime
import json
import pickle
from collections import OrderedDict
from pathlib import Path

PATH_MAP_SLACK_SEC = 2.0


SPLITS = ("train", "val")


def read_jsonl(path):
    path = Path(path)
    records = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                records.append((line_number, json.loads(line)))
    return records


def _slot_value(value, fallback_split=None):
    if isinstance(value, dict):
        split = value.get("split", fallback_split)
        index = value.get("map_index", value.get("slot_index"))
    else:
        split = fallback_split
        index = value
    if split not in SPLITS or not isinstance(index, int) or index < 0:
        return None
    return split, index


def fill_slot(record):
    value = record.get("fills_slot")
    if value is None and "slot_index" in record:
        value = record.get("slot_index")
    return _slot_value(value, record.get("split"))


def returned_slot(record):
    decision = record.get("decision")
    if decision not in {"return", "returned", "slot_return"}:
        return None
    value = record.get("slot")
    if value is None:
        value = record.get("map_index", record.get("slot_index"))
    return _slot_value(value, record.get("split"))


def _source_key(record):
    value = record.get("source_path")
    if value:
        return str(Path(value).expanduser().resolve())
    return f"source:{record.get('source_id', '')}"


def _event_time(record):
    return str(record.get("timestamp_utc", ""))


def _newer(existing, candidate):
    if existing is None:
        return candidate
    return candidate if _event_time(candidate[1]) >= _event_time(existing[1]) else existing


def _deletion_target(record):
    if record.get("decision") != "delete":
        return None
    timestamp = (record.get("target_timestamp_utc")
                 or record.get("review_timestamp_utc"))
    source_id = record.get("source_id")
    if not isinstance(timestamp, str) or not timestamp or not source_id:
        return None
    return str(source_id), timestamp


def _deleted_targets(review_records):
    result = {}
    for line_number, record in review_records:
        target = _deletion_target(record)
        if target is not None:
            result[target] = _newer(result.get(target), (line_number, record))
    return result


def _baseline_records(review_records, source_splits=None):
    """Assign old approval records using the builder's historical ordering."""
    deleted_targets = _deleted_targets(review_records)
    grouped = OrderedDict()
    for line_number, record in review_records:
        if record.get("decision") != "approve" or fill_slot(record) is not None:
            continue
        target = (record.get("source_id", ""), record.get("timestamp_utc", ""))
        if target in deleted_targets:
            continue
        split = record.get("split")
        if split is None and source_splits:
            split = source_splits.get(record.get("source_id"))
        if split not in SPLITS:
            continue
        grouped.setdefault(_source_key(record), []).append((line_number, record))

    next_index = {split: 0 for split in SPLITS}
    result = {}
    for source_records in grouped.values():
        for line_number, record in source_records:
            split = record.get("split")
            if split is None and source_splits:
                split = source_splits.get(record.get("source_id"))
            key = (split, next_index[split])
            effective = dict(record)
            effective["split"] = split
            result[key] = {
                "status": "occupied",
                "line_number": line_number,
                "record": effective,
            }
            next_index[split] += 1
    return result


def resolve_slots(review_path, replacement_path=None, source_splits=None):
    """Return occupied, filled, and pending canonical slots.

    A ``return`` review event creates a pending slot.  An approval carrying
    ``fills_slot`` takes that exact slot.  A ``delete`` event removes its
    target approval without creating a new slot.  Existing canonical
    replacement records are treated as pending because those maps were
    generated without a new human approval.
    """
    review_records = read_jsonl(review_path)
    replacement_records = read_jsonl(replacement_path) if replacement_path else []
    deleted_targets = _deleted_targets(review_records)
    baseline = _baseline_records(review_records, source_splits)

    returns = {}
    for line_number, record in review_records:
        slot = returned_slot(record)
        if slot is not None:
            returns[slot] = _newer(
                returns.get(slot), (line_number, record))
    for line_number, record in replacement_records:
        if record.get("status") in {"filled", "approved", "accepted"}:
            continue
        slot = _slot_value(record.get("map_index"), record.get("split"))
        if slot is not None:
            returns[slot] = _newer(
                returns.get(slot), (line_number, record))

    fills = {}
    for line_number, record in review_records:
        slot = fill_slot(record)
        if slot is not None and record.get("decision") == "approve":
            target = (record.get("source_id", ""),
                      record.get("timestamp_utc", ""))
            if target in deleted_targets:
                continue
            fills[slot] = _newer(fills.get(slot), (line_number, record))

    keys = set(baseline) | set(returns) | set(fills)
    slots = {}
    for key in keys:
        baseline_entry = baseline.get(key)
        return_entry = returns.get(key)
        fill_entry = fills.get(key)
        if fill_entry is not None and (
                return_entry is None
                or _event_time(fill_entry[1]) >= _event_time(return_entry[1])):
            slots[key] = {
                "status": "occupied",
                "line_number": fill_entry[0],
                "record": fill_entry[1],
                "filled": True,
            }
        elif return_entry is not None:
            slots[key] = {
                "status": "pending",
                "line_number": return_entry[0],
                "record": return_entry[1],
            }
        elif baseline_entry is not None:
            slots[key] = dict(baseline_entry)
        else:
            slots[key] = {
                "status": "pending",
                "line_number": None,
                "record": {},
            }

    return {
        "slots": slots,
        "baseline": baseline,
        "open_slots": open_slots(slots),
        "review_records": review_records,
        "replacement_records": replacement_records,
        "deleted_targets": deleted_targets,
    }


def open_slots(slots):
    result = []
    for (split, index), entry in sorted(
            slots.items(), key=lambda item: (SPLITS.index(item[0][0]), item[0][1])):
        if entry.get("status") != "pending":
            continue
        record = entry.get("record") or {}
        result.append({
            "key": f"{split}/map_{index:03d}",
            "split": split,
            "map_index": index,
            "source_id": record.get("source_id", ""),
            "center_xy": record.get("previous_center_xy")
                or record.get("center_xy"),
            "timestamp_utc": record.get("timestamp_utc", ""),
            "reason": record.get("reason", ""),
        })
    return result


def active_records(state):
    result = []
    for key, entry in sorted(
            state["slots"].items(),
            key=lambda item: (SPLITS.index(item[0][0]), item[0][1])):
        if entry.get("status") == "occupied":
            result.append((key, entry["line_number"], entry["record"]))
    return result


def filled_records(state):
    """Occupied slots that currently come from a human fill, not the baseline."""
    result = []
    for key, line_number, record in active_records(state):
        if fill_slot(record) is not None or state["slots"][key].get("filled"):
            result.append((key, line_number, record))
    return result


def current_round_filled_records(state):
    """Fills in this 打回/补位 round: occupying fills newer than the latest return."""
    last_return = ""
    for _, record in state.get("review_records") or []:
        if returned_slot(record) is not None:
            last_return = max(last_return, _event_time(record))
    for _, record in state.get("replacement_records") or []:
        if record.get("status") in {"filled", "approved", "accepted"}:
            continue
        if _slot_value(record.get("map_index"), record.get("split")) is not None:
            last_return = max(last_return, _event_time(record))
    if not last_return:
        return []
    result = []
    for key, line_number, record in filled_records(state):
        if _event_time(record) > last_return:
            result.append((key, line_number, record))
    return result


def env_results_match_canonical(dataset_root, split, index, canonical_map_path,
                                expected_paths=1):
    """True if this env already has a current-map test or a 打回 of this map."""
    env_dir = Path(dataset_root) / split / f"env{int(index):06d}"
    canonical = Path(canonical_map_path) if canonical_map_path else None
    if canonical is None or not canonical.is_file() or not env_dir.is_dir():
        return False
    map_mtime = canonical.stat().st_mtime
    marker = env_dir / "needs_return.json"
    if marker.is_file() and marker.stat().st_mtime >= map_mtime:
        return True
    paths = list(env_dir.glob("path_*.p"))
    if len(paths) < int(expected_paths):
        return False
    newest = max(path.stat().st_mtime for path in paths)
    return newest >= map_mtime


def _canonical_scene_and_split(canonical_map_path, expected_split=None):
    path = Path(canonical_map_path)
    scene = path.stem
    split = expected_split
    if split not in SPLITS:
        parent = path.parent.name
        split = parent if parent in SPLITS else None
    return scene, split


def existing_paths_belong_to_canonical(env_dir, canonical_map_path,
                                       expected_split=None,
                                       slack_sec=PATH_MAP_SLACK_SEC):
    """True if existing path_*.p files were generated on this canonical map.

    No paths is a match (nothing to conflict). A replaced canonical file
    (newer than map.p) or a map.p that names a different source is not.
    """
    env_dir = Path(env_dir)
    paths = list(env_dir.glob("path_*.p"))
    if not paths:
        return True
    canonical = Path(canonical_map_path) if canonical_map_path else None
    if canonical is None or not canonical.is_file():
        return False
    map_file = env_dir / "map.p"
    if not map_file.is_file():
        return False
    try:
        with map_file.open("rb") as stream:
            map_data = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, AttributeError, EOFError):
        return False
    if not isinstance(map_data, dict):
        return False
    selected = canonical.resolve()
    source_ok = False
    stored = map_data.get("external_map_path")
    if stored:
        try:
            source_ok = Path(stored).resolve() == selected
        except OSError:
            source_ok = False
    if not source_ok:
        source_map = map_data.get("source_map") or {}
        scene, split = _canonical_scene_and_split(
            canonical_map_path, expected_split)
        source_ok = source_map.get("scene") == scene and (
            split is None or source_map.get("split") == split)
    if not source_ok:
        return False
    map_mtime = map_file.stat().st_mtime
    canonical_mtime = canonical.stat().st_mtime
    if canonical_mtime > map_mtime + slack_sec:
        return False
    return True


def filled_slots_needing_work(state, dataset_root, split_map_paths,
                              test_generation=False, expected_by_split=None):
    """Filled slots that still need only-filled generation or testing.

    Maps whose current canonical file is already covered by a successful
    test or a 打回 of this map version are skipped.
    """
    dataset_root = Path(dataset_root)
    expected_by_split = expected_by_split or {}
    result = []
    for key, line_number, record in filled_records(state):
        split, index = key
        expected = int(expected_by_split.get(split, 1))
        paths = split_map_paths.get(split) or []
        canonical = paths[index] if index < len(paths) else ""
        if test_generation:
            if env_results_match_canonical(
                    dataset_root, split, index, canonical, expected):
                continue
        else:
            env_dir = dataset_root / split / f"env{int(index):06d}"
            if (env_dir / "needs_return.json").is_file():
                continue
            n_paths = (
                len(list(env_dir.glob("path_*.p"))) if env_dir.is_dir() else 0)
            if n_paths >= expected:
                continue
        result.append((key, line_number, record))
    return result


def _env_index_from_name(name):
    text = str(name or "")
    if text.startswith("env") and text[3:].isdigit():
        return int(text[3:])
    return None


def return_context_from_needs_return(marker_path):
    marker_path = Path(marker_path)
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    split = payload.get("split") or marker_path.parent.parent.name
    map_index = payload.get("env_id", payload.get("map_index"))
    if not isinstance(map_index, int):
        map_index = _env_index_from_name(marker_path.parent.name)
    map_path = payload.get("map_path") or ""
    metadata = {}
    if map_path:
        sidecar = Path(map_path).with_suffix(".json")
        if sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    manual = metadata.get("manual_review") or {}
    if not isinstance(map_index, int):
        map_index = metadata.get("canonical_map_index")
    if split not in SPLITS or not isinstance(map_index, int) or map_index < 0:
        raise ValueError(f"cannot resolve slot from {marker_path}")
    center = metadata.get("source_center_xy")
    if not isinstance(center, list):
        center = manual.get("center_xy")
    yaw = metadata.get("crop_yaw_deg")
    if yaw is None:
        yaw = manual.get("yaw_deg", 0.0)
    return {
        "split": split,
        "map_index": int(map_index),
        "source_id": manual.get("source_id", ""),
        "source_path": metadata.get("source_file") or manual.get("source_path")
        or map_path,
        "previous_center_xy": center,
        "previous_yaw_deg": float(yaw or 0.0),
        "size_m": float(manual.get("size_m") or metadata.get("patch_size_m") or 20.0),
        "scene_id": f"{split}/env{int(map_index):06d}",
        "display_id": f"{split}/env{int(map_index):06d}",
        "reason": payload.get("reason") or "test_generation: no_valid_trajectory",
        "marker_mtime": marker_path.stat().st_mtime,
    }


def _event_timestamp(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def append_return_record(review_file, context):
    """Write one return event unless that slot is already pending.

    A leftover needs_return.json from before a human fill must not reopen
    the slot. A marker newer than the fill means this map version failed.
    """
    review_file = Path(review_file)
    slot = (context["split"], int(context["map_index"]))
    latest_fill = None
    for _, record in reversed(read_jsonl(review_file)):
        if record.get("decision") == "approve" and fill_slot(record) == slot:
            latest_fill = record
            break
        if returned_slot(record) == slot:
            return record, False
    if latest_fill is not None:
        marker_mtime = _event_timestamp(context.get("marker_mtime"))
        fill_time = _event_timestamp(latest_fill.get("timestamp_utc"))
        if (marker_mtime is not None and fill_time is not None
                and marker_mtime <= fill_time):
            return latest_fill, False
    record = {
        "source_id": context.get("source_id", ""),
        "decision": "return",
        "split": context["split"],
        "map_index": int(context["map_index"]),
        "center_xy": context.get("previous_center_xy"),
        "size_m": context.get("size_m", 20.0),
        "yaw_deg": context.get("previous_yaw_deg", 0.0),
        "source_path": context.get("source_path", ""),
        "scene_id": context.get("scene_id", ""),
        "reason": context.get("reason") or f"8765: {context.get('display_id', '')}",
        "timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
    }
    review_file.parent.mkdir(parents=True, exist_ok=True)
    with review_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record, True


def append_returns_for_dataset(dataset_root, review_file):
    """Turn every env*/needs_return.json into a 8767 pending slot."""
    dataset_root = Path(dataset_root)
    written = []
    reused = []
    for split in SPLITS:
        for marker in sorted((dataset_root / split).glob("env*/needs_return.json")):
            context = return_context_from_needs_return(marker)
            record, created = append_return_record(review_file, context)
            (written if created else reused).append({
                "split": context["split"],
                "map_index": context["map_index"],
                "record": record,
            })
    return {"written": written, "already_open": reused}
