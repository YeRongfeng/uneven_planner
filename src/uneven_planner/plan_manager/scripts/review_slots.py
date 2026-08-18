#!/usr/bin/env python3
"""Resolve canonical map slots from the append-only review records."""

import datetime
import json
from collections import OrderedDict
from pathlib import Path


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
    }


def append_return_record(review_file, context):
    """Write one return event unless that slot is already pending."""
    review_file = Path(review_file)
    slot = (context["split"], int(context["map_index"]))
    for _, record in reversed(read_jsonl(review_file)):
        if record.get("decision") == "approve" and fill_slot(record) == slot:
            break
        if returned_slot(record) == slot:
            return record, False
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
