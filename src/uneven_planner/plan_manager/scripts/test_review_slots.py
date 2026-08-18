#!/usr/bin/env python3
"""Focused tests for manual-review slot transitions."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from review_slots import (
    active_records, append_return_record, append_returns_for_dataset,
    filled_records, resolve_slots,
)


class ReviewSlotTest(unittest.TestCase):
    def resolve(self, records):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review_path = root / "manual_regions.jsonl"
            review_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8")
            return resolve_slots(review_path)

    def test_return_and_fill_use_the_same_map_index(self):
        state = self.resolve([
            {"source_id": "source", "decision": "approve", "split": "val",
             "timestamp_utc": "2026-01-01T00:00:00+00:00"},
            {"source_id": "source", "decision": "return", "split": "val",
             "map_index": 26, "timestamp_utc": "2026-01-01T00:01:00+00:00"},
        ])
        self.assertEqual([slot["key"] for slot in state["open_slots"]],
                         ["val/map_026"])

        filled = self.resolve([
            {"source_id": "source", "decision": "approve", "split": "val",
             "timestamp_utc": "2026-01-01T00:00:00+00:00"},
            {"source_id": "source", "decision": "return", "split": "val",
             "map_index": 26, "timestamp_utc": "2026-01-01T00:01:00+00:00"},
            {"source_id": "replacement", "decision": "approve", "split": "val",
             "fills_slot": {"split": "val", "map_index": 26},
             "timestamp_utc": "2026-01-01T00:02:00+00:00"},
        ])
        self.assertEqual(filled["open_slots"], [])
        self.assertEqual(
            [key for key, _, _ in filled_records(filled)],
            [("val", 26)])

    def test_filled_records_ignore_baseline_approvals(self):
        state = self.resolve([
            {"source_id": "source", "decision": "approve", "split": "train",
             "timestamp_utc": "2026-01-01T00:00:00+00:00"},
            {"source_id": "source", "decision": "approve", "split": "val",
             "timestamp_utc": "2026-01-01T00:01:00+00:00"},
        ])
        self.assertEqual(filled_records(state), [])
        self.assertEqual(len(active_records(state)), 2)

    def test_delete_drops_extra_approval_without_opening_a_slot(self):
        state = self.resolve([
            {"source_id": "source", "decision": "approve", "split": "train",
             "timestamp_utc": "2026-01-01T00:00:00+00:00"},
            {"source_id": "source", "decision": "approve", "split": "train",
             "timestamp_utc": "2026-01-01T00:01:00+00:00"},
            {"source_id": "source", "decision": "delete",
             "target_timestamp_utc": "2026-01-01T00:01:00+00:00",
             "timestamp_utc": "2026-01-01T00:02:00+00:00"},
        ])
        self.assertEqual(state["open_slots"], [])
        self.assertEqual(
            [key for key, _, _ in active_records(state)],
            [("train", 0)])

    def test_append_return_is_idempotent_until_refilled(self):
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory) / "manual_regions.jsonl"
            first, created = append_return_record(review, {
                "split": "val", "map_index": 7, "display_id": "val/env000007",
            })
            second, created_again = append_return_record(review, {
                "split": "val", "map_index": 7, "display_id": "val/env000007",
            })
            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(first["map_index"], 7)
            self.assertEqual(second["timestamp_utc"], first["timestamp_utc"])

    def test_dataset_needs_return_markers_open_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = root / "val" / "env000003"
            env.mkdir(parents=True)
            canonical = root / "canonical" / "val" / "map_003.json"
            canonical.parent.mkdir(parents=True)
            canonical.write_text(json.dumps({
                "source_file": "/maps/a.laz",
                "source_center_xy": [1.0, 2.0],
                "crop_yaw_deg": 10.0,
                "manual_review": {"source_id": "a.laz", "size_m": 20},
            }) + "\n", encoding="utf-8")
            (env / "needs_return.json").write_text(json.dumps({
                "split": "val",
                "env_id": 3,
                "map_path": str(canonical.with_suffix(".pcd")),
                "reason": "no_valid_trajectory",
            }) + "\n", encoding="utf-8")
            review = root / "manual_regions.jsonl"
            result = append_returns_for_dataset(root, review)
            self.assertEqual(len(result["written"]), 1)
            state = resolve_slots(review)
            self.assertEqual(
                [slot["key"] for slot in state["open_slots"]],
                ["val/map_003"])

    def test_stale_needs_return_does_not_reopen_a_filled_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = root / "manual_regions.jsonl"
            review.write_text(
                json.dumps({
                    "source_id": "a.laz", "decision": "approve", "split": "val",
                    "fills_slot": {"split": "val", "map_index": 3},
                    "timestamp_utc": "2026-08-18T14:07:00+00:00",
                }) + "\n",
                encoding="utf-8")
            env = root / "val" / "env000003"
            env.mkdir(parents=True)
            marker = env / "needs_return.json"
            marker.write_text(json.dumps({
                "split": "val", "env_id": 3, "reason": "no_valid_trajectory",
            }) + "\n", encoding="utf-8")
            import os
            os.utime(marker, (1_000_000_000, 1_000_000_000))
            result = append_returns_for_dataset(root, review)
            self.assertEqual(result["written"], [])
            state = resolve_slots(review)
            self.assertEqual(state["open_slots"], [])
            self.assertEqual(
                [key for key, _, _ in filled_records(state)],
                [("val", 3)])


if __name__ == "__main__":
    unittest.main()
