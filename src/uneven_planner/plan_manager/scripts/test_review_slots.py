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

from review_slots import active_records, filled_records, resolve_slots


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


if __name__ == "__main__":
    unittest.main()
