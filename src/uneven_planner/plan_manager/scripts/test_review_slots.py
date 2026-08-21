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
    current_round_filled_records, env_results_match_canonical,
    existing_paths_belong_to_canonical, filled_records,
    filled_slots_needing_work, resolve_slots,
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

    def test_current_round_filled_records_ignore_older_fills(self):
        state = self.resolve([
            {"source_id": "old", "decision": "approve", "split": "train",
             "timestamp_utc": "2026-01-01T00:00:00+00:00"},
            {"source_id": "old", "decision": "return", "split": "train",
             "map_index": 0, "timestamp_utc": "2026-01-01T00:01:00+00:00"},
            {"source_id": "oldfill", "decision": "approve", "split": "train",
             "fills_slot": {"split": "train", "map_index": 0},
             "timestamp_utc": "2026-01-01T00:02:00+00:00"},
            {"source_id": "old", "decision": "return", "split": "val",
             "map_index": 1, "timestamp_utc": "2026-01-02T00:00:00+00:00"},
            {"source_id": "newfill", "decision": "approve", "split": "val",
             "fills_slot": {"split": "val", "map_index": 1},
             "timestamp_utc": "2026-01-02T00:01:00+00:00"},
        ])
        self.assertEqual(
            [key for key, _, _ in filled_records(state)],
            [("train", 0), ("val", 1)])
        self.assertEqual(
            [key for key, _, _ in current_round_filled_records(state)],
            [("val", 1)])

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

    def test_passed_fill_is_skipped_by_only_filled_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "maps" / "val" / "map_003.pcd"
            canonical.parent.mkdir(parents=True)
            canonical.write_text("pcd", encoding="utf-8")
            env = root / "val" / "env000003"
            env.mkdir(parents=True)
            path = env / "path_000000.p"
            path.write_text("path", encoding="utf-8")
            import os
            os.utime(canonical, (1_000_000_000, 1_000_000_000))
            os.utime(path, (1_000_000_100, 1_000_000_100))
            self.assertTrue(env_results_match_canonical(
                root, "val", 3, canonical, 1))
            review = root / "manual_regions.jsonl"
            review.write_text(
                json.dumps({
                    "source_id": "a.laz", "decision": "approve", "split": "val",
                    "fills_slot": {"split": "val", "map_index": 3},
                    "timestamp_utc": "2026-01-02T00:00:00+00:00",
                }) + "\n",
                encoding="utf-8")
            state = resolve_slots(review)
            needed = filled_slots_needing_work(
                state, root, {"val": [None, None, None, str(canonical)]},
                test_generation=True, expected_by_split={"val": 1})
            self.assertEqual(needed, [])

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

    def test_existing_paths_belong_to_canonical(self):
        import os
        import pickle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "val" / "map_003.pcd"
            other = root / "train" / "map_004.pcd"
            canonical.parent.mkdir(parents=True)
            other.parent.mkdir(parents=True)
            canonical.write_text("pcd", encoding="utf-8")
            other.write_text("other", encoding="utf-8")
            env = root / "env000003"
            env.mkdir()
            self.assertTrue(existing_paths_belong_to_canonical(
                env, canonical, expected_split="val"))

            (env / "path_0.p").write_bytes(b"path")
            self.assertFalse(existing_paths_belong_to_canonical(
                env, canonical, expected_split="val"))

            with (env / "map.p").open("wb") as stream:
                pickle.dump({
                    "source_map": {"scene": "map_003", "split": "val"},
                }, stream)
            os.utime(canonical, (1_000_000_000, 1_000_000_000))
            os.utime(env / "map.p", (1_000_000_100, 1_000_000_100))
            os.utime(env / "path_0.p", (1_000_000_200, 1_000_000_200))
            self.assertTrue(existing_paths_belong_to_canonical(
                env, canonical, expected_split="val"))

            os.utime(canonical, (1_000_000_300, 1_000_000_300))
            self.assertFalse(existing_paths_belong_to_canonical(
                env, canonical, expected_split="val"))

            os.utime(canonical, (1_000_000_000, 1_000_000_000))
            with (env / "map.p").open("wb") as stream:
                pickle.dump({
                    "source_map": {"scene": "map_004", "split": "train"},
                    "external_map_path": str(other),
                }, stream)
            os.utime(env / "map.p", (1_000_000_100, 1_000_000_100))
            self.assertFalse(existing_paths_belong_to_canonical(
                env, canonical, expected_split="val"))

            with (env / "map.p").open("wb") as stream:
                pickle.dump({
                    "source_map": {"scene": "map_003", "split": "val"},
                    "external_map_path": str(canonical),
                }, stream)
            os.utime(env / "map.p", (1_000_000_100, 1_000_000_100))
            self.assertTrue(existing_paths_belong_to_canonical(
                env, canonical, expected_split="val"))


if __name__ == "__main__":
    unittest.main()
