# Online mother-map pipeline test (2026-08-13)

## Purpose

Verify one complete invocation from independent LAZ mother maps through random
20 m sampling, quality gating, canonical planner loading, and trajectory
output. This is a wiring test, not planner-grade calibration.

## Parameters

- Train mother map: `2021-08-17_ULS_PflegeSchoenau.laz`
- Validation mother map: `2021-08-17_ULS_PferdstriebSued.laz`
- Sampling seeds: train `2026081401`, validation `2026081402`
- Planning seed: `2026081403`
- Accepted grades: easy, medium, hard
- Requested scenes: 1 train and 1 validation
- Requested trajectories: 2 train and 1 validation
- Sampling attempt budget: 80 per split
- Map: 20 m x 20 m, 0.2 m network grid
- Planner surface: 0.05 m spacing
- Local-plane fit radius: 0.35 m
- Canonical-map planning failure limit: 12 consecutive attempts
- ROS workers: 1

## Result

- Train sampling accepted an easy patch on attempt 1, score 23.2.
- Validation sampling accepted a medium patch on attempt 8, score 26.5.
- The train map loaded with 160,000 planner points and a 100 x 100 grid with
  100% valid cells.
- The validation map loaded with 159,920 planner points and a 100 x 100 grid
  with 99.95% valid cells.
- Saved trajectories: 2/2 train and 1/1 validation; every path has 100 points.
- Retry statistics: 3 planning failures, 0 stability failures, 0 empty
  trajectories, 0 watchdog timeouts, and 0 canonical-map rejections.
- Train and validation source paths in their saved maps point to different
  mother-map-derived canonical scenes.

Artifacts are under `dataset/tests/online_pipeline_20260813/`.

## Decision

The single-command production wiring works. The next run should be the planned
calibration batch rather than a 100-scene release: at least 10 accepted scenes
per easy/medium/hard grade, with train and validation split by survey/site.
