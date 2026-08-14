# Geometry-grade Stage-C calibration (2026-08-13)

## Protocol

- 10 independent 20 m scenes per geometry grade.
- Retained crop centres from one mother map are at least 20 m apart.
- Easy: 10 PflegeSchoenau scenes, accepted in 81 sampling attempts.
- Medium: 10 PferdstriebSued scenes, accepted in 211 attempts.
- Hard: 8 Zugmantel scenes accepted in 500 attempts plus 2
  PflegeSchoenau scenes accepted in 301 attempts. Thresholds were not relaxed.
- Every scene requested three train trajectories.
- A path slot could use at most five consecutive planning attempts. Exhausting
  that budget marked the map as planner-rejected and did not substitute another
  map in this calibration.
- Seeds were fixed per grade and scene. Distance/heading profiles followed the
  same deterministic mixture, but start/goal pairs were not identical across
  maps. This is a distributional planner calibration, not a fixed-pair benchmark.

## Results

| Geometry grade | Maps completed | Paths saved | Saved / attempts | First-attempt success | Median occupied XY | Median saved-path stability margin |
|---|---:|---:|---:|---:|---:|---:|
| easy | 9/10 | 29/30 | 29/61 (47.5%) | 13/30 (43.3%) | 0.5 | 11.53 m |
| medium | 10/10 | 30/30 | 30/60 (50.0%) | 14/30 (46.7%) | 87 | 1.76 m |
| hard | 8/10 | 25/30 | 25/58 (43.1%) | 10/27 initiated (37.0%) | 312 | 1.02 m |

Failure outcomes recorded by the generator were:

- easy: 32 planning failures, 0 strict stability failures;
- medium: 27 planning failures, 3 strict stability failures;
- hard: 29 planning failures, 4 strict stability failures.

The easy rejection occurred after two paths had been saved and the third path
slot exhausted its five-attempt budget. The two hard rejections saved one and
zero paths respectively before exhausting a path-slot budget.

## Interpretation

The geometry score is useful, but it is not a planner-success score. Attempt
yield was similar across grades and one easy map was rejected while every
medium map completed. With only ten maps per grade, that is not evidence for
moving the grade boundaries.

The grade does produce a clear change in planner-facing map structure:
median occupied XY cells rose from 0.5 to 87 to 312, while median continuous
stability margin of accepted paths fell from 11.53 m to 1.76 m to 1.02 m.
Hard maps also produced more strict stability failures and the lowest map
completion rate.

Therefore geometry grade and Stage-C outcome remain separate labels:

- Stage A decides whether the reconstructed map is valid.
- Stage B describes terrain geometry as easy, medium, hard, or extreme.
- Stage C records planner acceptance, attempts, occupancy, stability margin,
  and failure outcomes.

For production, retain the existing Stage-A/B thresholds. Use two canonical
candidates per requested easy or medium environment and three for hard. A
candidate that exhausts the Stage-C retry budget is quarantined, not relabelled
or made easier by changing the acceptance thresholds.

Machine-readable results are stored at
`dataset/tests/grade_calibration_20260813/stage_c_summary.json`.
