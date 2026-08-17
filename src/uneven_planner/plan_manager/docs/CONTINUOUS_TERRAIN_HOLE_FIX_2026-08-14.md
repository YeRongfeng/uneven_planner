# Continuous terrain hole fix — 2026-08-14

## Decision

Plane-fit RMSE is no longer an acceptance criterion and no longer changes map
validity.  Local non-planarity is terrain, not missing data.

The map now distinguishes two concepts:

- `valid_mask`: the finite surface delivered to the network and planner;
- `observed_mask`: cells with a direct local fit from the source point cloud.

Cells without enough direct support are completed from neighbouring terrain and
remain valid.  This prevents source sampling gaps from becoming artificial
obstacles or NaNs.

## Recorded reconstruction parameters

- Source: every finite XYZ return in the mother map; LAS classification is
  retained as provenance only and is not used to select the fitted surface.
- ALS local-fit radius: 0.9 m.
- ALS direct-fit minimum: 5 source points.
- Fit: distance-weighted `z=ax+by+c`.
- Robustness: one MAD refinement pass; it may change the estimate but never
  invalidates a cell.  If too few points remain, the all-point estimate is used.
- Gap completion: nearest finite initialization followed by harmonic relaxation
  while holding directly observed elevations fixed.
- Normals: finite differences on the completed elevation surface, normalized and
  upward-facing.
- Obstacles: raw returns above the fitted surface are projected to the same
  grid as `obstacle_mask`/`obstacle_height`; they are not inserted into the
  ground PCD.
- Planner PCD: dense 0.05 m bilinear resampling of that same completed surface,
  so UnevenMap and the network consume the same geometry.
- RMSE threshold: none.
- RMSE quality gate: none.

Map size and resolution remain independent runtime parameters and were not part
of this fix.

## Targeted test

Rebuilt scenes: `000, 001, 003, 007, 008, 019, 022, 025, 027, 033, 034`.

- Previous invalid cells: 476 total.
- New invalid cells: 0.
- All 11 maps passed the remaining finite-value, geometry, connectivity, and
  traversability checks.
- Scene 008 maximum neighbour step changed from the initial unguarded-fit result
  of 1.156 m to 0.197 m after requiring five points for a direct fit.
- Scene 034 changed from 0.758 m to 0.315 m by the same rule.

Targeted output:
`dataset/tests/desert_hole_fix_20260814/`

## Existing desert sample rebuild

The 164 complete scenes from the interrupted v1 desert sampling run were rebuilt
under:

`dataset/public_terrain_20m_v2/desert/sampled_scenes/train/`

Results:

- scenes: 164;
- invalid cells: 0;
- non-finite elevation cells: 0;
- quality pass: 164/164;
- minimum direct-observation fraction: 0.9797;
- maximum adjacent-cell elevation step: 0.4995 m;
- grades: 76 easy, 70 medium, 18 hard, 0 extreme.

The old v1 output was retained unchanged for comparison.  Scenes 164–178 in v1
had empty metadata because the earlier generation was interrupted, so they were
not treated as complete scenes and were not rebuilt.
