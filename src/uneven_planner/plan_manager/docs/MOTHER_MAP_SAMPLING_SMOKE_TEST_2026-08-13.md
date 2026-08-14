# Mother-map online sampling smoke test (2026-08-13)

## Scope

This test checks the proposed data path before integrating it into full-scale
dataset generation:

1. Treat a classified LAZ point cloud as a mother map.
2. Randomly sample a centre and yaw, then crop an exact 20 m x 20 m patch.
3. Reject patches by source density, raw-cell coverage, fit quality, runtime
   height contract, connected traversability, and terrain geometry metrics.
4. Send an accepted patch through the existing A* + ALM trajectory pipeline.

The deterministic source was
`2021-08-17_ULS_PflegeSchoenau.laz`, with seed `20260814`.

## Sampling result

- Mother map class-2 returns loaded: 12,921,405.
- Accepted patches: 5 in 14 attempts (35.7%).
- Accepted geometry: 2 easy and 3 medium patches.
- Every accepted network map is 20 m x 20 m at 0.2 m resolution
  (`100 x 100` cells).
- Every accepted planner PCD is the same robust fitted surface resampled at
  0.05 m (`400 x 400`, 160,000 points when fully valid).
- The manifest records every sampled centre, yaw, metric, and rejection reason:
  `dataset/tests/mother_map_sampling/pflege_surface/sampling_manifest.json`.

## Planner A/B result

The medium patch `scene_000` has geometry score 27.8, 100% valid cells,
236 source ground points/m2, 0.110 m detrended roughness, 0.060 m local
roughness, and 8.7 degree P95 slope.

With surface-consistent raw classified returns as the planner PCD:

- A* found a path in all 12 attempts.
- ALM reached its iteration limit in all 12 attempts.
- Saved trajectories: 0/3; the canonical patch was rejected.

With a 0.05 m dense resampling of the robust fitted surface:

- Train trajectories: 2/2 saved.
- Validation trajectories: 1/1 saved.
- All saved trajectories have shape `(100, 3)`.
- The first validation candidate was correctly rejected for terminal spatial
  overlap; the replacement succeeded.
- Final retry statistics: one planning rejection, zero unstable trajectories,
  zero map regenerations, and zero canonical-map rejections.

Artifacts are under
`dataset/tests/mother_map_sampling/planner_surface_medium/`.

## Hard and extreme counter-tests

The same representation was then tested on deterministic Zugmantel patches.

- Hard patch, score 65.0: 167 occupied XY cells and 3,177 occupied SE(2)
  states. All 3 train and 1 validation trajectories were eventually saved;
  one hard task required a replacement pose pair. Successful trajectories had
  minimum continuous-stability margins from 0.43 m to 6.41 m.
- Extreme patch, score 97.2: 1,319 occupied XY cells and 42,163 occupied SE(2)
  states. No trajectory survived 20 attempts: 19 planning/optimization
  failures and one continuous-stability failure. It was correctly quarantined.

The dense fitted surface therefore does not collapse difficult terrain into a
flat or universally traversable map. The preliminary grades produce distinct
planner behaviour; extreme patches remain stress tests rather than routine
training scenes.

## Fitting-scale ablation

The 0.6 m, 0.35 m, and 0.25 m local-plane radii were compared at identical
centres and yaw angles. Relative to 0.6 m, the 0.35 m surface changed elevation
by only 7 mm P95 on the medium patch, 16 mm on hard, and 28 mm on extreme,
while increasing the respective geometry scores from 27.8/65.0/97.2 to
29.4/68.8/99.8. The hard-map occupied XY count increased from 167 to 234.

The 0.35 m hard patch still produced 2 train and 1 validation trajectories,
with two planning failures and two strict stability failures before replacement
tasks succeeded. The 0.25 m radius retained still more structure, but at the
40 points/m2 source-density floor its expected neighbourhood count is only
about eight points, equal to the minimum fitting requirement. It is therefore
kept as an ablation setting rather than the default.

Raw-return absolute residual P95 against the fitted surface was 3.3 cm on the
medium patch, 6.1 cm on hard, and 9.1 cm on extreme. This mixes measurement
scatter and unresolved real micro-terrain, so raw points should be retained as
uncertainty evidence even though they are not suitable as UnevenMap input.

## Conclusion

The mother-map approach is viable, but map quality must be evaluated on the
same robust surface that UnevenMap consumes. Raw classified returns remain
useful as evidence for density and fit residuals; they must not be passed
directly to UnevenMap because measurement scatter changes its local covariance
surface and can make ALM fail even when the evaluated 100 x 100 map passes.

The production path should therefore perform online random cropping, quality
gating, and robust fitting, publish a 0.05 m dense fitted-surface PCD to the
planner, and save the corresponding 0.2 m grid for training. A 0.35 m fit
radius is the current candidate default, not a permanent constant. Dataset-scale
generation should not start until this sampler is integrated into the generator
and a multi-patch planner-calibration run establishes per-grade success-rate
thresholds.
