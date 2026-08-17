# 20 m Terrain Map Acceptance Standard (v3-continuous)

This standard separates map validity, geometric difficulty, and planner
difficulty. A map must pass all three stages before entering a released
trajectory dataset. Visual review is an audit mechanism, not an acceptance
criterion.

## 1. Scope and fixed contract

- Metric 20 m x 20 m terrain, sampled on a 100 x 100 grid at 0.2 m.
- Source coordinates may be translated and source elevation may be offset.
  The x, y, or z axes must never be scaled to satisfy the map size.
- Terrain height and normals come from an explicitly documented
  geometry-only reconstruction over raw finite XYZ returns. LAS classification
  is provenance only and is not available in the deployed ROS point cloud.
- Surface reconstruction has two explicit acquisition profiles. ULS uses a
  0.35 m local-plane radius, at least 8 points for a direct fit, and at least 40 source
  points/m2. ALS uses a 0.9 m radius, at least 5 points for a direct fit, at least 4 source
  points/m2, and at least 97% occupied 1 m support cells. ALS preserves
  metre-scale landform but cannot be treated as evidence of 0.4 m microterrain.
  The profile and parameters are recorded for every map. Released splits must
  be balanced by profile; they must not silently mix reconstruction scales.
- Keep two representations: a 100 x 100 grid for network input and a dense
  resampling of the same completed fitted surface for UnevenMap's local ellipsoid
  fit. The current planner-cloud reference spacing is 0.05 m. Raw returns
  provide density and fit evidence but are not passed directly to UnevenMap. A
  one-point-per-cell grid is also too sparse for planner fitting.
- Above-ground returns are retained separately as `obstacle_mask` and
  `obstacle_height` on the same grid. The dense ground PCD must contain no
  tree/canopy returns. Physical obstacle cells are inflated by the configured
  robot footprint before trajectory acceptance; they are not folded into the
  ground elevation surface.
- Local-plane RMSE and MAD residuals do not determine validity: non-planarity
  is terrain, not missing data. MAD may refine an estimate but must not reject a
  cell. Cells without enough direct points are harmonically completed from the
  neighbouring elevation surface, then normals are recomputed from that
  completed surface.
- Upward normals must be unit length. The network and planner surfaces must be
  finite over the complete crop. Retain `observed_mask` separately to record
  which cells had a direct source-cloud fit; it is diagnostic provenance, not
  a training-validity mask.
- The source, licence, CRS, crop center, and processing parameters must be retained.
- The current `UnevenMap` loader hard-clips PCD input to z in [-0.01, 5.0] m.
  Output z is therefore translated by the patch minimum. Maps with more than
  5 m true vertical span require a loader-contract change; they must not be
  vertically scaled to fit.
- Current vehicle-aware limits use `min_cnormal=0.8` (36.87 degrees), a 0.26 m
  wheelbase, and the 0.2 m training grid. If those parameters change, this
  policy must be recalibrated and versioned.

## 2. Stage A: hard quality gate

A map is rejected if any condition below is met:

| Metric | Rejection threshold | Purpose |
|---|---:|---|
| Finite completed grid fraction | < 1.00 | Do not deliver holes or NaNs to the network/planner |
| Runtime z range | outside [-0.01, 5.0] m | Prevent silent clipping by UnevenMap |
| ULS source-surface density | < 40 points/m2 | Support 0.35 m local fitting |
| ALS source-surface density | < 4 points/m2 | Support 0.9 m local fitting |
| ALS source support at 1 m | < 0.97 | Reject holes despite adequate mean density |
| Planner-surface density | < 40 points/m2 | Support UnevenMap fitting |
| Normal length error P99 | > 0.01 | Enforce the map representation contract |
| Adjacent-cell height jump | > 0.50 m | Detect isolated reconstruction spikes |
| Largest connected slope-valid area | < 0.70 | Preserve a meaningful planning region |
| Area steeper than 36.87 degrees | > 0.25 | Reject overwhelmingly non-traversable maps |

Automated pattern checks for scan stripes and repeated structures should be
added when a larger corpus is available. Until calibrated, suspect maps are
quarantined rather than accepted by visual preference.

## 3. Stage B: geometric difficulty

Difficulty is measured at three scales on the completed finite surface.
`observed_mask` remains available to audit where completion was used.

- Micro scale (0.4 m): vibration-sized roughness near the vehicle wheelbase.
- Local scale (1.2 m): mounds, ruts, ridge transitions, and local curvature.
- Map scale (20 m): standard deviation after removing the best-fit global
  plane, so a single smooth incline is not mistaken for complex terrain.
- Slope structure: P95 slope and the fraction over 20 degrees.

The ULS v2 geometric score is:

```
30 * clip(detrended_std / 0.45)
+ 25 * clip(local_abs_P90 / 0.12)
+ 15 * clip(micro_abs_P90 / 0.04)
+ 20 * clip(fraction_slope_over_20 / 0.20)
+ 10 * clip(slope_P95 / 30)
```

Each `clip` is limited to [0, 1]. Preliminary grades are:

- easy: score < 25
- medium: 25 <= score < 55
- hard: 55 <= score < 80
- extreme: score >= 80

These thresholds are version-1 screening values, not ground truth. They must
be calibrated against planner outcomes in Stage C.

ALS cannot resolve the 0.4 m micro term reliably. Its observable-scale score
omits that term and uses weights 35/30/25/10 for detrended map-scale relief,
1.2 m local structure, slope-over-20 fraction, and P95 slope respectively.
The same preliminary grade boundaries are retained while Stage C remains a
separate acceptance label. The score profile is written to every result.

## 4. Stage C: planner-based acceptance

For each quality-passing base map, use a deterministic benchmark set before
dataset generation:

- Sample start/goal pairs from slope-valid connected terrain, at least 8 m apart.
- Use the same fixed pair list and yaw protocol for every candidate map.
- Record planning success, stable-trajectory success, normalized path length,
  maximum/mean slope along the path, planning time, and failure reason.
- Reject maps with no meaningful connected tasks or systematic failures caused
  by preprocessing artifacts.
- Recalibrate easy/medium/hard from empirical success and detour statistics;
  geometry grade remains a diagnostic field.

A canonical map that produces no accepted trajectory for a required benchmark
profile within the configured retry budget is quarantined as planner-rejected;
it must not be reloaded indefinitely as if a different crop existed. Confirm
the planner itself with a known-good, size/density-valid control map before
attributing this failure to terrain geometry.

Recommended release mixture per terrain category is 30% easy, 50% medium, and
20% hard. Extreme maps are held out as stress tests and do not replace hard
training maps.

## 5. Independence and leakage rules

- Split by survey/site before cropping. Crops from one survey may not cross
  train and validation.
- Candidate centres are sampled uniformly from occupied source-area cells,
  with within-cell jitter and random yaw. Sampling directly from source points
  is density-weighted and is retained only as an explicit diagnostic mode.
- Adjacent crops are not independent scenes. Retained crop centers from one
  survey should normally be at least 20 m apart; overlapping crops require an
  explicit augmentation label and count as one base-map family.
- Train/validation random transformations do not create new terrain identity.
- Validation transformations are fixed and deterministic.
- Dataset manifests record a source/site/family identifier and policy version.

## 6. Tooling

Run the current Stage A/B checker with:

```
python3 terrain_map_quality.py '/path/to/maps/*.npz'
```

The checker returns a hard quality decision, diagnostic metrics, and a
preliminary geometry grade. It intentionally reports that planner calibration
is still required.
