# Recorded mother-map pilot test (2026-08-13)

## Reproducibility contract

- Map size/resolution: 20 m x 20 m, 0.2 m (`100 x 100`).
- Planner PCD resolution: 0.05 m (up to 160,000 points).
- Robust local-plane radius: 0.35 m.
- Minimum fit neighbours: 8; maximum fit RMSE: 0.12 m.
- Minimum source density: 40 points/m2.
- Minimum raw-cell and fitted valid coverage: 0.97.
- Minimum accepted-centre separation: 20 m.
- Source: PANGAEA 949228, CC-BY-SA-4.0, EPSG:25832.
- Sampling seeds: Pflege `2026081301`, Zugmantel `2026081302`,
  Pferdstrieb `2026081303`.
- Planner seeds: easy `2026081311`, medium `2026081312`, hard
  `2026081313`.
- Planner protocol: one canonical map, 2 train + 1 validation paths,
  12 consecutive failures before quarantine, distance mixture
  short/medium/long = 0.33/0.34/0.33, complexity mixture
  simple/moderate/high = 0.33/0.34/0.33.

The current sampling manifests retain source path, size and modification time,
dependency versions, complete arguments, accepted centres/yaws/metrics, and
every rejected attempt. Planner experiment manifests retain source-map records,
a full ROS parameter snapshot, dependency versions, every start/goal attempt
and outcome, retry statistics, and output counts. Earlier pilot artifacts may
still contain hashes from the original recording implementation.

## Sampling result

| Mother map | Seed | Accepted / attempted | Grades | Main rejection signals |
|---|---:|---:|---|---|
| PflegeSchoenau | 2026081301 | 3 / 12 | 3 easy | coverage, validity, z span |
| ZugmantelBandholz | 2026081302 | 3 / 9 | 1 easy, 1 medium, 1 hard | coverage, density, z span |
| PferdstriebSued | 2026081303 | 3 / 16 | 2 easy, 1 medium | coverage, density, validity, z span |

This is an acceptance-efficiency pilot, not a site-level success-rate estimate;
nine retained scenes are too few for distribution claims.

## Recorded planner result

| Grade | Geometry score | Occupied XY / SE(2) | Saved | Planning failures | Stability failures |
|---|---:|---:|---:|---:|---:|
| easy | 9.2 | 199 / 12,736 | 3 / 3 | 2 | 0 |
| medium | 50.7 | 186 / 6,766 | 3 / 3 | 2 | 1 |
| hard | 63.9 | 266 / 9,135 | 3 / 3 | 1 | 0 |

Every saved path has shape `(100, 3)`. The sample is deliberately too small to
rank planner difficulty by failure count. Occupancy is also not monotonic in
the geometry score, confirming that geometric grade and planner statistics
must remain separate labels.

## Artifacts

- Root: `dataset/tests/recorded_pilot_20260813/`
- Sampling manifests: `<site>/sampling_manifest.json`
- Planner manifests: `planner_<grade>/experiment_manifest.json`
- Maps and trajectories: `planner_<grade>/{train,val}/env000000/`

The manifest schema was extended after this run to record map-application
occupancy and successful-trajectory minimum stability margin directly. The
table above uses the original ROS run observations; future runs will carry
those values natively in `experiment_manifest.json`.

## Decision

The recording mechanism is suitable for a larger calibration run. The next
test should use at least 10 independent accepted scenes per grade and a fixed
benchmark task set per scene before changing grade thresholds or starting the
100-scene production datasets.
