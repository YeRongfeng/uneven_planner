# Zugmantel terrain smoke test — 2026-08-13

## Candidate

`dune_sandhausen_zugmantel_candidate_001`

- Contract: 20 m x 20 m, 0.2 m, 100 x 100.
- Audited valid grid: 97.73%.
- Dense planner cloud: 86,097 source points; 85,940 points applied.
- Planner occupancy: 454 XY cells / 18,266 SE(2) voxels occupied.
- Stage A/B result: quality pass, geometric score 82.2 (`extreme`).

## Planner observations

- The map loaded atomically and repeatedly completed SE(2) to RXS2
  construction without NaNs or PCL crashes.
- Long/high-complexity profile: two complete 15-attempt budgets produced no
  accepted trajectory. A* usually found a path, but ALM reached its iteration
  limit; a few paths exceeded the 22 m limit or exhausted A* memory.
- Short/simple profile: two complete 10-attempt budgets produced no accepted
  trajectory. A* found 7–11 m paths, but ALM still reached its iteration limit.
- No candidate `path_*.p` was saved.

## Control

The existing `map_my.pcd` was the only local control checked that satisfied
both 20 m crop size and local density (41.0 m extent, about 2,973 points/m2).
Under the same short/simple protocol it saved both train and validation paths:

- Train: accepted on attempt 3, 100 samples, 8.41 m XY length, minimum
  continuous stability margin 1.0296 m.
- Validation: accepted on its first attempt, 100 samples, 7.95 m XY length,
  minimum continuous stability margin 0.4425 m.

The old `desert.pcd` is only 12 m x 12 m and cannot provide a real 20 m crop.
`map_mountain_40m.pcd` spans about 600 m but is too sparse for a valid 20 m
window, so neither is a valid control.

## Decision

The candidate is structurally valid but planner-rejected for normal dataset
generation with the current vehicle and optimizer. Keep it quarantined as an
`extreme` stress map. Do not use it in the requested 100-scene train/validation
release unless planner parameters or the crop are changed and the deterministic
Stage-C benchmark is rerun.
