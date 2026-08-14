# Canonical planner-rejection replacement test (2026-08-13)

## Purpose

Verify that a scene which passes geometric quality gating but fails Stage-C
planning is quarantined and replaced automatically, without terminating the
dataset run or mixing trajectories from two maps.

## Setup

- One train and one validation environment.
- One trajectory requested per split.
- Two canonical candidates per environment.
- Train primary: known extreme Zugmantel patch, geometry score 97.2.
- Train replacement: known easy PflegeSchoenau patch, score 23.2.
- Canonical rejection threshold reduced to one consecutive failed planning
  attempt so the replacement path is exercised directly.
- Planning seed: `2026081413`.

## Result

- The extreme primary loaded successfully with 42,163 occupied SE(2) states
  and 1,319 occupied XY cells.
- Its first trajectory attempt reached the ALM iteration limit.
- The generator recorded one canonical rejection, removed the partial train
  environment outputs, and selected pool index 1.
- The easy replacement loaded with zero occupied cells and saved the requested
  100-point train trajectory on its first attempt.
- Validation then used its independent primary map and saved one 100-point
  trajectory.
- Final status was `completed` with one planning failure, one canonical map
  rejected, one canonical map replaced, and no stability failure.
- The final train `map.p` points to the easy replacement, not the rejected
  extreme source.

Artifacts are under `dataset/tests/canonical_replacement_20260813/`.

## Decision

The Stage-C replacement loop works. Production sampling now prepares two
quality-passing candidates per requested environment by default. If both are
planner-rejected, the worker stops with an explicit exhausted-pool result;
raising the candidate count is a run parameter, not a change to acceptance
criteria.
