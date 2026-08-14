# Public Mother-Map Pilot — 2026-08-13

## Decision

`desert`, `forest`, `hill`, `snow`, and `volcano` are retained as source-domain
metadata, not model targets. The generated `map.p` contains elevation and
surface normals only. Dataset balancing therefore uses source site,
acquisition profile, and measured easy/medium/hard geometry. Train and
validation must use different survey sites.

## Reconstruction profiles and fixed parameters

- Output: 20 m x 20 m, 0.2 m grid, 0.05 m planner surface.
- ALS profile: class-2 source surface, 0.9 m fit radius, at least 5 neighbours,
  at least 4 source points/m2, and at least 97% occupied 1 m support cells.
- ULS profile: class-2 source surface, 0.35 m fit radius, at least 8 neighbours,
  at least 40 source points/m2, and at least 97% occupied 0.2 m support cells.
- Common fit RMSE ceiling: 0.12 m per local fit and 0.08 m at P95.
- Sampling seeds: desert 20260813, forest 20260821, hill 20260822, volcano
  20260823, snow 20260824.
- Maximum attempts: 20 for the cross-domain pilot; retained crop centres are
  at least 20 m apart.

The White Sands source is entirely LAS class 1. It was reclassified with PDAL
2.8.4 SMRF after resetting classes and applying a statistical outlier filter:
cell 1.0 m, scalar 1.25, slope 0.15, threshold 0.5 m, window 18 m, last/only
returns. The resulting file contains 5,062,642 class-2 points out of 5,068,152.
The raw class-1 result is diagnostic only and is not release-eligible.

## Official sources and pilot outcome

| Domain | Survey / licence | Pilot result | Status |
|---|---|---:|---|
| desert | White Sands NM10, CC BY 4.0 | 3/3 in 3 attempts: 2 medium, 1 easy | candidate |
| forest | WA21 Lumbrazo, CC BY 4.0 | 3/3 in 15 attempts: 2 medium, 1 easy | candidate, low yield |
| hill | CO10 Tucker (Raleigh Peak), CC BY 4.0 | 3/3 in 18 attempts: 3 medium | candidate, low yield |
| volcano | CA16 Martens (Inyo Domes), CC BY 4.0 | 3/3 in 5 attempts: hard/medium/easy | candidate |
| snow | BCZO SnowOn09, CC BY 4.0 | 3/3 in 14 attempts: medium/easy/hard | candidate, low yield |

Rejected alternatives were MT05 McGlynn (0/3 in 20 attempts because class-2
support was too sparse) and CA19 DiBiase (0/3 in 20 standard ALS attempts and
0/3 in 30 high-density-ALS attempts because the selected bedrock tile commonly
exceeded the 5 m runtime range and failed fit/validity gates). WY12 Befus was
quarantined after producing only 1/3 scenes in 20 attempts.

The pilot outputs and complete per-attempt rejection records are under
`dataset/tests/white_sands_als_smrf_20260813` and
`dataset/tests/*smoke_20260813`.

## Historical gate after the first pilot

At this point candidate status was only Stage A/B. Release still required an
independent validation site per domain and the fixed Stage-C planner benchmark.
Those requirements are completed in the sections below.

## Stage-C pilot

All 15 retained scenes completed the planner pilot with two saved trajectories
per scene (30/30 paths). Retry outcomes were:

| Domain | Saved / attempts | Planning failures | Stability rejections | Median saved-path margin |
|---|---:|---:|---:|---:|
| desert | 6/14 | 8 | 0 | 18.78 m |
| forest | 6/9 | 3 | 0 | 16.93 m |
| hill | 6/16 | 7 | 3 | 0.57 m |
| snow | 6/13 | 5 | 2 | 13.39 m |
| volcano | 6/13 | 6 | 1 | 2.06 m |

This validates the online pipeline and shows that hill and volcano geometry is
materially more challenging to the current planner. It does not establish
train/validation independence: every domain currently has only one accepted
site. Production generation remains blocked until a second survey site per
domain passes the same gates.

## Independent validation and scale-20 audit

Independent validation sites now pass the same Stage A/B gate: CA22 Marvin
(desert), AZ17 Donager (forest), CA13 Johnstone (hill), CA22 Piske (snow), and
CA19 Herbst (volcano). The five-site Stage-C test completed all 15 maps and all
30 requested trajectories. Saved-path attempt yields were 6/8 desert, 6/12
forest, 6/11 hill, 6/9 snow, and 6/13 volcano. Volcano had the lowest median
stability margin (0.36 m), consistent with its higher geometric difficulty
rather than a preprocessing failure.

Candidate centres are now sampled uniformly from occupied 1 m source-support
cells with within-cell jitter, instead of selecting a random source point.
This avoids over-sampling dense flight lines. A 20-map audit for every
train/domain and validation/domain combination produced 200/200 accepted maps.
Attempt counts ranged from 22 to 325 per 20 accepted maps; the default
production budget is therefore 30 attempts per requested canonical candidate.

The 20-map grade distributions are recorded in
`config/public_mother_map_release.json`. They are naturally imbalanced: for
example, validation forest is mostly easy while validation volcano is mostly
hard. Domain labels must not be presented as equivalent difficulty labels. If
a fixed 30/50/20 easy/medium/hard mixture is required, more source sites are
needed; re-scaling or weakening the quality gate is not permitted.

The generated `map.p` retains `valid_mask`. Training-time rotation must apply
the identical transform to the four map channels, trajectory coordinates, and
valid mask. Invalid cells are excluded from features/losses and must never be
treated as measured terrain. Validation transforms remain fixed and
deterministic.

With these conditions the source and pipeline are release candidates. The
release wrapper also passed a filesystem-complete forest test with one train
scene, one validation scene, and 2/1 trajectories. Every map was 100x100x4,
every trajectory contained 100 poses, valid masks exceeded 97%, and train/val
metadata resolved to WA21 Lumbrazo/AZ17 Donager respectively. The
default eight-worker configuration then passed an 8+8 scene stress test. All
eight worker manifests completed independently, covered environment IDs 0--7,
and the automatic artifact verifier found no map, path, mask, or provenance
errors. The 55,000-trajectory production run is ready to start.
