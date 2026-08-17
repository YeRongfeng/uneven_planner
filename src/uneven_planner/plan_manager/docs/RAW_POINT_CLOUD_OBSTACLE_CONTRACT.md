# Raw point-cloud terrain/obstacle contract

The public terrain sampler consumes every finite XYZ return in the mother
LAZ. LAS classification is recorded as provenance only and is not used to
build a scene.

Each accepted scene has three separate representations:

1. `scene_NNN.pcd` is a dense fitted ground surface. It is the only cloud sent
   to `UnevenMap`; tree returns must never enter this cloud.
2. `scene_NNN_obstacles.pcd` (kept in the sampling cache) contains raw returns
   whose height above the fitted ground is in the configured collision band.
3. The NPZ sidecar stores `obstacle_mask` and `obstacle_height` on the same
   100x100 grid as elevation and normals. `map.p` carries the same fields and
   uses them to merge physical obstacles into `occupancy_hwy`.

The current ALS pilot parameters are recorded in each scene's JSON:

- lower-envelope cell: `1.0 m`
- ground candidate band: `[-0.25, +0.35] m`
- low-envelope outlier threshold: `0.75 m`
- obstacle collision band: `[0.25, 2.0] m` above fitted ground
- obstacle cell minimum returns: `1`
- planner occupancy inflation: `0.2 m`

The forest release additionally requires at least 10 obstacle cells per
accepted scene. This is a content requirement for the forest domain, not a
change to the common terrain difficulty score.

The runtime YRF path must expose the same physical-obstacle layer to the neural
planner. A ground-only `GridMap` remains insufficient: trajectories can be
generated around trees offline while the deployed network cannot distinguish
the corresponding maps.
