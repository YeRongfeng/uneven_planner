# Live terrain dataset viewer

The primary viewer is a local tool, not a generated dataset artifact. Start it
against a dataset root or a single domain root:

```bash
python3 src/uneven_planner/plan_manager/scripts/serve_terrain_dataset_viewer.py \
  /path/to/dataset --port 8765
```

Then open `http://127.0.0.1:8765/`.

When the viewer is started from the 8767 mother-map review tool, its header also
contains `快速加载中间地图` and `快速加载最终结果`. These buttons use the
current 8767 `训练地图文件夹` and `最终数据文件夹`, so the intermediate
canonical maps and final trajectory scenes can be switched without browsing for
their directories again.

The dataset root may be empty or may not exist yet. The server discovers final
scenes under `train/val/env*/map.p` and canonical intermediate maps under
`train/val/map_*.npz` (with their JSON sidecars). The browser refreshes the scene
index every five seconds and reloads the selected scene only when its map or
trajectory set changes. Current trajectory selection, all-trajectory mode,
camera, and filters remain UI state rather than dataset files.

The live page is not limited to the directory passed at startup. Use `选择目录`
in the header to browse server-side folders and load another dataset or domain;
`重新读取` rescans the currently selected directory. A dataset can therefore be
inspected without restarting the viewer or manually changing a shell command.

The 8767 review fit is an all-class human-aid view: it reads all finite XYZ
returns, removes only returns clearly below a geometry-derived lower envelope,
and does not apply an upper height limit. Consequently, class1/class2 returns,
tree canopy, and other high returns can form visible protrusions in this fit.
The 8765 canonical view uses a different data layer: its `elevation` grid is
the continuous ground surface used by the planner, while retained raw returns
and obstacle information are carried separately. The two views are therefore
intentionally complementary, not numerically identical.

`make_terrain_dataset_viewer.py` remains available only when a standalone,
non-updating HTML snapshot is explicitly needed for sharing or archival.
