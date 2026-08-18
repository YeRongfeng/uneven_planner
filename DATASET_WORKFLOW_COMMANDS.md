# uneven_planner 网页数据制作速查

8767 是主要工具。正常制作数据时，只需要启动网页服务，然后在浏览器中完成地图导入、人工筛选和数据生成；不需要手动移动文件、编辑 JSONL 或运行内部生成脚本。

## Quick Start

第一次在这台机器上做数据，按下面三步走：编译工程、打开网页、在浏览器里做完一轮。之后每次只需启动 8767。

### 1. 环境和编译

需要 Ubuntu 20.04、ROS Noetic，以及本机上的 `vim` conda 环境。系统依赖和仿真插件见仓库根目录 [README.md](README.md)。

```bash
cd /home/sdu/uneven_planner
source /opt/ros/noetic/setup.bash
source /home/sdu/miniconda3/etc/profile.d/conda.sh
conda activate vim

catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

只看点云、框选和拟合时，8767 本身不依赖这次编译。点「生成最终训练数据」会启动 ROS 规划器，必须先 `catkin_make` 成功，并且工作空间里有 `devel/setup.bash`。改过 C++ 规划代码后要重新编译再生成。

### 2. 启动网页

在已经 `conda activate vim` 的项目目录里：

```bash
python3 src/uneven_planner/plan_manager/scripts/serve_mother_map_review.py \
  --port 8767 \
  --title "原始点云地图人工筛选"
```

本机浏览器打开 <http://127.0.0.1:8767/>。不要关这个终端。8765 结果查看器不要单独手开，用网页「生成」页里的按钮启动和关闭。

### 3. 在网页里做完一轮

1. 「地图」页导入 LAS/LAZ，或选择已有的 `dataset/external`。
2. 填写「本次数据名称」并点「确认」（例如 `forest`）。
3. 顶部先选地形目录，再选具体原始地图；中间画布框选约 20 m 区域。
4. 看二维、3D 和「拟合地形」，选训练集或验证集，点通过或拒绝。
5. 「生成」页先点「生成训练用地图」，完成后再点「生成最终训练数据」。
6. 同一页「启动结果查看器」→「打开结果查看器」（<http://127.0.0.1:8765/>）检查 `map.p` 和轨迹；看完点「关闭结果查看器」。
7. 8765 里某张图不行，点「打回，去8767补位」，回到 8767 补那个编号后再生成。

最终结果在 `dataset/public_terrain_20m/<任务名>/`。停止 8767：回到启动它的终端按 `Ctrl+C`。

## 启动 8767

在项目目录执行：

```bash
cd /home/sdu/uneven_planner
source /home/sdu/miniconda3/etc/profile.d/conda.sh
conda activate vim

python3 src/uneven_planner/plan_manager/scripts/serve_mother_map_review.py \
  --port 8767 \
  --title "原始点云地图人工筛选"
```

打开 <http://127.0.0.1:8767/>。

启动命令保持通用，不绑定某一种地形或任务。
8767 启动时不会自动读取 LAZ；地图和任务名称由网页操作主动选择。
继续以前的标注时，在“恢复以前的标注”中浏览并加载对应的 JSONL 文件。

## 在网页中操作

1. 在“选择原始地图”中选择“从电脑选择地图文件”，或点击“选择已有地图文件夹”。已有地图可以直接选择 `dataset/external`。
2. 设置“本次数据名称”，点击右侧“确认”；默认输出路径会随名称同步，手动改过的路径会保留。
3. 在顶部先选择“地形目录”，再选择“具体原始地图”。第二个下拉框会显示来源目录和文件名。
4. 在地图上框选候选区域，结合二维点云、3D 点云和全部点云拟合结果判断。
5. 选择训练集或验证集，点击“通过并保存”或“拒绝并保存”。也可以使用“通过/拒绝并随机下一块”。
6. 需要继续以前的工作时，勾选显示全部历史选区，或从历史列表点击回看。重复选择重叠目录不会重复显示地图。
7. 人工筛选完成后，依次点击“生成训练用地图”和“生成最终训练数据”。训练轨迹数和验证轨迹数可在“调整生成数量或保存位置”中分行设置。
8. 生成后在“生成”页点击“启动结果查看器”，再点击“打开结果查看器”；检查完点击“关闭结果查看器”。
9. 如果在 8765 里发现某张地图或环境不合适，点击“打回，去8767补位”。该编号会在 8767 的“补充空位”中出现；重新框选并通过后选择对应空位，再重新生成训练用地图。

网页会自动管理以下内容：

```text
dataset/external/       原始 LAS/LAZ 地图
dataset/external/SOURCES.md  地图来源链接
dataset/reviews/        人工筛选记录
dataset/canonical/      训练用中间地图
dataset/public_terrain_20m/  最终训练数据
```

最终训练数据位于 `dataset/public_terrain_20m/<任务名>/`，包括 `train/`、`val/`、`map.p` 和 `path_*.p`。

## 查看最终数据

生成前或生成后，都可以在 8767 的“生成”页点击“启动结果查看器”。即使最终数据文件夹还不存在，
服务也会启动并显示“还没有生成场景”；目录出现后会自动看到新结果。服务默认使用
<http://127.0.0.1:8765/>；点击“打开结果查看器”检查地图和轨迹，完成后点击“关闭结果查看器”。

## 停止服务

8767 仍在运行时，查看器用页面按钮启动和关闭。停止 8767 时回到运行服务的终端按 `Ctrl+C`；已有地图和人工记录不会丢失。

更完整的网页操作说明见：

- [8767 原始点云地图人工筛选](src/uneven_planner/plan_manager/docs/MOTHER_MAP_MANUAL_REVIEW.md)
- [人工标注地图并生成训练数据](src/uneven_planner/plan_manager/docs/MANUAL_MOTHER_MAP_DATASET_WORKFLOW.md)
