# 人工标注地图并生成训练数据

这套流程适用于 forest、desert、hill、snow 以及其他包含 LAS/LAZ 点云的地形。
8767 页面不依赖具体地形名称；下面的 `forest` 只是示例任务名。

## 操作者实际要做的事

1. 启动 8767，在网页中选择或导入下载的 `.las/.laz` 原始点云。
2. 在地图上框选候选区域，用二维、3D 和拟合结果辅助判断，手动点击通过或拒绝。
3. 点“生成训练用地图”。
4. 点“生成最终训练数据”。
5. 在“生成”页启动并打开结果查看器查看结果，检查 `map.p` 和 `path_*.p` 数量后关闭查看器。

正常情况下不需要手动创建目录、移动文件、填写服务器路径或编辑 JSONL。

## 启动 8767

```bash
cd /home/sdu/uneven_planner
source /home/sdu/miniconda3/etc/profile.d/conda.sh
conda activate vim

python3 src/uneven_planner/plan_manager/scripts/serve_mother_map_review.py \
  --port 8767 \
  --title "原始点云地图人工筛选"
```

打开 `http://127.0.0.1:8767/`。

换地形时在网页“地图”页修改“本次数据名称”并点击“确认”；地图筛选逻辑不变。

可以不在命令后面写地图目录，之后用“从电脑选择地图文件”导入；也可以把项目内已有地图目录
作为最后一个参数传入。`--domain` 只是这批数据的名称，换成其他任务时改它即可。

页面顶部的“当前原始地图”分成两级选择：

- “地形目录”选择 `forest`、`desert`、`hill`、`snow` 或 `volcano`。
- “具体原始地图”选择该地形下的来源目录和文件，例如
  `forest_wa21 / 642000_5265000.laz`。

因此，选择已有地图文件夹时可以直接选择 `dataset/external`；网页会按第一层地形目录分组，
不会把所有地图混成只显示文件名的一张列表。第二级选项的完整路径也会在浏览器悬停时显示。
同一目录重复选择，或先选择子目录再选择覆盖它的上级目录，网页都会按唯一文件保留一张地图。

## 原始地图放在哪里

最简单的做法是在网页中点“从电脑选择地图文件”，选择下载位置中的 LAS/LAZ 文件。
页面会自动复制到：

```text
dataset/external/<任务名>/<任务名>_imported/raw/
```

下载的原文件保留不动。若地图已经放在项目目录中，点“选择已有地图文件夹”；这个操作只读取，
不会复制。该按钮中的目录列表是运行 8767 的这台电脑上的项目目录，不是网络服务器；操作者不需要
填写目录文本。

## 标注记录是什么

每次点击通过或拒绝，页面会追加一行记录到：

```text
dataset/reviews/<任务名>/manual_regions.jsonl
```

它相当于人工筛选的工作记录，包含候选中心、大小、旋转、通过/拒绝、train/val 选择和备注。
它不是点云地图，也不是最终训练数据。网页会自动加载它，所以正常工作不需要选择审核文件。

“恢复以前的标注”用于换电脑、加载备份或继续另一轮筛选；“下载标注备份”用于交接。

## 训练用地图是什么

人工通过的区域还不能直接交给 ROS 生成器。第 3 步会从原始点云中按人工记录裁出完整的 20m 地图，
并写出程序需要的 `.pcd/.npz/.json` 文件和一份自动清单。

这批中间文件实际放在：

```text
dataset/canonical/<任务名>/<运行名>/
```

程序内部把它叫 canonical map。它只是“人工筛选到最终轨迹生成之间的中间地图”，不是新的数据类型；
网页默认自动管理，操作者通常不需要知道这个名字。

## 最终训练数据是什么

第 4 步使用第 3 步生成的清单，调用现有 ROS 生成器，默认输出到：

```text
dataset/public_terrain_20m/<任务名>/
```

最终训练数据包括：

```text
train/env*/map.p
train/env*/path_*.p
val/env*/map.p
val/env*/path_*.p
```

其中 `map.p` 是地图，`path_*.p` 是轨迹。拿去训练的是这里的最终文件，不是 JSONL，也不是
`dataset/canonical` 中间文件。

## 直接运行底层命令（仅用于维护或排错）

先设置与启动页面相同的任务变量：

```bash
TASK_NAME=forest
REVIEW_FILE="dataset/reviews/${TASK_NAME}/manual_regions.jsonl"
CANONICAL_DIR="dataset/canonical/${TASK_NAME}/reviewed_regions_20260816"
DATASET_DIR="dataset/public_terrain_20m/${TASK_NAME}"
```

网页第 3 步等价于：

```bash
python3 src/uneven_planner/plan_manager/scripts/build_approved_canonical_maps.py \
  "${REVIEW_FILE}" \
  "${CANONICAL_DIR}" \
  --domain "${TASK_NAME}" \
  --train-site-id "${TASK_NAME}_train" \
  --val-site-id "${TASK_NAME}_val" \
  --train-source-profile als \
  --val-source-profile als
```

网页第 4 步等价于：

```bash
bash src/uneven_planner/plan_manager/scripts/generate_approved_canonical_dataset.sh \
  "${CANONICAL_DIR}/approved_canonical_manifest.json" \
  "${DATASET_DIR}" \
  100 10 4 11412
```

这两个命令只给维护者使用。网页会代替操作者填写中间文件路径，并显示任务状态。

## 查看和检验

8765 查看最终数据：在 8767 的“生成”页点击“启动结果查看器”，再点击“打开结果查看器”；检查完点击“关闭结果查看器”。

完成后可以检查当前任务：

```bash
python3 src/uneven_planner/plan_manager/scripts/verify_public_20m_dataset.py \
  "${DATASET_DIR}" \
  --scenes 100 --train-paths 100 --val-paths 10 \
  --train-site "${TASK_NAME}_train" \
  --val-site "${TASK_NAME}_val"
```

只有在文件数量和轨迹点数检查通过后，才把这一轮称为完成。

## 关于 `tmp`

`/home/sdu/tmp` 可能存在以前运行留下的备份或未完成结果，但它不是网页默认工作目录。
当前 forest 的人工记录在 `dataset/reviews/forest/manual_regions.jsonl`；其他任务按同样规则使用仓库内的
`dataset/reviews/<任务名>/`、`dataset/external/<任务名>/<任务名>_imported/` 和最终输出目录。
