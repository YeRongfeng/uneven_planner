# 外部地图数据集生成使用指南

## 概述

本系统支持两种地图模式：
1. **自动生成地形**：使用原始的 `terrain_dataset_generation.launch`
2. **外部地图**：使用新的 `terrain_dataset_generation_external_map.launch`

## 外部地图配置

### 1. 基本使用

```bash
# 使用PCD格式的外部地图
roslaunch plan_manager terrain_dataset_generation_external_map.launch \
    external_map_path:=/path/to/your/map.pcd

# 使用PLY格式的外部地图
roslaunch plan_manager terrain_dataset_generation_external_map.launch \
    external_map_path:=/path/to/your/map.ply \
    external_map_format:=ply

# 使用TXT格式的外部地图
roslaunch plan_manager terrain_dataset_generation_external_map.launch \
    external_map_path:=/path/to/your/map.txt \
    external_map_format:=txt

# 使用高度图格式的外部地图
roslaunch plan_manager terrain_dataset_generation_external_map.launch \
    external_map_path:=/path/to/your/heightmap.npy \
    external_map_format:=heightmap
```

### 2. 完整参数配置示例

```bash
roslaunch plan_manager terrain_dataset_generation_external_map.launch \
    external_map_path:=/home/user/maps/terrain_42m.pcd \
    external_map_format:=pcd \
    target_map_size:=40.0 \
    target_resolution:=0.4 \
    num_environments:=50 \
    train_paths_per_env:=30 \
    val_paths_per_env:=3 \
    dataset_dir:=/home/user/datasets/my_terrain_dataset
```

## 关键参数说明

### 外部地图参数
- `external_map_path`: 外部地图文件的完整路径
- `external_map_format`: 文件格式 (pcd, ply, txt, heightmap)
- `target_map_size`: 目标地图尺寸（米），默认40m用于处理42m原始地图
- `target_resolution`: 目标分辨率（米），默认0.4m生成100x100栅格

### 数据集参数
- `num_environments`: 生成的环境数量
- `train_paths_per_env`: 每个环境的训练路径数
- `val_paths_per_env`: 每个环境的验证路径数
- `dataset_dir`: 数据集输出目录

### 路径生成参数
- `min_distance`: 起点终点最小距离（米），建议5.0m用于40m地图
- `publish_delay`: 路径生成间隔（秒）

## 地图处理流程

### 42m → 40m 地图处理
1. **加载**: 读取42m×42m的外部地图文件
2. **栅格化**: 使用0.4m分辨率转换为栅格（约105×105）
3. **裁剪**: 从中心裁剪出100×100栅格（对应40m×40m）
4. **保存**: 保存为标准格式供路径规划使用

### 支持的文件格式

#### PCD/PLY格式
- 标准点云格式
- 支持x, y, z坐标
- 自动处理点云密度

#### TXT格式
预期格式：每行包含x, y, z坐标
```
x1 y1 z1
x2 y2 z2
...
```

#### 高度图格式
- `.npy`: NumPy数组文件
- 图像格式: `.png`, `.jpg`等（灰度值映射为高度）

## 输出结构

```
dataset_dir/
├── train/
│   ├── env000000/
│   │   ├── map.p              # 地图数据
│   │   ├── terrain_2d.png     # 2D可视化
│   │   ├── terrain_3d.png     # 3D可视化  
│   │   ├── path_000000.p      # 路径数据
│   │   ├── path_000001.p
│   │   └── ...
│   └── ...
└── val/
    ├── env000000/
    │   ├── map.p              # 从train复制
    │   ├── terrain_2d.png     # 从train复制
    │   ├── terrain_3d.png     # 从train复制
    │   ├── path_000000.p      # 验证路径
    │   └── ...
    └── ...
```

## 故障排除

### 常见问题

1. **地图文件不存在**
   ```
   ERROR: External map file not found: /path/to/map.pcd
   ```
   解决：检查文件路径是否正确，文件是否存在

2. **格式不支持**
   ```
   ERROR: Unsupported external map format: xyz
   ```
   解决：使用支持的格式 (pcd, ply, txt, heightmap)

3. **内存不足**
   ```
   WARNING: Large point cloud detected, sampling...
   ```
   解决：系统会自动采样，或者预处理减少点云密度

4. **地图尺寸警告**
   ```
   WARNING: Grid size mismatch! Cropping from (105, 105) to (100, 100)
   ```
   解决：这是正常的，系统会自动裁剪到目标尺寸

### 调试模式

启用RViz可视化：
```bash
roslaunch plan_manager terrain_dataset_generation_external_map.launch \
    external_map_path:=/path/to/your/map.pcd \
    enable_rviz:=true
```

### 日志分析

关键日志信息：
- `Generated grid shape`: 确认栅格尺寸
- `Cropping region`: 显示裁剪区域
- `External map: saved with target bounds`: 确认最终结果

## 性能优化

1. **大型点云**: 系统会自动采样超过100万点的点云
2. **批量处理**: 可以并行运行多个实例处理不同的地图
3. **内存管理**: 使用分块处理避免内存溢出

## 与原始launch文件的兼容性

原始的 `terrain_dataset_generation.launch` 现在支持外部地图参数：

```bash
# 使用原始launch文件启用外部地图
roslaunch plan_manager terrain_dataset_generation.launch \
    use_external_map:=true \
    external_map_path:=/path/to/map.pcd \
    target_map_size:=40.0 \
    target_resolution:=0.4
```

推荐使用专用的 `terrain_dataset_generation_external_map.launch` 以获得更好的配置体验。
