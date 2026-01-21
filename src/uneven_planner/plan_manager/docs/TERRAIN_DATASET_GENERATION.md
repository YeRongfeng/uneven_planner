# 地形数据集生成系统

这个系统可以自动生成基于随机地形的大规模路径规划训练数据集。

## 系统架构

```
随机地形生成器 -> 点云消息 -> 不平坦地形映射 -> 路径规划器 -> 数据集保存
     ↓              ↓           ↓            ↓           ↓
TerrainGenerator  PointCloud2  UnevenMap  OnlyPlanner  数据集文件
```

## 主要组件

### 1. TerrainGenerator (地形生成器)
- 基于 Perlin 噪声生成随机地形
- 支持山峰、山谷、山脊等多种地形特征
- 将高度图转换为带颜色的点云数据

### 2. GridTransformer (地图转换器)
- 将点云数据转换为包含高程和法向量的栅格地图
- 支持多分辨率插值
- 生成四通道张量 [高程, 法向量X, 法向量Y, 法向量Z]

### 3. TerrainDatasetGenerator (数据集生成器)
- 协调整个数据生成流程
- 管理多环境和多路径的生成
- 按照指定的目录结构保存数据

## 数据集结构

```
dataset/
├── train/
│   ├── env000001/
│   │   ├── map.p                    # 地图数据（四通道张量 + 元信息）
│   │   ├── terrain_2d.png          # 2D地形可视化
│   │   ├── terrain_3d.png          # 3D地形可视化
│   │   ├── path_0.p                # 路径数据
│   │   ├── path_1.p
│   │   └── ... (共50条训练路径)
│   ├── env000002/
│   └── ... (共1000个环境)
└── val/
    ├── env000001/
    │   ├── map.p
    │   ├── terrain_2d.png
    │   ├── terrain_3d.png
    │   ├── path_0.p
    │   └── ... (共5条验证路径)
    └── ... (共1000个环境)
```

## 数据格式

### map.p 文件内容
```python
{
    'grid_map': np.array,           # 四通道张量 [H, W, 4]
    'bounds': tuple,                # (min_x, max_x, min_y, max_y)
    'resolution': float,            # 栅格分辨率
    'center': tuple,                # 地图中心坐标
    'size': tuple,                  # 栅格尺寸 (width, height)
    'terrain_info': dict,           # 地形生成信息
    'heightmap': np.array          # 原始高度图
}
```

### path_X.p 文件内容
```python
{
    'path': np.array,              # 轨迹点 [N, 3] (x, y, yaw)
    'env_id': int,                 # 环境ID
    'path_id': int,                # 路径ID
    'phase': str                   # 'train' 或 'val'
}
```

## 使用方法

### 1. 小规模测试（推荐先运行）
```bash
# 生成3个环境，每个环境5条训练路径+2条验证路径
roslaunch plan_manager test_terrain_dataset.launch
```

### 2. 完整数据集生成
```bash
# 生成1000个环境，每个环境50条训练路径+5条验证路径
roslaunch plan_manager terrain_dataset_generation.launch
```

### 3. 自定义参数
```bash
roslaunch plan_manager terrain_dataset_generation.launch \
    num_environments:=500 \
    train_paths_per_env:=30 \
    val_paths_per_env:=3 \
    dataset_dir:=/path/to/your/dataset \
    enable_rviz:=true
```

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_environments | 1000 | 要生成的环境数量 |
| train_paths_per_env | 50 | 每个环境的训练路径数量 |
| val_paths_per_env | 5 | 每个环境的验证路径数量 |
| dataset_dir | datasets/terrain_dataset | 数据集输出目录 |
| map_size | 10.0 | 地图尺寸（米）|
| map_resolution | 0.02 | 地形分辨率（米）|
| max_height | 3.0 | 最大地形高度（米）|
| min_distance | 2.0 | 路径起终点最小距离（米）|

## 系统要求

### 依赖包
- ROS Melodic/Noetic
- Open3D
- matplotlib
- scipy
- noise (Perlin噪声库)
- CuPy (可选，用于GPU加速)

### 硬件建议
- 内存：至少8GB（大规模生成建议16GB+）
- 存储：每1000个环境约需要10-50GB存储空间
- GPU：可选，用于栅格处理加速

## 性能估算

以默认参数为例：
- 单个环境生成时间：~5-10秒
- 单条路径生成时间：~0.5-2秒
- 总计用时：约15-25小时（1000环境 × 55路径）
- 存储空间：约20-40GB

## 故障排除

### 常见问题
1. **点云消息未发布**：检查地形生成器是否正常工作
2. **路径规划失败**：调整 min_distance 参数或地图边界
3. **内存不足**：减少 num_environments 或使用更大分辨率
4. **RViz显示问题**：检查话题名称和坐标系设置

### 调试建议
1. 先运行小规模测试验证系统工作
2. 启用RViz观察地形和路径生成过程
3. 检查输出目录中的可视化图片
4. 监控ROS话题通信状态

## 扩展功能

### 添加新的地形类型
在 `TerrainGenerator.generate_terrain()` 中添加新的地形特征函数。

### 修改路径生成策略
在 `TerrainDatasetGenerator.generate_pose_pair()` 中自定义位姿生成逻辑。

### 自定义数据格式
修改 `process_trajectory()` 函数中的数据保存格式。

## 许可证

本项目遵循与主项目相同的许可证。
