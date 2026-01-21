#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集生成脚本
用于生成不平坦地形路径规划数据集

工作流程：
1. 随机生成起始点和目标点位姿
2. 发布起始点位姿给路径规划器（车辆传送）
3. 发布目标点位姿给路径规划器
4. 接收优化后的轨迹
5. 对轨迹进行均匀采样
6. 保存路径数据为pickle文件
7. 重复步骤1-6，直到生成指定数量的路径
"""

import rospy
import numpy as np
import pickle
import os
import random
import time
from geometry_msgs.msg import PoseStamped
from mpc_controller.msg import SE2Traj
from std_msgs.msg import Bool
import tf.transformations as tf_trans

# 地图转换相关库
import open3d as o3d
from scipy import interpolate
from scipy.spatial import cKDTree
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy available, will use GPU acceleration for grid processing")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available, using CPU-only processing")

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

# 可视化相关库
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors


class GridTransformer:
    """
    地图转换器：将点云转换为包含高程和法向量的栅格地图
    基于 grid_transformer.cpp 的 Python 实现
    """

    def __init__(self, coarse_resolution=0.4, fine_resolution=0.2, voxel_size=0.2):
        """
        初始化地图转换器

        Args:
            coarse_resolution: 粗糙栅格分辨率 (m)
            fine_resolution: 精细栅格分辨率 (m)
            voxel_size: 体素降采样大小 (m)
        """
        self.coarse_resolution = coarse_resolution
        self.fine_resolution = fine_resolution
        self.voxel_size = voxel_size
        self.search_radius = voxel_size * 2.5  # 法向量计算搜索半径

        # 栅格地图数据
        self.elevation_map = None
        self.normal_x_map = None
        self.normal_y_map = None
        self.normal_z_map = None

        # 地图元信息
        self.map_bounds = None  # (min_x, max_x, min_y, max_y)
        self.map_center = None
        self.map_size = None    # (width, height) in cells

    def pointcloud_from_ros(self, ros_pointcloud):
        """
        从 ROS PointCloud2 消息转换为 Open3D 点云

        Args:
            ros_pointcloud: sensor_msgs/PointCloud2

        Returns:
            open3d.geometry.PointCloud
        """
        # 从 ROS 消息中提取点云数据
        points_list = []
        for point in pc2.read_points(ros_pointcloud, skip_nans=True):
            points_list.append([point[0], point[1], point[2]])

        if not points_list:
            rospy.logwarn("Empty point cloud received")
            return None

        # 转换为 Open3D 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points_list))

        return pcd

    def preprocess_pointcloud(self, pcd):
        """
        点云预处理：降采样和法向量计算

        Args:
            pcd: open3d.geometry.PointCloud

        Returns:
            tuple: (downsampled_pcd, normals_array)
        """
        rospy.loginfo("Starting point cloud preprocessing...")

        # 体素降采样
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        rospy.loginfo(f"Downsampling: {len(pcd.points)} -> {len(downsampled_pcd.points)} points")

        # 计算法向量（与C++版本保持一致：不进行方向调整）
        downsampled_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamRadius(radius=self.search_radius)
        )

        # 注意：与grid_transformer.cpp保持一致，不进行法向量方向调整
        # C++版本直接使用NormalEstimationOMP计算的原始法向量

        # 提取法向量数组
        normals_array = np.asarray(downsampled_pcd.normals)
        points_array = np.asarray(downsampled_pcd.points)

        rospy.loginfo("Point cloud preprocessing completed")
        return points_array, normals_array

    def calculate_map_bounds(self, points):
        """
        计算地图边界

        Args:
            points: numpy array of shape (N, 3)

        Returns:
            tuple: (min_x, max_x, min_y, max_y)
        """
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)

        # 添加一些边界缓冲
        buffer = max(self.coarse_resolution, self.fine_resolution) * 2

        bounds = (
            min_bound[0] - buffer,  # min_x
            max_bound[0] + buffer,  # max_x
            min_bound[1] - buffer,  # min_y
            max_bound[1] + buffer   # max_y
        )

        return bounds

    def allocate_points_to_grid(self, points, normals, bounds, resolution):
        """
        将点云分配到栅格单元中

        Args:
            points: numpy array of shape (N, 3)
            normals: numpy array of shape (N, 3)
            bounds: tuple (min_x, max_x, min_y, max_y)
            resolution: grid resolution in meters

        Returns:
            dict: grid_cell -> list of point indices
        """
        min_x, max_x, min_y, max_y = bounds

        # 计算栅格尺寸
        width = int(np.ceil((max_x - min_x) / resolution))
        height = int(np.ceil((max_y - min_y) / resolution))

        # 初始化栅格字典
        grid_dict = {}

        # 分配点到栅格
        for i, point in enumerate(points):
            x, y = point[0], point[1]

            # 计算栅格索引
            grid_x = int((x - min_x) / resolution)
            grid_y = int((y - min_y) / resolution)

            # 边界检查
            grid_x = max(0, min(grid_x, width - 1))
            grid_y = max(0, min(grid_y, height - 1))

            # 添加到对应栅格
            grid_key = (grid_x, grid_y)
            if grid_key not in grid_dict:
                grid_dict[grid_key] = []
            grid_dict[grid_key].append(i)

        return grid_dict, (width, height)

    def process_grid_cells(self, points, normals, grid_dict, map_size):
        """
        处理栅格单元，计算高程和法向量
        基于 C++ 版本的核心算法：选择法向量Z分量最小的点

        Args:
            points: numpy array of shape (N, 3)
            normals: numpy array of shape (N, 3)
            grid_dict: dict mapping grid cells to point indices
            map_size: tuple (width, height)

        Returns:
            tuple: (elevation_grid, normal_x_grid, normal_y_grid, normal_z_grid)
        """
        width, height = map_size

        # 初始化栅格数组
        if CUPY_AVAILABLE:
            elevation_grid = cp.full((height, width), cp.nan, dtype=cp.float32)
            normal_x_grid = cp.full((height, width), cp.nan, dtype=cp.float32)
            normal_y_grid = cp.full((height, width), cp.nan, dtype=cp.float32)
            normal_z_grid = cp.full((height, width), cp.nan, dtype=cp.float32)
        else:
            elevation_grid = np.full((height, width), np.nan, dtype=np.float32)
            normal_x_grid = np.full((height, width), np.nan, dtype=np.float32)
            normal_y_grid = np.full((height, width), np.nan, dtype=np.float32)
            normal_z_grid = np.full((height, width), np.nan, dtype=np.float32)

        rospy.loginfo(f"Processing {len(grid_dict)} grid cells...")

        # 处理每个栅格单元
        for (grid_x, grid_y), point_indices in grid_dict.items():
            if len(point_indices) < 1:
                continue

            # 获取该栅格内的所有点和法向量
            cell_points = points[point_indices]
            cell_normals = normals[point_indices]

            # 核心算法：找到法向量Z分量最小的点（最接近水平面）
            # 计算归一化后的法向量Z分量的绝对值
            norm_lengths = np.linalg.norm(cell_normals, axis=1)
            normalized_z = np.abs(cell_normals[:, 2] / norm_lengths)

            # 选择Z分量最小的点的索引
            min_z_local_idx = np.argmin(normalized_z)
            min_z_global_idx = point_indices[min_z_local_idx]

            # 获取选中点的坐标和法向量（与C++版本一致）
            selected_point = points[min_z_global_idx]
            selected_normal = normals[min_z_global_idx]
            norm_length = np.linalg.norm(selected_normal)
            normalized_normal = selected_normal / norm_length

            # 存储到栅格中（使用选中点的高程，与C++版本一致）
            elevation_grid[grid_y, grid_x] = selected_point[2]  # 使用选中点的Z坐标
            normal_x_grid[grid_y, grid_x] = normalized_normal[0]
            normal_y_grid[grid_y, grid_x] = normalized_normal[1]
            normal_z_grid[grid_y, grid_x] = normalized_normal[2]

        # 如果使用GPU，转换回CPU
        if CUPY_AVAILABLE:
            elevation_grid = cp.asnumpy(elevation_grid)
            normal_x_grid = cp.asnumpy(normal_x_grid)
            normal_y_grid = cp.asnumpy(normal_y_grid)
            normal_z_grid = cp.asnumpy(normal_z_grid)

        rospy.loginfo("Grid cell processing completed")
        return elevation_grid, normal_x_grid, normal_y_grid, normal_z_grid

    def interpolate_grid(self, coarse_grids, bounds, target_resolution):
        """
        将粗糙栅格插值到精细分辨率

        Args:
            coarse_grids: tuple of (elevation, normal_x, normal_y, normal_z) grids
            bounds: tuple (min_x, max_x, min_y, max_y)
            target_resolution: target grid resolution

        Returns:
            tuple: interpolated grids
        """
        elevation_coarse, normal_x_coarse, normal_y_coarse, normal_z_coarse = coarse_grids
        min_x, max_x, min_y, max_y = bounds

        # 计算目标栅格尺寸
        target_width = int(np.ceil((max_x - min_x) / target_resolution))
        target_height = int(np.ceil((max_y - min_y) / target_resolution))

        # 创建坐标网格
        coarse_height, coarse_width = elevation_coarse.shape

        # 原始坐标
        x_coarse = np.linspace(min_x, max_x, coarse_width)
        y_coarse = np.linspace(min_y, max_y, coarse_height)

        # 目标坐标
        x_fine = np.linspace(min_x, max_x, target_width)
        y_fine = np.linspace(min_y, max_y, target_height)

        # 创建插值函数并进行插值
        rospy.loginfo("Performing grid interpolation...")

        # 高程使用三次插值
        valid_mask = ~np.isnan(elevation_coarse)
        if np.sum(valid_mask) > 0:
            points = []
            values = []
            for i in range(coarse_height):
                for j in range(coarse_width):
                    if valid_mask[i, j]:
                        points.append([x_coarse[j], y_coarse[i]])
                        values.append(elevation_coarse[i, j])

            if len(points) > 3:  # 需要至少4个点进行插值
                from scipy.interpolate import griddata
                xi, yi = np.meshgrid(x_fine, y_fine)
                elevation_fine = griddata(points, values, (xi, yi), method='cubic', fill_value=np.nan)
            else:
                elevation_fine = np.full((target_height, target_width), np.nan)
        else:
            elevation_fine = np.full((target_height, target_width), np.nan)

        # 法向量使用最近邻插值
        normal_x_fine = self._interpolate_nearest(normal_x_coarse, x_coarse, y_coarse, x_fine, y_fine)
        normal_y_fine = self._interpolate_nearest(normal_y_coarse, x_coarse, y_coarse, x_fine, y_fine)
        normal_z_fine = self._interpolate_nearest(normal_z_coarse, x_coarse, y_coarse, x_fine, y_fine)

        rospy.loginfo("Grid interpolation completed")
        return elevation_fine, normal_x_fine, normal_y_fine, normal_z_fine

    def _interpolate_nearest(self, data, x_coarse, y_coarse, x_fine, y_fine):
        """最近邻插值辅助函数"""
        coarse_height, coarse_width = data.shape
        target_height, target_width = len(y_fine), len(x_fine)

        result = np.full((target_height, target_width), np.nan)

        for i in range(target_height):
            for j in range(target_width):
                # 找到最近的粗糙栅格点
                x_idx = np.argmin(np.abs(x_coarse - x_fine[j]))
                y_idx = np.argmin(np.abs(y_coarse - y_fine[i]))

                if not np.isnan(data[y_idx, x_idx]):
                    result[i, j] = data[y_idx, x_idx]

        return result

    def transform_pointcloud_to_grid(self, ros_pointcloud):
        """
        主要转换函数：将ROS点云转换为栅格地图

        Args:
            ros_pointcloud: sensor_msgs/PointCloud2

        Returns:
            dict: 包含 'elevation', 'normal_x', 'normal_y', 'normal_z', 'bounds', 'resolution' 的字典
        """
        rospy.loginfo("Starting point cloud to grid transformation...")

        # 1. 转换点云格式
        pcd = self.pointcloud_from_ros(ros_pointcloud)
        if pcd is None:
            return None

        # 2. 预处理点云
        points, normals = self.preprocess_pointcloud(pcd)

        # 3. 计算地图边界
        bounds = self.calculate_map_bounds(points)
        self.map_bounds = bounds
        self.map_center = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2)

        # 4. 生成粗糙栅格
        grid_dict, map_size = self.allocate_points_to_grid(points, normals, bounds, self.coarse_resolution)
        coarse_grids = self.process_grid_cells(points, normals, grid_dict, map_size)

        # 5. 插值到精细栅格
        fine_grids = self.interpolate_grid(coarse_grids, bounds, self.fine_resolution)

        # 6. 存储结果
        self.elevation_map, self.normal_x_map, self.normal_y_map, self.normal_z_map = fine_grids
        self.map_size = (fine_grids[0].shape[1], fine_grids[0].shape[0])  # (width, height)

        result = {
            'elevation': self.elevation_map,
            'normal_x': self.normal_x_map,
            'normal_y': self.normal_y_map,
            'normal_z': self.normal_z_map,
            'bounds': bounds,
            'resolution': self.fine_resolution,
            'center': self.map_center,
            'size': self.map_size
        }

        rospy.loginfo("Point cloud to grid transformation completed!")
        return result


class DataGenerate:
    def __init__(self):
        """初始化数据生成器"""
        rospy.init_node('data_generate_node', anonymous=False)

        # 读取参数
        self.start_index = rospy.get_param('~start_index', 0)
        self.path_num = rospy.get_param('~path_num', 100)
        self.export_dir = rospy.get_param('~export_dir', 'data')
        self.map_name = rospy.get_param('~map_name', 'default_map')
        self.publish_delay = rospy.get_param('~publish_delay', 0.05)  # 极快重试：50ms

        # 地图边界参数（用于随机生成位姿）
        self.map_x_min = rospy.get_param('~map_x_min', -5.0)
        self.map_x_max = rospy.get_param('~map_x_max', 5.0)
        self.map_y_min = rospy.get_param('~map_y_min', -5.0)
        self.map_y_max = rospy.get_param('~map_y_max', 5.0)
        self.min_distance = rospy.get_param('~min_distance', 2.0)  # 起点和终点之间的最小距离

        # 地图转换参数
        self.enable_map_transform = rospy.get_param('~enable_map_transform', True)
        self.enable_map_visualization = rospy.get_param('~enable_map_visualization', True)
        self.coarse_resolution = rospy.get_param('~coarse_resolution', 0.4)
        self.fine_resolution = rospy.get_param('~fine_resolution', 0.2)
        self.voxel_size = rospy.get_param('~voxel_size', 0.2)

        # 从launch文件中读取PCD文件路径（使用现有的参数）
        self.pcd_file_path = rospy.get_param('/uneven_map/map_pcd', '')

        # 状态变量
        self.current_path_id = self.start_index
        self.waiting_for_result = False  # 等待规划结果
        self.current_trajectory = None

        # 创建输出目录
        self.output_dir = os.path.join(self.export_dir, self.map_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            rospy.loginfo(f"Created output directory: {self.output_dir}")

        # 地图数据存储
        self.grid_map_tensor = None  # 四通道张量 [H, W, 4]
        self.map_initialized = False

        # 一次性处理地图（在创建输出目录之后）
        if self.enable_map_transform and self.pcd_file_path:
            self.process_map_once()
        else:
            rospy.loginfo("Map transformation disabled or no PCD file specified")

        # ROS通信
        self.start_pose_pub = rospy.Publisher('/data_generate_node/start_pose', PoseStamped, queue_size=1)
        self.target_pose_pub = rospy.Publisher('/data_generate_node/target_pose', PoseStamped, queue_size=1)
        self.traj_sub = rospy.Subscriber('/data_generate_node/optimized_traj', SE2Traj, self.trajectory_callback)
        self.result_sub = rospy.Subscriber('/data_generate_node/planning_result', Bool, self.planning_result_callback)
        
        rospy.loginfo(f"DataGenerate initialized:")
        rospy.loginfo(f"  - start_index: {self.start_index}")
        rospy.loginfo(f"  - path_num: {self.path_num}")
        rospy.loginfo(f"  - export_dir: {self.export_dir}")
        rospy.loginfo(f"  - map_name: {self.map_name}")
        rospy.loginfo(f"  - output_dir: {self.output_dir}")
        rospy.loginfo(f"  - enable_map_transform: {self.enable_map_transform}")
        rospy.loginfo(f"  - enable_map_visualization: {self.enable_map_visualization}")
        rospy.loginfo(f"  - pcd_file_path: {self.pcd_file_path}")

    def process_map_once(self):
        """
        一次性处理地图：加载PCD文件，转换为四通道张量并保存
        """
        rospy.loginfo(f"Starting one-time map processing for map: {self.map_name}")

        if not os.path.exists(self.pcd_file_path):
            rospy.logerr(f"PCD file not found: {self.pcd_file_path}")
            return

        try:
            # 初始化地图转换器
            grid_transformer = GridTransformer(
                coarse_resolution=self.coarse_resolution,
                fine_resolution=self.fine_resolution,
                voxel_size=self.voxel_size
            )

            # 加载PCD文件
            rospy.loginfo(f"Loading PCD file: {self.pcd_file_path}")
            pcd = o3d.io.read_point_cloud(self.pcd_file_path)

            if len(pcd.points) == 0:
                rospy.logerr("Empty point cloud loaded!")
                return

            rospy.loginfo(f"Loaded {len(pcd.points)} points from PCD file")

            # 预处理点云
            points, normals = grid_transformer.preprocess_pointcloud(pcd)

            # 计算地图边界
            bounds = grid_transformer.calculate_map_bounds(points)

            # 生成粗糙栅格
            grid_dict, map_size = grid_transformer.allocate_points_to_grid(
                points, normals, bounds, self.coarse_resolution
            )
            coarse_grids = grid_transformer.process_grid_cells(points, normals, grid_dict, map_size)

            # 插值到精细栅格
            fine_grids = grid_transformer.interpolate_grid(coarse_grids, bounds, self.fine_resolution)

            # 裁剪地图到指定范围
            cropped_grids, cropped_bounds = self.crop_grid_map(fine_grids, bounds, self.fine_resolution)

            # 转换为四通道张量 [H, W, 4]
            elevation, normal_x, normal_y, normal_z = cropped_grids
            height, width = elevation.shape

            # 创建四通道张量
            self.grid_map_tensor = np.stack([elevation, normal_x, normal_y, normal_z], axis=2)
            rospy.loginfo(f"Created grid map tensor with shape: {self.grid_map_tensor.shape}")
            rospy.loginfo(f"Cropped map bounds: x=[{cropped_bounds[0]:.2f}, {cropped_bounds[1]:.2f}], y=[{cropped_bounds[2]:.2f}, {cropped_bounds[3]:.2f}]")

            # 保存地图张量
            self.save_map_tensor(cropped_bounds, self.fine_resolution)

            # 生成可视化图片（如果启用）
            if self.enable_map_visualization:
                self.visualize_grid_map(cropped_grids, cropped_bounds)
            else:
                rospy.loginfo("Map visualization disabled")

            # 标记地图已初始化
            self.map_initialized = True
            rospy.loginfo("Map processing completed successfully!")

            # 清理地图转换器对象（释放内存）
            del grid_transformer
            rospy.loginfo("Grid transformer object destroyed to free memory")

        except Exception as e:
            rospy.logerr(f"Error in map processing: {str(e)}")

    def save_map_tensor(self, bounds, resolution):
        """
        保存地图张量为 map.p 文件
        """
        try:
            map_data = {
                'tensor': self.grid_map_tensor,  # [H, W, 4] - elevation, normal_x, normal_y, normal_z
                'bounds': bounds,                # (min_x, max_x, min_y, max_y)
                'resolution': resolution,        # 栅格分辨率
                'map_name': self.map_name,       # 地图名称
                'channels': ['elevation', 'normal_x', 'normal_y', 'normal_z'],
                'shape': self.grid_map_tensor.shape
            }

            map_file = os.path.join(self.output_dir, "map.p")
            with open(map_file, 'wb') as f:
                pickle.dump(map_data, f)

            rospy.loginfo(f"Saved map tensor to: {map_file}")
            rospy.loginfo(f"Map tensor shape: {self.grid_map_tensor.shape}")
            rospy.loginfo(f"Map bounds: x=[{bounds[0]:.2f}, {bounds[1]:.2f}], y=[{bounds[2]:.2f}, {bounds[3]:.2f}]")
            rospy.loginfo(f"Map resolution: {resolution:.2f}m")

        except Exception as e:
            rospy.logerr(f"Failed to save map tensor: {str(e)}")

    def crop_grid_map(self, fine_grids, original_bounds, resolution):
        """
        根据地图边界参数裁剪栅格地图，确保精确的边界和尺寸

        Args:
            fine_grids: tuple of (elevation, normal_x, normal_y, normal_z) grids
            original_bounds: tuple (min_x, max_x, min_y, max_y) 原始边界
            resolution: 栅格分辨率

        Returns:
            tuple: (cropped_grids, cropped_bounds)
        """
        elevation, normal_x, normal_y, normal_z = fine_grids
        orig_min_x, orig_max_x, orig_min_y, orig_max_y = original_bounds

        # 目标裁剪边界（来自地图边界参数）
        crop_min_x = max(self.map_x_min, orig_min_x)
        crop_max_x = min(self.map_x_max, orig_max_x)
        crop_min_y = max(self.map_y_min, orig_min_y)
        crop_max_y = min(self.map_y_max, orig_max_y)

        # 检查裁剪边界是否有效
        if crop_min_x >= crop_max_x or crop_min_y >= crop_max_y:
            rospy.logwarn("Invalid crop boundaries, using original map")
            return fine_grids, original_bounds

        rospy.loginfo(f"Cropping map from [{orig_min_x:.2f}, {orig_max_x:.2f}, {orig_min_y:.2f}, {orig_max_y:.2f}] "
                     f"to [{crop_min_x:.2f}, {crop_max_x:.2f}, {crop_min_y:.2f}, {crop_max_y:.2f}]")

        # 计算目标栅格尺寸（基于精确的分辨率）
        target_width = int((crop_max_x - crop_min_x) / resolution)
        target_height = int((crop_max_y - crop_min_y) / resolution)

        rospy.loginfo(f"Target grid size: {target_width} x {target_height}")

        # 重新生成精确的栅格
        # 创建新的坐标网格，确保边界精确
        x_coords = np.linspace(crop_min_x + resolution/2, crop_max_x - resolution/2, target_width)
        y_coords = np.linspace(crop_min_y + resolution/2, crop_max_y - resolution/2, target_height)

        # 创建原始栅格的坐标
        orig_height, orig_width = elevation.shape
        orig_x_coords = np.linspace(orig_min_x + resolution/2, orig_max_x - resolution/2, orig_width)
        orig_y_coords = np.linspace(orig_min_y + resolution/2, orig_max_y - resolution/2, orig_height)

        # 使用插值重新采样到目标网格
        from scipy.interpolate import RegularGridInterpolator

        # 创建插值器
        points = (orig_y_coords, orig_x_coords)  # 注意：y在前，x在后
        xi, yi = np.meshgrid(x_coords, y_coords, indexing='xy')
        query_points = np.stack([yi.ravel(), xi.ravel()], axis=1)

        # 插值各个通道
        cropped_elevation = self._interpolate_channel(elevation, points, query_points, target_height, target_width)
        cropped_normal_x = self._interpolate_channel(normal_x, points, query_points, target_height, target_width)
        cropped_normal_y = self._interpolate_channel(normal_y, points, query_points, target_height, target_width)
        cropped_normal_z = self._interpolate_channel(normal_z, points, query_points, target_height, target_width)

        # 精确的边界
        actual_min_x = crop_min_x
        actual_max_x = crop_max_x
        actual_min_y = crop_min_y
        actual_max_y = crop_max_y

        cropped_grids = (cropped_elevation, cropped_normal_x, cropped_normal_y, cropped_normal_z)
        cropped_bounds = (actual_min_x, actual_max_x, actual_min_y, actual_max_y)

        rospy.loginfo(f"Cropped grid shape: {cropped_elevation.shape}")

        return cropped_grids, cropped_bounds

    def _interpolate_channel(self, data, points, query_points, target_height, target_width):
        """
        插值单个数据通道

        Args:
            data: 原始数据数组
            points: 原始网格坐标点
            query_points: 查询点坐标
            target_height: 目标高度
            target_width: 目标宽度

        Returns:
            插值后的数据数组
        """
        from scipy.interpolate import RegularGridInterpolator

        # 创建插值器，使用最近邻插值避免NaN传播
        interpolator = RegularGridInterpolator(
            points, data,
            method='nearest',  # 使用最近邻插值
            bounds_error=False,
            fill_value=np.nan
        )

        # 执行插值
        interpolated = interpolator(query_points)

        # 重塑为目标形状
        return interpolated.reshape(target_height, target_width)

    def visualize_grid_map(self, fine_grids, bounds):
        """
        可视化栅格地图：生成3D栅格图和2D热力图

        Args:
            fine_grids: tuple of (elevation, normal_x, normal_y, normal_z) grids
            bounds: tuple (min_x, max_x, min_y, max_y)
        """
        elevation, normal_x, normal_y, normal_z = fine_grids

        rospy.loginfo("Starting grid map visualization...")

        # 设置matplotlib为非交互模式
        plt.ioff()

        try:
            # 生成3D栅格图
            self._create_3d_grid_visualization(elevation, normal_x, normal_y, normal_z, bounds)

            # 生成2D热力图
            self._create_2d_heatmap_visualization(elevation, normal_x, normal_y, normal_z, bounds)

            rospy.loginfo("Grid map visualization completed!")

        except Exception as e:
            rospy.logerr(f"Error in grid map visualization: {str(e)}")

    def _create_3d_grid_visualization(self, elevation, normal_x, normal_y, normal_z, bounds):
        """
        创建3D栅格图可视化：平面根据法向量旋转
        """
        height, width = elevation.shape

        # 创建网格坐标
        x = np.linspace(bounds[0], bounds[1], width)
        y = np.linspace(bounds[2], bounds[3], height)

        # 栅格大小
        dx = (bounds[1] - bounds[0]) / width
        dy = (bounds[3] - bounds[2]) / height

        # 创建两个视角的图
        views = [
            {'elev': 30, 'azim': 45, 'name': 'front'},
            {'elev': 30, 'azim': 225, 'name': 'back'}
        ]

        for view in views:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # 收集所有有效的栅格面片
            patches = []
            colors_list = []

            # 减少密度以提高性能
            step = max(1, min(width, height) // 50)

            for i in range(0, height, step):
                for j in range(0, width, step):
                    if not np.isnan(elevation[i, j]):
                        # 栅格中心位置
                        cx = x[j]
                        cy = y[i]
                        cz = elevation[i, j]

                        # 法向量
                        nx = normal_x[i, j]
                        ny = normal_y[i, j]
                        nz = normal_z[i, j]

                        if not (np.isnan(nx) or np.isnan(ny) or np.isnan(nz)):
                            # 创建根据法向量倾斜的面片
                            patch_vertices = self._create_tilted_patch(
                                cx, cy, cz, nx, ny, nz, dx*step*0.8, dy*step*0.8
                            )
                            patches.append(patch_vertices)

                            # 根据高程设置颜色
                            colors_list.append(cz)

            if patches:
                # 创建3D面片集合
                poly_collection = Poly3DCollection(patches, alpha=0.8)

                # 设置颜色映射
                if colors_list:
                    norm = colors.Normalize(vmin=min(colors_list), vmax=max(colors_list))
                    poly_collection.set_facecolors(plt.cm.terrain(norm(colors_list)))

                ax.add_collection3d(poly_collection)

            # 设置坐标轴
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Elevation (m)')
            ax.set_title(f'3D Grid Map - {self.map_name} ({view["name"]} view)')

            # 设置视角
            ax.view_init(elev=view['elev'], azim=view['azim'])

            # 设置坐标轴范围
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            if colors_list:
                ax.set_zlim(min(colors_list), max(colors_list))

            # 设置等比例坐标轴（重要！）
            try:
                # 尝试使用新版本matplotlib的方法
                ax.set_box_aspect([
                    bounds[1] - bounds[0],  # X轴范围
                    bounds[3] - bounds[2],  # Y轴范围
                    max(colors_list) - min(colors_list) if colors_list else 1  # Z轴范围
                ])
            except AttributeError:
                # 兼容旧版本matplotlib
                x_range = bounds[1] - bounds[0]
                y_range = bounds[3] - bounds[2]
                z_range = max(colors_list) - min(colors_list) if colors_list else 1

                # 计算最大范围作为基准
                max_range = max(x_range, y_range, z_range)

                # 调整坐标轴范围使其等比例
                x_center = (bounds[0] + bounds[1]) / 2
                y_center = (bounds[2] + bounds[3]) / 2
                z_center = (min(colors_list) + max(colors_list)) / 2 if colors_list else 0

                ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
                ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
                ax.set_zlim(z_center - max_range/2, z_center + max_range/2)

            # 添加颜色条
            if colors_list:
                mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.terrain)
                mappable.set_array(colors_list)
                plt.colorbar(mappable, ax=ax, shrink=0.8, label='Elevation (m)')

            # 保存图片
            output_file = os.path.join(self.output_dir, f"map_3d_{view['name']}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            rospy.loginfo(f"Saved 3D visualization ({view['name']} view): {output_file}")

    def _create_tilted_patch(self, cx, cy, cz, nx, ny, nz, dx, dy):
        """
        创建根据法向量倾斜的矩形面片

        Args:
            cx, cy, cz: 面片中心坐标
            nx, ny, nz: 法向量分量
            dx, dy: 面片尺寸

        Returns:
            面片顶点坐标数组 [4, 3]
        """
        # 归一化法向量
        normal = np.array([nx, ny, nz])
        normal_length = np.linalg.norm(normal)
        if normal_length < 1e-6:  # 避免除零
            normal = np.array([0, 0, 1])  # 默认向上
        else:
            normal = normal / normal_length

        # 创建局部坐标系的矩形顶点（在XY平面上）
        local_vertices = np.array([
            [-dx/2, -dy/2, 0],
            [dx/2, -dy/2, 0],
            [dx/2, dy/2, 0],
            [-dx/2, dy/2, 0]
        ])

        # 计算旋转矩阵，将Z轴对齐到法向量方向
        z_axis = np.array([0, 0, 1])

        # 如果法向量已经接近Z轴，不需要旋转
        if abs(np.dot(normal, z_axis)) > 0.999:
            if np.dot(normal, z_axis) < 0:  # 如果法向量向下，翻转面片
                local_vertices[:, 2] *= -1
            rotated_vertices = local_vertices
        else:
            # 计算旋转轴（Z轴 × 法向量）
            rotation_axis = np.cross(z_axis, normal)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            # 计算旋转角度
            cos_angle = np.dot(z_axis, normal)
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            # 使用罗德里格旋转公式旋转每个顶点
            rotated_vertices = []
            for vertex in local_vertices:
                rotated_vertex = self._rodrigues_rotation(vertex, rotation_axis, angle)
                rotated_vertices.append(rotated_vertex)
            rotated_vertices = np.array(rotated_vertices)

        # 平移到实际位置
        world_vertices = rotated_vertices + np.array([cx, cy, cz])

        return world_vertices

    def _rodrigues_rotation(self, vector, axis, angle):
        """
        使用罗德里格旋转公式旋转向量

        Args:
            vector: 要旋转的向量
            axis: 旋转轴（单位向量）
            angle: 旋转角度（弧度）

        Returns:
            旋转后的向量
        """
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # 罗德里格旋转公式：v' = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
        rotated = (vector * cos_angle +
                  np.cross(axis, vector) * sin_angle +
                  axis * np.dot(axis, vector) * (1 - cos_angle))

        return rotated

    def _create_2d_heatmap_visualization(self, elevation, normal_x, normal_y, normal_z, bounds):
        """
        创建2D热力图可视化：高程热力图 + 法向量箭头
        """
        height, width = elevation.shape

        # 创建坐标网格
        x = np.linspace(bounds[0], bounds[1], width)
        y = np.linspace(bounds[2], bounds[3], height)

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制高程热力图
        elevation_masked = np.ma.masked_invalid(elevation)
        im = ax.imshow(elevation_masked, extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
                      origin='lower', cmap='terrain', aspect='equal')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Elevation (m)', fontsize=12)

        # 绘制法向量箭头（只显示部分，避免过于密集）
        step = max(1, min(width, height) // 20)  # 自适应步长

        for i in range(0, height, step):
            for j in range(0, width, step):
                if not (np.isnan(elevation[i, j]) or
                       np.isnan(normal_x[i, j]) or
                       np.isnan(normal_y[i, j])):

                    # 栅格中心坐标
                    cx = x[j]
                    cy = y[i]

                    # 法向量的XY分量
                    nx = normal_x[i, j]
                    ny = normal_y[i, j]

                    # 箭头长度（根据地图尺寸自适应）
                    arrow_scale = min(bounds[1] - bounds[0], bounds[3] - bounds[2]) / 30

                    # 绘制箭头
                    ax.arrow(cx, cy, nx * arrow_scale, ny * arrow_scale,
                            head_width=arrow_scale*0.3, head_length=arrow_scale*0.2,
                            fc='red', ec='red', alpha=0.7, linewidth=1)

        # 设置标题和标签
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'2D Grid Map - {self.map_name}\n(Elevation Heatmap + Normal Vectors)', fontsize=14)

        # 设置网格
        ax.grid(True, alpha=0.3)

        # 保存图片
        output_file = os.path.join(self.output_dir, "map_2d_heatmap.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        rospy.loginfo(f"Saved 2D heatmap visualization: {output_file}")

    def generate_random_pose(self):
        """生成完全随机的位姿"""
        x = random.uniform(self.map_x_min, self.map_x_max)
        y = random.uniform(self.map_y_min, self.map_y_max)
        yaw = random.uniform(-np.pi, np.pi)
        return x, y, yaw
    
    def create_pose_stamped(self, x, y, yaw):
        """创建PoseStamped消息"""
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        
        # 将偏航角转换为四元数
        quaternion = tf_trans.quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        
        return pose
    
    def generate_pose_pair(self):
        """生成起始点和目标点位姿对"""
        max_attempts = 50  # 最大尝试次数

        for attempt in range(max_attempts):
            # 生成起始点
            start_x, start_y, start_yaw = self.generate_random_pose()

            # 生成目标点，确保与起始点有足够距离
            for _ in range(20):  # 为目标点尝试20次
                target_x, target_y, target_yaw = self.generate_random_pose()
                distance = np.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
                if distance >= self.min_distance:
                    rospy.loginfo(f"Generated pose pair (attempt {attempt + 1}):")
                    rospy.loginfo(f"  Start: [{start_x:.3f}, {start_y:.3f}, {start_yaw:.3f}]")
                    rospy.loginfo(f"  Target: [{target_x:.3f}, {target_y:.3f}, {target_yaw:.3f}]")
                    rospy.loginfo(f"  Distance: {distance:.3f}m")
                    return (start_x, start_y, start_yaw), (target_x, target_y, target_yaw)

        # 如果所有尝试都失败，使用默认的安全位姿对
        rospy.logwarn("Failed to generate valid pose pair, using default safe poses")
        return (0.0, 0.0, 0.0), (1.0, 1.0, 0.0)
    
    def planning_result_callback(self, msg):
        """规划结果回调函数"""
        rospy.loginfo(f"Received planning result: {msg.data}, waiting_for_result: {self.waiting_for_result}")
        if not self.waiting_for_result:
            rospy.logwarn("Received planning result but not waiting for one. Ignoring.")
            return

        if msg.data:  # true表示成功
            rospy.loginfo(f"Path planning succeeded for path_{self.current_path_id}")
            # 成功时继续等待轨迹消息，不重置状态
        else:  # false表示失败
            rospy.logwarn(f"Path planning failed for path_{self.current_path_id}, retrying with new poses")
            self.waiting_for_result = False
            # 重新生成位姿对，不增加计数
            rospy.Timer(rospy.Duration(self.publish_delay), self.generate_next_path, oneshot=True)

    def trajectory_callback(self, msg):
        """轨迹回调函数"""
        if not self.waiting_for_result:
            rospy.logwarn("Received trajectory but not waiting for one. Ignoring.")
            return

        rospy.loginfo(f"Received trajectory for path_{self.current_path_id}")
        self.current_trajectory = msg
        self.waiting_for_result = False

        # 处理轨迹数据
        self.process_trajectory()
    
    def sample_trajectory(self, trajectory):
        """直接提取已采样的轨迹点（不进行重新采样）"""
        trajectory_points = []

        # 处理位置轨迹
        pos_pts = trajectory.pos_pts
        angle_pts = trajectory.angle_pts

        if len(pos_pts) == 0 or len(angle_pts) == 0:
            rospy.logwarn("Empty trajectory received")
            return None

        # 确保位置点和角度点数量一致
        min_length = min(len(pos_pts), len(angle_pts))
        if min_length == 0:
            rospy.logwarn("No valid trajectory points")
            return None

        rospy.loginfo(f"Extracting {min_length} trajectory points (pos_pts: {len(pos_pts)}, angle_pts: {len(angle_pts)})")

        # 直接提取轨迹点，不进行重新采样
        for i in range(min_length):
            x = pos_pts[i].x
            y = pos_pts[i].y
            yaw = angle_pts[i].x  # 假设角度存储在x字段中

            trajectory_points.append([x, y, yaw])

        return np.array(trajectory_points)
    
    def process_trajectory(self):
        """处理接收到的轨迹"""
        # 提取已采样的轨迹点
        trajectory_path = self.sample_trajectory(self.current_trajectory)

        if trajectory_path is None:
            rospy.logwarn(f"Failed to extract trajectory for path_{self.current_path_id}")
            return

        # 创建路径数据字典（简化版本，只包含轨迹）
        path_data = {
            'path': trajectory_path,
            'map_name': self.map_name  # 关联地图名称
        }

        # 保存为pickle文件
        output_file = os.path.join(self.output_dir, f"path_{self.current_path_id}.p")
        with open(output_file, 'wb') as f:
            pickle.dump(path_data, f)

        rospy.loginfo(f"Saved path_{self.current_path_id}.p with {len(trajectory_path)} points")
        
        # 移动到下一个路径
        self.current_path_id += 1
        
        # 检查是否完成
        if self.current_path_id >= self.start_index + self.path_num:
            rospy.loginfo(f"Data generation completed! Generated {self.path_num} paths.")
            rospy.signal_shutdown("Data generation completed")
            return
        
        # 延迟后生成下一个路径
        rospy.Timer(rospy.Duration(self.publish_delay), self.generate_next_path, oneshot=True)
    
    def generate_next_path(self, event):
        """生成下一个路径"""
        # 生成随机位姿对
        (start_x, start_y, start_yaw), (target_x, target_y, target_yaw) = self.generate_pose_pair()
        
        # 创建位姿消息
        start_pose = self.create_pose_stamped(start_x, start_y, start_yaw)
        target_pose = self.create_pose_stamped(target_x, target_y, target_yaw)
        
        # 在发布位姿之前设置等待状态，避免竞态条件
        self.waiting_for_result = True
        rospy.loginfo(f"Set waiting_for_result = True for path_{self.current_path_id}")

        # 发布起始点位姿
        self.start_pose_pub.publish(start_pose)
        rospy.loginfo(f"Published start pose for path_{self.current_path_id}: [{start_x:.3f}, {start_y:.3f}, {start_yaw:.3f}]")

        # 稍微延迟后发布目标点位姿
        rospy.sleep(0.1)
        self.target_pose_pub.publish(target_pose)
        rospy.loginfo(f"Published target pose for path_{self.current_path_id}: [{target_x:.3f}, {target_y:.3f}, {target_yaw:.3f}]")
    
    def start_generation(self):
        """开始数据生成"""
        rospy.loginfo(f"Starting data generation from path_{self.start_index} to path_{self.start_index + self.path_num - 1}")

        # 等待一段时间确保所有节点都已启动
        rospy.sleep(3.0)  # 恢复足够的初始等待时间，确保轨迹规划节点完全初始化

        # 检查地图状态
        if self.enable_map_transform:
            if self.map_initialized:
                rospy.loginfo("Map processing completed, starting path generation...")
            else:
                rospy.logwarn("Map processing failed, but continuing with path generation...")

        # 生成第一个路径
        self.generate_next_path(None)
    
    def run(self):
        """运行数据生成器"""
        self.start_generation()
        rospy.spin()


if __name__ == '__main__':
    try:
        data_generator = DataGenerate()
        data_generator.run()
    except rospy.ROSInterruptException:
        pass
