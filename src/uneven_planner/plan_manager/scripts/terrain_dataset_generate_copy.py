#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
随机地形数据集生成器
基于随机地形生成和路径规划，创建大规模训练数据集

工作流程：
1. 随机生成地形（基于generator.py）
2. 将地形转换为点云并通过ROS发布
3. 等待地图初始化完成
4. 为每个地形生成多条路径
5. 保存地形数据和路径数据到指定的数据集结构中
"""

import rospy
from std_srvs.srv import Trigger

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，避免线程问题
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 添加3D绘图支持
import open3d as o3d
import noise
import random
import math
import time
import os
import pickle
import tf.transformations as tf_trans
from scipy.ndimage import gaussian_filter, zoom
from scipy.interpolate import CubicSpline

# ROS消息类型
from geometry_msgs.msg import PoseStamped
from mpc_controller.msg import SE2Traj
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, MapMetaData
import sensor_msgs.point_cloud2 as pc2

# 地图转换相关
from scipy import interpolate
from scipy.spatial import cKDTree
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy available, will use GPU acceleration for grid processing")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available, using CPU-only processing")
    
import torch

def compute_map_yaw_bins(normal_x, normal_y, normal_z, yaw_bins=18):
    """
    计算地图上每个点的分箱角度是否会倾覆（PyTorch版本，高效批量处理）
    
    基于相同的物理约束来判断每个yaw_bin角度是否会导致倾覆，
    不会倾覆的角度分箱标为1，会倾覆的标为0。
    
    Args:
        normal_x: 地形法向量x分量 (H, W) 
        normal_y: 地形法向量y分量 (H, W)
        normal_z: 地形法向量z分量 (H, W)
        yaw_bins: 朝向角度的分箱数量，默认为18

    Returns:
        torch.Tensor: 每个点每个角度分箱的倾覆状态，形状为(H, W, yaw_bins)，1表示不会倾覆，0表示会倾覆
    """
    # 确保输入是torch张量
    if not isinstance(normal_x, torch.Tensor):
        normal_x = torch.tensor(normal_x, dtype=torch.float32)
        normal_y = torch.tensor(normal_y, dtype=torch.float32) 
        normal_z = torch.tensor(normal_z, dtype=torch.float32)
    
    device = normal_x.device
    H, W = normal_x.shape
    
    # 地形约束参数
    # h = 8.0         # 机器人高度
    # min_edge = 10.0 # 最小边长约束
    # max_edge = 20.0 # 最大边长约束
    h = 35.0         # 机器人高度
    min_edge = 8.0 # 最小边长约束
    max_edge = 15.0 # 最大边长约束
    
    # 计算地形倾斜角度对应的tan值
    terrain_slope = torch.arctan2(torch.sqrt(normal_x**2 + normal_y**2), torch.abs(normal_z))
    
    # 计算约束参数b
    b_vals = h * torch.tan(terrain_slope)
    
    # 根据b值分类地形约束类型
    mask_reachable = b_vals < min_edge  # 完全可达
    mask_partial = (b_vals >= min_edge) & (b_vals < max_edge)  # 部分可达
    mask_complex = (b_vals >= max_edge) & (b_vals < torch.sqrt(torch.tensor(max_edge**2 + min_edge**2, device=device)))  # 复杂约束
    mask_unreachable = b_vals >= torch.sqrt(torch.tensor(max_edge**2 + min_edge**2, device=device))  # 完全不可达
    
    # 初始化结果数组：所有角度分箱都标记为不会倾覆(1)
    yaw_stability = torch.ones((H, W, yaw_bins), dtype=torch.float32, device=device)
    
    # 定义角度分箱的中心角度：从-π到π均匀分布
    bin_angles = torch.linspace(-torch.pi, torch.pi, yaw_bins + 1, device=device)[:-1]  # 去掉最后一个
    
    def safe_arctan_transform(nz_vals, theta_local_vals):
        """安全的arctan变换，考虑角度的正确象限"""
        tan_vals = torch.tan(theta_local_vals)
        arctan_result = torch.arctan(nz_vals * tan_vals)
        
        # 如果原始角度在第二或第三象限（cos < 0），需要调整arctan结果
        cos_local = torch.cos(theta_local_vals)
        sin_local = torch.sin(theta_local_vals)
        
        # 调整第二象限的角度：arctan结果需要加π
        second_quadrant = (cos_local < 0) & (sin_local > 0)
        arctan_result = torch.where(second_quadrant, arctan_result + torch.pi, arctan_result)
        
        # 调整第三象限的角度：arctan结果需要减π
        third_quadrant = (cos_local < 0) & (sin_local < 0)
        arctan_result = torch.where(third_quadrant, arctan_result - torch.pi, arctan_result)

        # arctan_result = normalize_angle(torch.pi * 0.5 - arctan_result)  # 作 theta=pi/2-arctan_result映射，并确保结果在[-π, π]范围内

        return arctan_result

    def normalize_angle(angle):
        """标准化角度到[-π, π]"""
        angle = torch.where(angle > torch.pi, angle - 2*torch.pi, angle)
        angle = torch.where(angle < -torch.pi, angle + 2*torch.pi, angle)
        return angle
    
    def check_angle_in_range_vectorized(angles, starts, ends):
        """向量化检查角度是否在范围内"""
        # angles: (yaw_bins,), starts: (N,), ends: (N,)
        # 返回: (N, yaw_bins) 布尔张量
        angles = angles.unsqueeze(0)  # (1, yaw_bins)
        starts = starts.unsqueeze(1)  # (N, 1)
        ends = ends.unsqueeze(1)      # (N, 1)
        
        angles = normalize_angle(angles)
        starts = normalize_angle(starts)
        ends = normalize_angle(ends)
        
        # 正常情况：start <= end
        normal_case = starts <= ends
        in_range_normal = (angles >= starts) & (angles <= ends) & normal_case
        
        # 跨越边界情况：start > end
        cross_boundary = starts > ends
        in_range_cross = ((angles >= starts) | (angles <= ends)) & cross_boundary
        
        return in_range_normal | in_range_cross
    
    # 处理完全不可达区域：所有角度都标记为会倾覆(0)
    yaw_stability[mask_unreachable] = 0.0
    
    # 处理部分可达区域 - 向量化处理
    if torch.any(mask_partial):
        # 获取部分可达区域的坐标和值
        partial_indices = torch.where(mask_partial)
        partial_b = b_vals[mask_partial]
        partial_nx = normal_x[mask_partial]
        partial_ny = normal_y[mask_partial]
        partial_nz = normal_z[mask_partial]
        
        # 批量计算约束边界角度（局部坐标系）
        s1_vals = torch.arcsin(min_edge / partial_b)
        e1_vals = torch.pi - s1_vals
        s2_vals = -s1_vals
        e2_vals = -torch.pi + s1_vals
        
        # 考虑地形法向量的影响：从局部坐标系转换到全局坐标系
        normal_proj_angles = torch.arctan2(partial_ny, partial_nx)
        
        # 计算全局坐标系下的边界参数
        s1_transforms = safe_arctan_transform(partial_nz, s1_vals)
        e1_transforms = safe_arctan_transform(partial_nz, e1_vals)
        s2_transforms = safe_arctan_transform(partial_nz, s2_vals)
        e2_transforms = safe_arctan_transform(partial_nz, e2_vals)
        
        s1_globals = normalize_angle(normal_proj_angles + s1_transforms)
        e1_globals = normalize_angle(normal_proj_angles + e1_transforms)
        s2_globals = normalize_angle(normal_proj_angles + s2_transforms)
        e2_globals = normalize_angle(normal_proj_angles + e2_transforms)
        
        # 向量化检查每个角度分箱是否在不可达区域
        in_unreachable_region1 = check_angle_in_range_vectorized(bin_angles, s1_globals, e1_globals)  # (N, yaw_bins)
        in_unreachable_region2 = check_angle_in_range_vectorized(bin_angles, e2_globals, s2_globals)  # (N, yaw_bins)
        
        unreachable_mask = in_unreachable_region1 | in_unreachable_region2  # (N, yaw_bins)
        
        # 更新yaw_stability
        for idx, (i, j) in enumerate(zip(partial_indices[0], partial_indices[1])):
            yaw_stability[i, j, unreachable_mask[idx]] = 0.0
    
    # 处理复杂约束区域 - 向量化处理
    if torch.any(mask_complex):
        # 获取复杂约束区域的坐标和值
        complex_indices = torch.where(mask_complex)
        complex_b = b_vals[complex_indices]
        complex_nx = normal_x[complex_indices]
        complex_ny = normal_y[complex_indices]
        complex_nz = normal_z[complex_indices]
        
        # 批量计算复杂约束的边界参数
        r1_vals = torch.arcsin(min_edge / complex_b)
        r2_vals = torch.arccos(max_edge / complex_b)
        
        # 计算所有边界角度（局部坐标系）
        s1_vals = -r2_vals
        e1_vals = r2_vals
        s2_vals = r1_vals
        e2_vals = torch.pi - r1_vals
        p1_vals = torch.pi - r2_vals
        p2_vals = -torch.pi + r2_vals
        s3_vals = -torch.pi + r1_vals
        e3_vals = -r1_vals
        
        # 考虑地形法向量的影响：从局部坐标系转换到全局坐标系
        normal_proj_angles = torch.arctan2(complex_ny, complex_nx)
        
        # 计算全局坐标系下的边界参数
        s1_transforms = safe_arctan_transform(complex_nz, s1_vals)
        e1_transforms = safe_arctan_transform(complex_nz, e1_vals)
        s2_transforms = safe_arctan_transform(complex_nz, s2_vals)
        e2_transforms = safe_arctan_transform(complex_nz, e2_vals)
        p1_transforms = safe_arctan_transform(complex_nz, p1_vals)
        p2_transforms = safe_arctan_transform(complex_nz, p2_vals)
        s3_transforms = safe_arctan_transform(complex_nz, s3_vals)
        e3_transforms = safe_arctan_transform(complex_nz, e3_vals)
        
        s1_globals = normalize_angle(normal_proj_angles + s1_transforms)
        e1_globals = normalize_angle(normal_proj_angles + e1_transforms)
        s2_globals = normalize_angle(normal_proj_angles + s2_transforms)
        e2_globals = normalize_angle(normal_proj_angles + e2_transforms)
        p1_globals = normalize_angle(normal_proj_angles + p1_transforms)
        p2_globals = normalize_angle(normal_proj_angles + p2_transforms)
        s3_globals = normalize_angle(normal_proj_angles + s3_transforms)
        e3_globals = normalize_angle(normal_proj_angles + e3_transforms)
        
        # 向量化检查每个角度分箱是否在不可达区域
        in_unreachable1 = check_angle_in_range_vectorized(bin_angles, s1_globals, e1_globals)
        in_unreachable2 = check_angle_in_range_vectorized(bin_angles, s2_globals, e2_globals)
        in_unreachable3 = check_angle_in_range_vectorized(bin_angles, s3_globals, e3_globals)
        
        # 处理p1和p2边界（单侧边界）
        bin_angles_expanded = bin_angles.unsqueeze(0)  # (1, yaw_bins)
        p1_expanded = p1_globals.unsqueeze(1)  # (N, 1)
        p2_expanded = p2_globals.unsqueeze(1)  # (N, 1)
        
        in_unreachable_p1 = bin_angles_expanded > p1_expanded
        in_unreachable_p2 = bin_angles_expanded < p2_expanded
        
        unreachable_mask = (in_unreachable1 | in_unreachable2 | in_unreachable3 | 
                           in_unreachable_p1 | in_unreachable_p2)  # (N, yaw_bins)
        
        # 更新yaw_stability
        for idx, (i, j) in enumerate(zip(complex_indices[0], complex_indices[1])):
            yaw_stability[i, j, unreachable_mask[idx]] = 0.0

    return yaw_stability



class TerrainGenerator:
    """
    地形生成器：基于generator.py的地形生成逻辑
    """
    
    def __init__(self, map_size=10.0, resolution=0.02, max_height=3.0, min_height=0.0):
        """
        初始化地形生成器
        
        Args:
            map_size: 地图尺寸（米）
            resolution: 网格分辨率（米）
            max_height: 最大高度（米）
            min_height: 最小高度（米）
        """
        self.MAP_SIZE = map_size
        self.RESOLUTION = resolution
        self.MAX_HEIGHT = max_height
        self.MIN_HEIGHT = min_height
        
        self.num_pts = int(self.MAP_SIZE / self.RESOLUTION)
        
        # 创建坐标网格
        x = np.linspace(-self.MAP_SIZE/2, self.MAP_SIZE/2, self.num_pts)
        y = np.linspace(-self.MAP_SIZE/2, self.MAP_SIZE/2, self.num_pts)
        self.xx, self.yy = np.meshgrid(x, y, indexing='xy')
    
    def perlin_noise_grid(self, size, cycles, octaves=4, seed=0):
        """生成多尺度Perlin噪声"""
        grid = np.zeros((size, size), dtype=np.float32)
        for i in range(size):
            for j in range(size):
                nx = i / (size - 1) * cycles
                ny = j / (size - 1) * cycles
                grid[i, j] = noise.pnoise2(nx + seed, ny + seed, octaves=octaves, 
                                          persistence=0.5, lacunarity=2.0,
                                          repeatx=1024, repeaty=1024, base=0)
        mn, mx = grid.min(), grid.max()
        if mx - mn > 1e-9:
            grid = 2 * (grid - mn) / (mx - mn) - 1
        return grid
    
    def upsample(self, field, target_shape):
        """上采样低分辨率场到目标形状"""
        return zoom(field, (target_shape[0] / field.shape[0], target_shape[1] / field.shape[1]), order=3)
    
    def add_elliptical_peak(self, hmap, cx, cy, a, b, angle_deg, height, exponent=1.3):
        """添加椭圆形峰 - 简化版本，避免8字形凹陷"""
        th = math.radians(angle_deg)
        xr = (self.xx - cx) * math.cos(th) + (self.yy - cy) * math.sin(th)
        yr = -(self.xx - cx) * math.sin(th) + (self.yy - cy) * math.cos(th)
        
        # 基本椭圆距离
        r2_base = (xr / a) ** 2 + (yr / b) ** 2
        
        # 简单的、保证正值的轻微变化
        angle_to_center = np.arctan2(yr, xr)
        
        # 使用绝对值确保变化始终为正，避免凹陷
        gentle_variation = 1.0 + 0.03 * np.abs(np.sin(2 * angle_to_center + random.uniform(0, 2*math.pi)))  # 小幅度正值变化
        
        # 应用轻微变化，但保持椭圆形状
        r2 = r2_base * gentle_variation
        
        # 添加非常轻微的Perlin噪声纹理，但限制在小范围内
        if random.random() < 0.7:  # 70%概率添加纹理
            noise_seed = random.random() * 100
            mask = r2_base < 4  # 只在峰的核心区域添加纹理
            if np.any(mask):
                nx = xr[mask] / (a * 1.5)  # 大尺度，低频噪声
                ny = yr[mask] / (b * 1.5)
                
                noise_values = np.zeros(len(nx))
                for idx in range(len(nx)):
                    # 非常轻微的纹理
                    noise_val = 0.01 * noise.pnoise2(nx[idx] + noise_seed, ny[idx] + noise_seed, octaves=1)
                    noise_values[idx] = max(0, noise_val)  # 确保非负
                
                noise_distortion = np.zeros_like(r2_base)
                noise_distortion[mask] = noise_values
                r2 += noise_distortion
        
        # 确保r2始终为正
        r2 = np.maximum(r2, 0.01)
        
        # 基本高斯影响函数
        influence = np.exp(-r2)
        
        # 简化的指数调制，避免复杂的角度变化
        stable_exponent = exponent * random.uniform(0.9, 1.1)  # 只有轻微的随机变化
        stable_exponent = np.clip(stable_exponent, 0.8, 1.5)  # 保守的范围
        influence = influence ** stable_exponent
        
        # 轻微的高度随机化，但避免角度依赖
        height_multiplier = random.uniform(0.95, 1.05)  # 全局高度的轻微变化
        final_height = height * height_multiplier
        
        hmap += final_height * influence
        return hmap
    
    def add_elliptical_valley(self, hmap, cx, cy, a, b, angle_deg, depth, exponent=1.0):
        """添加椭圆形谷 - 简化版本，避免不自然的形状"""
        th = math.radians(angle_deg)
        xr = (self.xx - cx) * math.cos(th) + (self.yy - cy) * math.sin(th)
        yr = -(self.xx - cx) * math.sin(th) + (self.yy - cy) * math.cos(th)
        
        # 基本椭圆距离
        r2_base = (xr / a) ** 2 + (yr / b) ** 2
        
        # 简单的正值轻微变化
        angle_to_center = np.arctan2(yr, xr)
        gentle_variation = 1.0 + 0.02 * np.abs(np.sin(2 * angle_to_center + random.uniform(0, 2*math.pi)))
        
        r2 = r2_base * gentle_variation
        
        # 轻微的纹理（可选）
        if random.random() < 0.5:  # 50%概率添加纹理
            noise_seed = random.random() * 100
            mask = r2_base < 6
            if np.any(mask):
                nx = xr[mask] / (a * 2.0)  # 大尺度
                ny = yr[mask] / (b * 2.0)
                
                noise_values = np.zeros(len(nx))
                for idx in range(len(nx)):
                    noise_val = 0.008 * noise.pnoise2(nx[idx] + noise_seed, ny[idx] + noise_seed, octaves=1)
                    noise_values[idx] = max(0, noise_val)  # 确保非负
                
                noise_distortion = np.zeros_like(r2_base)
                noise_distortion[mask] = noise_values
                r2 += noise_distortion
        
        r2 = np.maximum(r2, 0.01)
        influence = np.exp(-r2)
        
        # 简化的指数调制
        stable_exponent = exponent * random.uniform(0.95, 1.05)
        stable_exponent = np.clip(stable_exponent, 0.8, 1.3)
        influence = influence ** stable_exponent
        
        # 简单的深度随机化
        depth_multiplier = random.uniform(0.9, 1.1)
        final_depth = depth * depth_multiplier
        
        hmap += final_depth * influence
        return hmap
    
    def generate_smooth_ridge_path(self, length=10, num_points=40, max_curve=2.5):
        """生成平滑的山脊路径"""
        xs = np.linspace(0, length, num_points)
        ctrl_pts_x = np.linspace(0, length, 7)
        ctrl_pts_y = max_curve * np.sin(np.linspace(0, 2 * math.pi, 7) + random.uniform(0, 2*math.pi))
        ctrl_pts_y += np.random.uniform(-0.3, 0.3, size=7)
        
        cs = CubicSpline(ctrl_pts_x, ctrl_pts_y, bc_type='natural')
        ys = cs(xs)
        
        path = np.vstack((xs, ys)).T
        
        # 随机旋转
        theta = random.uniform(0, 2 * math.pi)
        rot = np.array([[math.cos(theta), -math.sin(theta)],
                        [math.sin(theta),  math.cos(theta)]])
        path_rot = path @ rot
        
        # 平移到地图中心附近
        min_xy = path_rot.min(axis=0)
        max_xy = path_rot.max(axis=0)
        shift = np.array([0, 0]) - (min_xy + max_xy)/2
        shift += np.random.uniform(-self.MAP_SIZE/4, self.MAP_SIZE/4, size=2)
        path_final = path_rot + shift
        
        path_final = np.clip(path_final, -self.MAP_SIZE/2, self.MAP_SIZE/2)
        return path_final
    
    def add_continuous_ridge(self, hmap, path, width, height):
        """沿路径添加连续山脊"""
        dist_along = np.zeros(len(path))
        for i in range(1, len(path)):
            dist_along[i] = dist_along[i-1] + np.linalg.norm(path[i]-path[i-1])
        total_len = dist_along[-1]
        norm_dist = dist_along / total_len
        
        long_profile = (np.sin(norm_dist * math.pi * 2 * random.uniform(1.5, 3)) * 0.3 + 
                        np.sin(norm_dist * math.pi * 4 * random.uniform(2,4)) * 0.15 + 1)
        long_profile /= long_profile.max()
        long_profile *= height
        
        points = np.stack([self.xx.ravel(), self.yy.ravel()], axis=1)
        ridge_heights = np.zeros(points.shape[0], dtype=np.float32)
        
        batch_size = 50000
        for start_idx in range(0, points.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, points.shape[0])
            batch_points = points[start_idx:end_idx]
            dists = np.linalg.norm(batch_points[:, None, :] - path[None, :, :], axis=2)
            min_idx = np.argmin(dists, axis=1)
            min_dist = dists[np.arange(len(batch_points)), min_idx]
            lat_w = np.exp(-(min_dist / width) ** 2)
            ridge_heights[start_idx:end_idx] = long_profile[min_idx] * lat_w

        ridge_layer = ridge_heights.reshape(self.xx.shape)
        hmap += ridge_layer
        return hmap
    
    def generate_terrain(self, rng_seed=None):
        """
        生成随机地形
        
        Args:
            rng_seed: 随机种子，如果为None则使用时间戳
            
        Returns:
            tuple: (heightmap, terrain_info)
        """
        if rng_seed is None:
            rng_seed = time.time_ns() % (2**32)
        
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        
        rospy.loginfo(f"Generating terrain with RNG seed: {rng_seed}")
        
        start_time = time.time()
        
        # 多尺度噪声基础起伏
        global_cycles = random.uniform(0.7, 1.2)
        meso_cycles = random.uniform(2.0, 3.5)
        micro_cycles = random.uniform(2.0, 4.0)
        low_res = 128
        
        global_noise = self.perlin_noise_grid(low_res, global_cycles, octaves=4, seed=random.random() * 10)
        meso_noise = self.perlin_noise_grid(low_res, meso_cycles, octaves=3, seed=random.random() * 10 + 20)
        micro_noise = self.perlin_noise_grid(low_res, micro_cycles, octaves=2, seed=random.random() * 10 + 40)
        
        global_up = self.upsample(global_noise, (self.num_pts, self.num_pts))
        meso_up = self.upsample(meso_noise, (self.num_pts, self.num_pts))
        micro_up = self.upsample(micro_noise, (self.num_pts, self.num_pts))
        
        base = 0.8 * global_up + 0.5 * meso_up + 0.15 * micro_up
        base = base * 0.6
        
        # 宏观波动
        wave_amp = random.uniform(0.8, 2.0)
        wave_freq = 2 * math.pi / self.MAP_SIZE
        offset_x = random.uniform(0, 2 * math.pi)
        offset_y = random.uniform(0, 2 * math.pi)
        sinusoid = wave_amp * (np.sin(wave_freq * self.xx + offset_x) + np.sin(wave_freq * self.yy + offset_y))
        base += sinusoid
        
        # 轻微倾斜
        tilt_angle = random.uniform(-math.pi / 6, math.pi / 6)
        tilt = (self.xx * math.cos(tilt_angle) + self.yy * math.sin(tilt_angle)) / self.MAP_SIZE
        base += 0.05 * tilt
        
        # 归一化
        base = (base - base.min()) / (base.max() - base.min())
        heightmap = base * self.MAX_HEIGHT * 0.3
        
        # 决定是否生成山脊
        generate_ridge = random.random() < 0.5
        terrain_features = {"ridge": generate_ridge}
        
        if generate_ridge:
            ridge_path = self.generate_smooth_ridge_path(
                length=random.uniform(8.0, 10.0),
                num_points=200,
                max_curve=random.uniform(1.0, 2.5)
            )
            ridge_width = random.uniform(0.7, 1.3)
            ridge_height = random.uniform(0.2, 0.3)
            heightmap = self.add_continuous_ridge(heightmap, ridge_path, ridge_width, ridge_height)
            
            terrain_features.update({
                "ridge_width": ridge_width,
                "ridge_height": ridge_height
            })
            
            heightmap = gaussian_filter(heightmap, sigma=1.5)  # 从1.2增加到1.5，加强山脊平滑
            peaks_num = random.randint(1, 2)
        else:
            peaks_num = random.randint(2, 4)
        
        # 主要地形特征
        placed_centers = []
        min_dist = 2.5
        terrain_features["peaks"] = []
        
        for _ in range(peaks_num):
            for _ in range(50):
                cx = random.uniform(-self.MAP_SIZE/2 + 0.5, self.MAP_SIZE/2 - 0.5)
                cy = random.uniform(-self.MAP_SIZE/2 + 0.5, self.MAP_SIZE/2 - 0.5)
                if all(math.hypot(cx - px, cy - py) > min_dist for (px, py) in placed_centers):
                    break
            
            placed_centers.append((cx, cy))
            height = random.uniform(0.3, 0.6)
            a = random.uniform(height*3, height*5)
            b = random.uniform(height*2, height*3)
            angle = random.uniform(0, 360)
            exponent = random.uniform(0.3, 0.5)
            
            heightmap = self.add_elliptical_peak(heightmap, cx, cy, a, b, angle, height, exponent)
            
            terrain_features["peaks"].append({
                "center": (cx, cy),
                "semi_axes": (a, b),
                "angle": angle,
                "height": height,
                "exponent": exponent
            })
        
        # 添加谷
        if random.random() < 0.5:
            for _ in range(50):
                cx = random.uniform(-self.MAP_SIZE/2 + 1.0, self.MAP_SIZE/2 - 1.0)
                cy = random.uniform(-self.MAP_SIZE/2 + 1.0, self.MAP_SIZE/2 - 1.0)
                if all(math.hypot(cx - px, cy - py) > min_dist for (px, py) in placed_centers):
                    break
            
            placed_centers.append((cx, cy))
            a = random.uniform(2.5, 4.5)
            b = random.uniform(2.0, 3.6)
            angle = random.uniform(0, 360)
            depth = -random.uniform(0.15, 0.3)
            
            heightmap = self.add_elliptical_valley(heightmap, cx, cy, a, b, angle, depth)
            
            terrain_features["valley"] = {
                "center": (cx, cy),
                "semi_axes": (a, b),
                "angle": angle,
                "depth": depth
            }
        
        # 增强平滑处理
        heightmap = gaussian_filter(heightmap, sigma=1.5)  # 从1.0增加到1.5
        
        # 添加多阶段平滑：先用较小的sigma处理高频噪声，再用较大的sigma整体平滑
        heightmap = gaussian_filter(heightmap, sigma=0.8)  # 先处理高频
        heightmap = gaussian_filter(heightmap, sigma=1.2)  # 再整体平滑
        
        micro_detail_cycles = 20
        micro_detail_noise = self.perlin_noise_grid(low_res, micro_detail_cycles, octaves=2, seed=random.random() * 100)
        micro_detail = self.upsample(micro_detail_noise, (self.num_pts, self.num_pts))
        heightmap += micro_detail * random.uniform(0.000, 0.005)
        
        # 最终归一化
        heightmap = np.clip(heightmap, self.MIN_HEIGHT, self.MAX_HEIGHT)
        heightmap = ((heightmap - heightmap.min()) / (heightmap.max() - heightmap.min()) * 
                    (self.MAX_HEIGHT - self.MIN_HEIGHT) + self.MIN_HEIGHT)
        
        generation_time = time.time() - start_time
        rospy.loginfo(f"Terrain generated in {generation_time:.2f} seconds")
        
        terrain_info = {
            "rng_seed": rng_seed,
            "generation_time": generation_time,
            "features": terrain_features,
            "map_size": self.MAP_SIZE,
            "resolution": self.RESOLUTION,
            "height_range": (self.MIN_HEIGHT, self.MAX_HEIGHT),
            "actual_height_range": (heightmap.min(), heightmap.max())
        }
        
        return heightmap, terrain_info
    
    def heightmap_to_pointcloud(self, heightmap):
        """
        将高度图转换为点云
        
        Args:
            heightmap: 高度图数组
            
        Returns:
            open3d.geometry.PointCloud
        """
        # 创建点云坐标
        points = np.stack([self.xx.ravel(), self.yy.ravel(), heightmap.ravel()], axis=-1).astype(np.float32)
        
        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 使用地形色彩映射进行着色
        import matplotlib.cm as cm
        
        # 设置颜色映射范围
        color_max_height = self.MAX_HEIGHT * 1.0
        color_min_height = self.MIN_HEIGHT
        
        norm_heights = np.clip((heightmap.ravel() - color_min_height) / (color_max_height - color_min_height), 0.0, 1.0)
        colors = cm.terrain(norm_heights)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
        
        return pcd


class GridTransformer:
    """
    地图转换器：将点云转换为包含高程和法向量的栅格地图
    基于原有的地图转换逻辑
    """

    def __init__(self, coarse_resolution=0.2, fine_resolution=0.1, voxel_size=0.1):
        self.coarse_resolution = coarse_resolution
        self.fine_resolution = fine_resolution
        self.voxel_size = voxel_size
        self.search_radius = voxel_size * 2.5

        # 栅格地图数据
        self.elevation_map = None
        self.normal_x_map = None
        self.normal_y_map = None
        self.normal_z_map = None

        # 地图元信息
        self.map_bounds = None
        self.map_center = None
        self.map_size = None

    def preprocess_pointcloud(self, pcd):
        """点云预处理：降采样和法向量计算"""
        rospy.loginfo("Starting point cloud preprocessing...")

        # 体素降采样
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        rospy.loginfo(f"Downsampling: {len(pcd.points)} -> {len(downsampled_pcd.points)} points")

        # 计算法向量
        downsampled_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamRadius(radius=self.search_radius)
        )

        normals_array = np.asarray(downsampled_pcd.normals)
        points_array = np.asarray(downsampled_pcd.points)

        rospy.loginfo("Point cloud preprocessing completed")
        return points_array, normals_array

    def calculate_map_bounds(self, points):
        """计算地图边界"""
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)

        buffer = max(self.coarse_resolution, self.fine_resolution) * 2

        bounds = (
            min_bound[0] - buffer,
            max_bound[0] + buffer,
            min_bound[1] - buffer,
            max_bound[1] + buffer
        )

        return bounds

    def transform_pointcloud_to_grid(self, pcd, map_bounds=None):
        """
        将点云转换为栅格地图
        
        Args:
            pcd: open3d.geometry.PointCloud
            map_bounds: 可选的地图边界 (min_x, max_x, min_y, max_y)
            
        Returns:
            dict: 包含栅格地图数据的字典
        """
        rospy.loginfo("Starting point cloud to grid transformation...")

        # 预处理点云
        points, normals = self.preprocess_pointcloud(pcd)

        # 计算或使用指定的地图边界
        if map_bounds is None:
            bounds = self.calculate_map_bounds(points)
        else:
            bounds = map_bounds
            
        self.map_bounds = bounds
        self.map_center = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2)

        # 分配点到栅格
        grid_dict, map_size = self.allocate_points_to_grid(points, normals, bounds, self.coarse_resolution)
        
        # 处理栅格单元
        coarse_grids = self.process_grid_cells(points, normals, grid_dict, map_size)

        # 插值到精细栅格
        fine_grids = self.interpolate_grid(coarse_grids, bounds, self.fine_resolution)

        # 存储结果
        self.elevation_map, self.normal_x_map, self.normal_y_map, self.normal_z_map = fine_grids
        self.map_size = (fine_grids[0].shape[1], fine_grids[0].shape[0])

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

    def allocate_points_to_grid(self, points, normals, bounds, resolution):
        """将点云分配到栅格单元中"""
        min_x, max_x, min_y, max_y = bounds

        width = int(np.ceil((max_x - min_x) / resolution))
        height = int(np.ceil((max_y - min_y) / resolution))

        grid_dict = {}

        for i, point in enumerate(points):
            x, y = point[0], point[1]

            grid_x = int((x - min_x) / resolution)
            grid_y = int((y - min_y) / resolution)

            grid_x = max(0, min(grid_x, width - 1))
            grid_y = max(0, min(grid_y, height - 1))

            grid_key = (grid_x, grid_y)
            if grid_key not in grid_dict:
                grid_dict[grid_key] = []
            grid_dict[grid_key].append(i)

        return grid_dict, (width, height)

    def process_grid_cells(self, points, normals, grid_dict, map_size):
        """处理栅格单元，计算高程和法向量"""
        width, height = map_size

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

        for (grid_x, grid_y), point_indices in grid_dict.items():
            if len(point_indices) < 1:
                continue

            cell_points = points[point_indices]
            cell_normals = normals[point_indices]

            # 找到法向量Z分量最小的点
            norm_lengths = np.linalg.norm(cell_normals, axis=1)
            normalized_z = np.abs(cell_normals[:, 2] / norm_lengths)

            min_z_local_idx = np.argmin(normalized_z)
            min_z_global_idx = point_indices[min_z_local_idx]

            selected_point = points[min_z_global_idx]
            selected_normal = normals[min_z_global_idx]
            norm_length = np.linalg.norm(selected_normal)
            normalized_normal = selected_normal / norm_length

            elevation_grid[grid_y, grid_x] = selected_point[2]
            normal_x_grid[grid_y, grid_x] = normalized_normal[0]
            normal_y_grid[grid_y, grid_x] = normalized_normal[1]
            normal_z_grid[grid_y, grid_x] = normalized_normal[2]

        # 转换回CPU
        if CUPY_AVAILABLE:
            elevation_grid = cp.asnumpy(elevation_grid)
            normal_x_grid = cp.asnumpy(normal_x_grid)
            normal_y_grid = cp.asnumpy(normal_y_grid)
            normal_z_grid = cp.asnumpy(normal_z_grid)

        rospy.loginfo("Grid cell processing completed")
        return elevation_grid, normal_x_grid, normal_y_grid, normal_z_grid

    def interpolate_grid(self, coarse_grids, bounds, target_resolution):
        """将粗糙栅格插值到精细分辨率"""
        elevation_coarse, normal_x_coarse, normal_y_coarse, normal_z_coarse = coarse_grids
        min_x, max_x, min_y, max_y = bounds

        target_width = int(np.ceil((max_x - min_x) / target_resolution))
        target_height = int(np.ceil((max_y - min_y) / target_resolution))

        coarse_height, coarse_width = elevation_coarse.shape

        x_coarse = np.linspace(min_x, max_x, coarse_width)
        y_coarse = np.linspace(min_y, max_y, coarse_height)

        x_fine = np.linspace(min_x, max_x, target_width)
        y_fine = np.linspace(min_y, max_y, target_height)

        rospy.loginfo("Performing grid interpolation...")

        # 高程使用三次插值
        valid_mask = ~np.isnan(elevation_coarse)
        if np.sum(valid_mask) > 0:
            points = []
            values = []
            for i in range(coarse_height):
                for j in range(coarse_width):
                    if valid_mask[i, j]:
                        points.append([y_coarse[i], x_coarse[j]])
                        values.append(elevation_coarse[i, j])

            if len(points) > 3:
                from scipy.interpolate import griddata
                xi, yi = np.meshgrid(x_fine, y_fine, indexing='xy')
                elevation_fine = griddata(points, values, (yi, xi), method='cubic', fill_value=np.nan)
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
                yi_idx = np.argmin(np.abs(y_coarse - y_fine[i]))
                xi_idx = np.argmin(np.abs(x_coarse - x_fine[j]))
                result[i, j] = data[yi_idx, xi_idx]

        return result


class TerrainDatasetGenerator:
    """
    地形数据集生成器：结合地形生成和路径规划
    """
    
    def __init__(self):
        """初始化数据集生成器"""
        rospy.init_node('terrain_dataset_generator', anonymous=False)
        
        # 读取参数
        self.num_environments = rospy.get_param('~num_environments', 1000)
        self.train_paths_per_env = rospy.get_param('~train_paths_per_env', 50)
        self.val_paths_per_env = rospy.get_param('~val_paths_per_env', 5)
        self.dataset_dir = rospy.get_param('~dataset_dir', 'dataset')
        self.start_env_id = rospy.get_param('~start_env_id', 0)
        
        # 地形生成参数
        self.map_size = rospy.get_param('~map_size', 10.0)
        self.map_resolution = rospy.get_param('~map_resolution', 0.02)
        self.max_height = rospy.get_param('~max_height', 3.0)
        self.min_height = rospy.get_param('~min_height', 0.0)
        
        # 地图转换参数
        self.coarse_resolution = rospy.get_param('~coarse_resolution', 0.2)
        self.fine_resolution = rospy.get_param('~fine_resolution', 0.1)
        self.voxel_size = rospy.get_param('~voxel_size', 0.1)
        
        # 路径生成参数
        self.map_x_min = -self.map_size / 2 + 1.0
        self.map_x_max = self.map_size / 2 - 1.0
        self.map_y_min = -self.map_size / 2 + 1.0
        self.map_y_max = self.map_size / 2 - 1.0
        self.min_distance = rospy.get_param('~min_distance', 2.0)
        self.publish_delay = rospy.get_param('~publish_delay', 0.1)
        
        # 状态变量
        self.current_env_id = self.start_env_id
        self.current_path_id = 0
        self.current_phase = 'train'  # 'train' or 'val'
        self.paths_generated_for_current_env = 0
        self.waiting_for_result = False
        self.current_trajectory = None
        self.map_update_timeout = rospy.get_param('~map_update_timeout', 10.0)

        # 新增：用于轨迹稳定性验证的当前地图缓存
        self.current_normal_x = None
        self.current_normal_y = None
        self.current_normal_z = None
        self.current_map_bounds = None
        self.current_resolution = None
        self.current_yaw_scores = None
        self.current_yaw_bins = None

        # 添加轨迹标识，防止地图切换时的轨迹混乱
        self.expected_env_id = self.start_env_id
        self.expected_path_id = 0
        self.map_generation_timestamp = 0.0
        
        # 创建数据集目录结构
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.val_dir = os.path.join(self.dataset_dir, 'val')
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        
        # 初始化组件
        self.terrain_generator = TerrainGenerator(
            map_size=self.map_size,
            resolution=self.map_resolution,
            max_height=self.max_height,
            min_height=self.min_height
        )
        
        self.grid_transformer = GridTransformer(
            coarse_resolution=self.coarse_resolution,
            fine_resolution=self.fine_resolution,
            voxel_size=self.voxel_size
        )
        
        # ROS通信
        self.pointcloud_pub = rospy.Publisher('/uneven_map/pointcloud', PointCloud2, queue_size=1, latch=True)
        # Occupancy map publisher (external occ map to uneven_map)
        self.occ_pub = rospy.Publisher('/external_occ_map', OccupancyGrid, queue_size=1, latch=True)
        # 3D occupancy grid publisher (HWY layout) for receivers that expect HWY flattened
        from std_msgs.msg import Float32MultiArray
        self.occ3d_pub = rospy.Publisher('/external_occ_grid_hwy', Float32MultiArray, queue_size=1, latch=True)
        self.start_pose_pub = rospy.Publisher('/data_generate_node/start_pose', PoseStamped, queue_size=1)
        self.target_pose_pub = rospy.Publisher('/data_generate_node/target_pose', PoseStamped, queue_size=1)
        
        self.traj_sub = rospy.Subscriber('/data_generate_node/optimized_traj', SE2Traj, self.trajectory_callback)
        self.result_sub = rospy.Subscriber('/data_generate_node/planning_result', Bool, self.planning_result_callback)
        self.map_regen_sub = rospy.Subscriber('/data_generate_node/map_regeneration_request', Bool, self.map_regeneration_callback)
        
        rospy.loginfo(f"TerrainDatasetGenerator initialized:")
        rospy.loginfo(f"  - num_environments: {self.num_environments}")
        rospy.loginfo(f"  - train_paths_per_env: {self.train_paths_per_env}")
        rospy.loginfo(f"  - val_paths_per_env: {self.val_paths_per_env}")
        rospy.loginfo(f"  - dataset_dir: {self.dataset_dir}")
        rospy.loginfo(f"  - start_env_id: {self.start_env_id}")
        
        # 等待服务启动
        rospy.loginfo("Waiting for /start_data_generation service...")
        rospy.wait_for_service('/start_data_generation', timeout=30)
        try:
            trigger = rospy.ServiceProxy('/start_data_generation', Trigger)
            resp = trigger()
            rospy.loginfo("start_data_generation called: success=%s msg=%s", resp.success, resp.message)
        except Exception as e:
            rospy.logerr("Failed to call /start_data_generation: %s", e)
    
    def pointcloud_to_ros_message(self, pcd):
        """将Open3D点云转换为ROS PointCloud2消息"""
        import sensor_msgs.point_cloud2 as pc2
        from sensor_msgs.msg import PointField
        
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # 创建ROS PointCloud2消息
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        
        # 使用更简单可靠的方法创建点云
        ros_pointcloud = pc2.create_cloud_xyz32(header, points)
        
        return ros_pointcloud
    
    def copy_map_to_val_directory(self):
        """将地图文件从训练目录复制到验证目录"""
        env_name = f"env{self.current_env_id:06d}"
        train_env_dir = os.path.join(self.train_dir, env_name)
        val_env_dir = os.path.join(self.val_dir, env_name)
        
        # 确保验证目录存在
        os.makedirs(val_env_dir, exist_ok=True)
        
        # 需要复制的文件列表
        files_to_copy = ['map.p', 'terrain_2d.png', 'terrain_3d.png']
        
        import shutil
        for file_name in files_to_copy:
            src_file = os.path.join(train_env_dir, file_name)
            dst_file = os.path.join(val_env_dir, file_name)
            
            if os.path.exists(src_file):
                try:
                    shutil.copy2(src_file, dst_file)
                    rospy.loginfo(f"Copied {file_name} to val directory: {env_name}")
                except Exception as e:
                    rospy.logwarn(f"Failed to copy {file_name} to val directory: {e}")
            else:
                rospy.logwarn(f"Source file not found: {src_file}")
    
    def generate_new_environment(self):
        """生成新的地形环境"""
        env_name = f"env{self.current_env_id:06d}"
        
        # 选择输出目录
        if self.current_phase == 'train':
            env_dir = os.path.join(self.train_dir, env_name)
        else:
            env_dir = os.path.join(self.val_dir, env_name)
        
        os.makedirs(env_dir, exist_ok=True)
        
        # 如果重新生成，清理旧的路径文件但保留目录结构
        import glob
        path_files = glob.glob(os.path.join(env_dir, "path_*.p"))
        for path_file in path_files:
            try:
                os.remove(path_file)
                rospy.loginfo(f"Removed old path file: {os.path.basename(path_file)}")
            except OSError:
                pass
        
        rospy.loginfo(f"Generating environment {env_name} for {self.current_phase} phase")
        
        # 生成地形
        heightmap, terrain_info = self.terrain_generator.generate_terrain()
        
        # 转换为点云
        pcd = self.terrain_generator.heightmap_to_pointcloud(heightmap)
        
        # 转换为栅格地图
        map_bounds = (-self.map_size/2, self.map_size/2, -self.map_size/2, self.map_size/2)
        grid_map_data = self.grid_transformer.transform_pointcloud_to_grid(pcd, map_bounds)

        # 保存当前地图的法向量/边界/分辨率，供接收到轨迹后验证使用
        try:
            self.current_normal_x = np.array(grid_map_data.get('normal_x'))
            self.current_normal_y = np.array(grid_map_data.get('normal_y'))
            self.current_normal_z = np.array(grid_map_data.get('normal_z'))
            self.current_map_bounds = grid_map_data.get('bounds')
            self.current_resolution = float(grid_map_data.get('resolution'))
            # 计算yaw_bins并预计算每个格子的朝向稳定性
            yaw_res = rospy.get_param('uneven_map/yaw_resolution', 0.1)
            self.current_yaw_bins = int(math.ceil(2.0 * math.pi / float(yaw_res)))
            try:
                # compute_map_yaw_bins 返回 torch.Tensor，可转换为 numpy
                yaw_scores = compute_map_yaw_bins(self.current_normal_x, self.current_normal_y, self.current_normal_z, self.current_yaw_bins)
                yaw_scores = np.array(yaw_scores)
                # normalize to (H, W, Y)
                try:
                    Hn, Wn = self.current_normal_x.shape
                    Y = int(self.current_yaw_bins)
                    aligned = None
                    if yaw_scores.ndim == 3:
                        if yaw_scores.shape == (Hn, Wn, Y):
                            aligned = yaw_scores
                        elif yaw_scores.shape == (Y, Hn, Wn):
                            aligned = np.transpose(yaw_scores, (1, 2, 0))
                        elif yaw_scores.shape == (Wn, Hn, Y):
                            aligned = np.transpose(yaw_scores, (1, 0, 2))
                        elif yaw_scores.shape[0] == Y and yaw_scores.shape[1] == Hn and yaw_scores.shape[2] == Wn:
                            aligned = np.transpose(yaw_scores, (1, 2, 0))
                    if aligned is None:
                        rospy.logwarn(f"Precomputed yaw_scores has unexpected shape {yaw_scores.shape}, discarding it")
                        self.current_yaw_scores = None
                    else:
                        self.current_yaw_scores = np.array(aligned)
                        rospy.loginfo(f"Precomputed yaw stability map with shape {self.current_yaw_scores.shape}")
                except Exception as e:
                    rospy.logwarn(f"Failed to align yaw_scores shape for caching: {e}")
                    self.current_yaw_scores = None
            except Exception as e:
                rospy.logwarn(f"Failed to compute yaw stability map: {e}")
                self.current_yaw_scores = None
        except Exception as e:
            rospy.logwarn(f"Failed to cache map normals for trajectory validation: {e}")
            self.current_normal_x = self.current_normal_y = self.current_normal_z = None
            self.current_map_bounds = None
            self.current_resolution = None
            self.current_yaw_scores = None
            self.current_yaw_bins = None

        # Ensure grid_map_tensor is defined: build from elevation + normals, with fallbacks
        try:
            elevation = grid_map_data.get('elevation', None)
            nx = grid_map_data.get('normal_x', None)
            ny = grid_map_data.get('normal_y', None)
            nz = grid_map_data.get('normal_z', None)

            # If a stacked tensor was provided instead
            if elevation is None and 'tensor' in grid_map_data:
                try:
                    tensor = np.array(grid_map_data['tensor'])
                    if tensor.ndim == 3 and tensor.shape[2] >= 4:
                        elevation = tensor[:, :, 0]
                        nx = tensor[:, :, 1]
                        ny = tensor[:, :, 2]
                        nz = tensor[:, :, 3]
                except Exception:
                    pass

            # Convert to numpy arrays if necessary
            if elevation is not None:
                elevation = np.array(elevation, dtype=np.float32)
            if nx is not None:
                nx = np.array(nx, dtype=np.float32)
            if ny is not None:
                ny = np.array(ny, dtype=np.float32)
            if nz is not None:
                nz = np.array(nz, dtype=np.float32)

            # If any component missing, create safe defaults based on available shape
            if elevation is None:
                shape = None
                for arr in (nx, ny, nz):
                    if arr is not None:
                        shape = arr.shape
                        break
                if shape is None:
                    shape = (1, 1)
                elevation = np.zeros(shape, dtype=np.float32)
            if nx is None:
                nx = np.zeros_like(elevation)
            if ny is None:
                ny = np.zeros_like(elevation)
            if nz is None:
                nz = np.ones_like(elevation, dtype=np.float32)  # assume flat if unknown

            grid_map_tensor = np.stack([elevation, nx, ny, nz], axis=-1)
        except Exception as e:
            rospy.logwarn(f"Failed to build grid_map_tensor from grid_map_data: {e}")
            grid_map_tensor = np.zeros((1, 1, 4), dtype=np.float32)

        # 保存地图数据，按照正确的格式
        map_data = {
            'tensor': grid_map_tensor,
            'bounds': grid_map_data['bounds'],
            'resolution': grid_map_data['resolution'],
            'map_name': env_name,
            'channels': ['elevation', 'normal_x', 'normal_y', 'normal_z'],
            'shape': grid_map_tensor.shape
        }
        
        map_file = os.path.join(env_dir, 'map.p')
        with open(map_file, 'wb') as f:
            pickle.dump(map_data, f)
        
        rospy.loginfo(f"Saved map data: {map_file}")
        # Create and publish occupancy map for uneven_map to consume
        try:
            occ_msg = self.create_occupancy_grid(grid_map_data)
            # if occ_msg is not None:
            #     # Wait briefly for uneven_map to subscribe, avoid occ callback accessing uninitialized buffers
            #     wait_start = rospy.Time.now()
            #     wait_timeout = rospy.Duration(5.0)  # adjust as needed
            #     while self.occ_pub.get_num_connections() == 0 and (rospy.Time.now() - wait_start) < wait_timeout:
            #         rospy.sleep(0.05)
            #     if self.occ_pub.get_num_connections() == 0:
            #         rospy.logwarn("No subscribers for /external_occ_map after waiting, publishing anyway")
            #     # self.occ_pub.publish(occ_msg)
            #     rospy.loginfo(f"Published occupancy grid for {env_name} to /external_occ_map (subscribers={self.occ_pub.get_num_connections()})")
        except Exception as e:
            rospy.logwarn(f"Failed to create/publish occupancy grid: {e}")
        
        # 保存地形可视化
        self.save_terrain_visualizations(heightmap, terrain_info, env_dir)
        
        # 发布点云到ROS
        ros_pointcloud = self.pointcloud_to_ros_message(pcd)
        self.pointcloud_pub.publish(ros_pointcloud)
        
        rospy.loginfo(f"Published terrain point cloud for {env_name}")
        
        # 更新地图生成时间戳，用于轨迹验证
        self.map_generation_timestamp = rospy.Time.now().to_sec()
        
        # 清理任何待处理的轨迹回调状态，防止旧地图的轨迹被保存到新地图
        self.waiting_for_result = False
        self.current_trajectory = None
        
        # 更新期望的环境和路径ID
        self.expected_env_id = self.current_env_id
        self.expected_path_id = 0
        
        # 等待地图更新 - 增加等待时间确保地图系统完全处理新地形
        rospy.sleep(3.0)
        
        # 重置路径计数
        self.current_path_id = 0
        self.paths_generated_for_current_env = 0
        
        return env_dir
    
    def save_terrain_visualizations(self, heightmap, terrain_info, env_dir):
        """保存地形可视化图片"""
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        
        try:
            plt.ioff()  # 关闭交互模式
            
            # 2D高度图
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(heightmap.T, origin='lower', 
                          extent=[-self.map_size/2, self.map_size/2, -self.map_size/2, self.map_size/2],
                          cmap='terrain', interpolation='bilinear')
            plt.colorbar(im, ax=ax, label='Height (m)')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Terrain Height Map\n(Seed: {terrain_info["rng_seed"]})')
            plt.tight_layout()
            plt.savefig(os.path.join(env_dir, 'terrain_2d.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            plt.clf()  # 清除当前图形
            
            # 3D可视化
            fig = plt.figure(figsize=(12, 8))
            plt.tight_layout()
            plt.savefig(os.path.join(env_dir, 'terrain_2d.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            plt.clf()  # 清除当前图形
            
            # 3D可视化
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            x = np.linspace(-self.map_size/2, self.map_size/2, heightmap.shape[1])
            y = np.linspace(-self.map_size/2, self.map_size/2, heightmap.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # 降采样以减少内存使用
            step = max(1, min(heightmap.shape) // 100)
            X_sub = X[::step, ::step]
            Y_sub = Y[::step, ::step]
            Z_sub = heightmap[::step, ::step]
            
            surf = ax.plot_surface(X_sub, Y_sub, Z_sub, cmap='terrain', alpha=0.8)
            
            # 设置轴标签
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Height (m)')
            
            # 设置轴的实际数据范围
            ax.set_xlim(-self.map_size/2, self.map_size/2)  # -5到5米
            ax.set_ylim(-self.map_size/2, self.map_size/2)  # -5到5米
            
            # Z轴使用实际的高度范围
            z_min, z_max = heightmap.min(), heightmap.max()
            z_buffer = (z_max - z_min) * 0.05  # 添加5%的缓冲
            ax.set_zlim(z_min - z_buffer, z_max + z_buffer)
            
            # 关键：确保1米在每个轴上的视觉长度相同
            # 计算每个轴的实际数据范围
            x_data_range = self.map_size  # 10米
            y_data_range = self.map_size  # 10米
            z_data_range = (z_max + z_buffer) - (z_min - z_buffer)  # 实际高度范围
            
            # 兼容性更好的方法设置轴比例
            # 对于旧版本的matplotlib，使用替代方法
            try:
                # matplotlib 3.3+ 支持 set_box_aspect
                ax.set_box_aspect([x_data_range, y_data_range, z_data_range])
            except AttributeError:
                # 对于旧版本matplotlib的备用方法
                # 通过设置轴范围来近似实现比例一致性
                max_range = max(x_data_range, y_data_range, z_data_range)
                
                # 计算需要扩展的范围使所有轴具有相同的显示范围
                x_center = 0.0  # X轴中心
                y_center = 0.0  # Y轴中心
                z_center = (z_min + z_max) / 2  # Z轴中心
                
                half_range = max_range / 2
                
                # 只扩展较小的轴，保持数据完整可见
                if x_data_range < max_range:
                    ax.set_xlim(x_center - half_range, x_center + half_range)
                if y_data_range < max_range:
                    ax.set_ylim(y_center - half_range, y_center + half_range)
                if z_data_range < max_range:
                    ax.set_zlim(z_center - half_range, z_center + half_range)
            
            # 注意：不要使用归一化的比例，而要使用实际的数据范围
            # 这样10米的XY范围就会比6米的Z范围在视觉上更长
            
            ax.set_title(f'3D Terrain View\n(Features: {list(terrain_info["features"].keys())})')
            plt.colorbar(surf, ax=ax, shrink=0.5)
            plt.savefig(os.path.join(env_dir, 'terrain_3d.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            plt.clf()  # 清除当前图形
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            rospy.loginfo("Saved terrain visualizations")
            
        except Exception as e:
            rospy.logwarn(f"Failed to save terrain visualizations: {e}")
            # 确保即使出错也清理资源
            plt.close('all')
            import gc
            gc.collect()
    
    def generate_random_pose(self):
        """生成随机位姿"""
        x = random.uniform(self.map_x_min, self.map_x_max)
        y = random.uniform(self.map_y_min, self.map_y_max)
        yaw = random.uniform(-np.pi, np.pi)
        return (x, y, yaw)
    
    def create_pose_stamped(self, x, y, yaw):
        """创建PoseStamped消息"""
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        
        quaternion = tf_trans.quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        
        return pose
    
    def generate_pose_pair(self):
        """生成起始点和目标点位姿对"""
        max_attempts = 50
        
        for attempt in range(max_attempts):
            start_x, start_y, start_yaw = self.generate_random_pose()
            target_x, target_y, target_yaw = self.generate_random_pose()
            
            distance = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
            
            if distance >= self.min_distance:
                return (start_x, start_y, start_yaw), (target_x, target_y, target_yaw)
        
        # 默认安全位姿对
        rospy.logwarn("Failed to generate valid pose pair, using default safe poses")
        return (-2.0, -2.0, 0.0), (2.0, 2.0, 0.0)
    
    def planning_result_callback(self, msg):
        """规划结果回调函数"""
        if not self.waiting_for_result:
            rospy.logdebug("Received planning result but not waiting for result, ignoring")
            return
        
        # 验证当前状态是否匹配
        if self.current_env_id != self.expected_env_id or self.current_path_id != self.expected_path_id:
            rospy.logwarn(f"Received planning result for mismatched env/path: "
                         f"current({self.current_env_id}, {self.current_path_id}) vs "
                         f"expected({self.expected_env_id}, {self.expected_path_id}), ignoring")
            return
        
        if msg.data:
            rospy.loginfo(f"Planning successful for path_{self.current_path_id}")
            # 等待轨迹消息
        else:
            rospy.logwarn(f"Planning failed for path_{self.current_path_id}, retrying...")
            self.waiting_for_result = False
            # 延迟后重试
            rospy.Timer(rospy.Duration(self.publish_delay), self.generate_next_path, oneshot=True)
    
    def map_regeneration_callback(self, msg):
        """地图重新生成请求回调函数"""
        if not msg.data:
            rospy.loginfo("Received map regeneration request with data=False, ignoring")
            return
        
        rospy.logwarn(f"Received map regeneration request for env{self.current_env_id:06d}")
        rospy.logwarn(f"Current state: waiting_for_result={self.waiting_for_result}, current_path_id={self.current_path_id}")
        
        # 清理所有待处理状态，防止旧轨迹干扰
        self.waiting_for_result = False
        self.current_trajectory = None
        
        # 重新生成当前环境的地形
        rospy.Timer(rospy.Duration(0.5), self.regenerate_current_environment, oneshot=True)
    
    def trajectory_callback(self, msg):
        """轨迹回调函数"""
        if not self.waiting_for_result:
            rospy.logdebug("Received trajectory but not waiting for result, ignoring")
            return
        
        # 检查是否是当前地图和路径对应的轨迹
        if self.current_env_id != self.expected_env_id or self.current_path_id != self.expected_path_id:
            rospy.logwarn(f"Received trajectory for mismatched env/path: "
                         f"current({self.current_env_id}, {self.current_path_id}) vs "
                         f"expected({self.expected_env_id}, {self.expected_path_id}), ignoring")
            return
        
        rospy.loginfo(f"Received trajectory for path_{self.current_path_id}")

        # 先不直接接受轨迹，先验证每个轨迹点是否在不会倾覆的位置
        try:
            traj_points = self.sample_trajectory(msg)
        except Exception as e:
            rospy.logwarn(f"Failed to sample trajectory for validation: {e}")
            traj_points = None

        # 如果无法获取轨迹点，视为无效
        if traj_points is None:
            rospy.logwarn(f"Invalid trajectory for path_{self.current_path_id}, retrying...")
            self.waiting_for_result = False
            rospy.Timer(rospy.Duration(self.publish_delay), self.generate_next_path, oneshot=True)
            return

        # 如果预计算了 yaw_scores，使用其进行精确验证
        invalid_found = False
        # 强制要求使用预计算的 yaw_scores 进行验证；如果不可用则直接拒绝轨迹，避免回退导致保存不安全轨迹
        if self.current_yaw_scores is None or self.current_map_bounds is None or self.current_resolution is None:
            rospy.logwarn("Rejecting trajectory because precomputed yaw stability map or map bounds/resolution is unavailable")
            invalid_found = True
        else:
            min_x, max_x, min_y, max_y = self.current_map_bounds
            H, W, Y = self.current_yaw_scores.shape[0], self.current_yaw_scores.shape[1], self.current_yaw_scores.shape[2]
            for (x, y, yaw) in traj_points:
                # 转换到栅格索引
                ix = int(math.floor((x - min_x) / self.current_resolution))
                iy = int(math.floor((y - min_y) / self.current_resolution))
                if ix < 0 or ix >= W or iy < 0 or iy >= H:
                    rospy.logwarn(f"Trajectory point out of precomputed map bounds: [{x:.3f},{y:.3f}], rejecting trajectory")
                    invalid_found = True
                    break

                # yaw bin
                yaw_norm = yaw
                # normalize to [-pi, pi)
                while yaw_norm < -math.pi:
                    yaw_norm += 2*math.pi
                while yaw_norm >= math.pi:
                    yaw_norm -= 2*math.pi
                bin_width = 2*math.pi / float(Y)
                yidx = int(math.floor((yaw_norm + math.pi) / bin_width))
                if yidx < 0:
                    yidx = 0
                if yidx >= Y:
                    yidx = Y-1

                # yaw_scores: 1 -> stable, 0 -> tipping
                # 注意：numpy 数组的第一个维度是行（y / height），第二个维度是列（x / width）
                # 之前错误地使用了 [ix, iy, yidx]，会导致 X/Y 轴混淆。改为 [iy, ix, yidx]
                try:
                    val = self.current_yaw_scores[iy, ix, yidx]
                except IndexError:
                    rospy.logwarn(f"Yaw score index error for traj point [{x:.3f},{y:.3f}] -> idx (ix={ix}, iy={iy}, yidx={yidx})")
                    invalid_found = True
                    break

                # 处理可能的 NaN / 非数值情况并进行阈值判定
                try:
                    # 将 val 转为浮点数后比较阈值
                    stable = not (math.isnan(float(val)) or float(val) <= 0.5)
                except Exception:
                    # 回退：任何非零/非 False 的值视为稳定
                    try:
                        stable = bool(val)
                    except Exception:
                        stable = False

                if not stable:
                    rospy.logwarn(f"Detected tipping at traj point x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} -> unstable, discarding trajectory")
                    invalid_found = True
                    break
        if invalid_found:
            rospy.logwarn(f"Trajectory for path_{self.current_path_id} invalid due to tipping points, retrying...")
            self.waiting_for_result = False
            # 继续生成新的路径尝试
            rospy.Timer(rospy.Duration(self.publish_delay), self.generate_next_path, oneshot=True)
            return

        # 验证通过，接受轨迹并继续处理
        self.current_trajectory = msg
        self.waiting_for_result = False
        self.process_trajectory()

    def sample_trajectory(self, trajectory):
        """提取轨迹点"""
        trajectory_points = []
        
        pos_pts = trajectory.pos_pts
        angle_pts = trajectory.angle_pts
        
        if len(pos_pts) ==  0 or len(angle_pts) == 0:
            rospy.logwarn("Empty trajectory received")
            return None
        
        min_length = min(len(pos_pts), len(angle_pts))
        if min_length == 0:
            rospy.logwarn("Zero-length trajectory")
            return None
        
        for i in range(min_length):
            x = pos_pts[i].x
            y = pos_pts[i].y
            yaw = angle_pts[i].x  # 角度存储在Point的x字段中
            trajectory_points.append([x, y, yaw])
        
        return np.array(trajectory_points)
    
    def process_trajectory(self):
        """处理接收到的轨迹"""
        trajectory_path = self.sample_trajectory(self.current_trajectory)
        
        if trajectory_path is None:
            rospy.logwarn(f"Invalid trajectory for path_{self.current_path_id}, retrying...")
            rospy.Timer(rospy.Duration(self.publish_delay), self.generate_next_path, oneshot=True)
            return
        
        # 确定当前环境目录
        env_name = f"env{self.current_env_id:06d}"
        if self.current_phase == 'train':
            env_dir = os.path.join(self.train_dir, env_name)
        else:
            env_dir = os.path.join(self.val_dir, env_name)
        
        # 确保目录存在
        os.makedirs(env_dir, exist_ok=True)
        
        # 创建路径数据，按照正确的格式
        path_data = {
            'path': trajectory_path,
            'map_name': env_name
        }
        
        # 保存路径文件
        path_file = os.path.join(env_dir, f"path_{self.current_path_id}.p")
        with open(path_file, 'wb') as f:
            pickle.dump(path_data, f)
        
        rospy.loginfo(f"Saved {path_file} with {len(trajectory_path)} points")
        
        # 更新计数
        self.current_path_id += 1
        self.paths_generated_for_current_env += 1
        
        # 检查当前环境是否完成
        target_paths = self.train_paths_per_env if self.current_phase == 'train' else self.val_paths_per_env
        
        if self.paths_generated_for_current_env >= target_paths:
            rospy.loginfo(f"Completed {self.current_phase} phase for environment {self.current_env_id}")
            
            if self.current_phase == 'train':
                # 切换到验证阶段，复制地图文件到val目录
                # 清理待处理状态，确保验证阶段有干净的开始
                self.waiting_for_result = False
                self.current_trajectory = None
                
                self.current_phase = 'val'
                self.current_path_id = 0
                self.paths_generated_for_current_env = 0
                
                
                # 复制地图文件到验证目录
                self.copy_map_to_val_directory()
            else:
                # 当前环境完全完成，移动到下一个环境
                # 清理待处理状态，防止旧轨迹干扰新环境
                self.waiting_for_result = False
                self.current_trajectory = None
                
                self.current_env_id += 1
                self.current_phase = 'train'
                self.current_path_id = 0
                self.paths_generated_for_current_env = 0
                
                # 检查是否所有环境都已完成
                if self.current_env_id >= self.start_env_id + self.num_environments:
                    rospy.loginfo("Dataset generation completed!")
                    rospy.signal_shutdown("Dataset generation completed")
                    return
                
                # 生成新环境
                rospy.Timer(rospy.Duration(1.0), self.start_new_environment, oneshot=True)
                return
        
        # 生成当前环境的下一条路径
        rospy.Timer(rospy.Duration(self.publish_delay), self.generate_next_path, oneshot=True)
    
    def regenerate_current_environment(self, event):
        """重新生成当前环境的地形"""
        rospy.logwarn(f"Regenerating terrain for env{self.current_env_id:06d}")
        
        # 清理待处理状态，防止旧轨迹干扰
        self.waiting_for_result = False
        self.current_trajectory = None
        
        # 重置当前路径ID为0，重新开始生成该环境的路径
        self.current_path_id = 0
        self.paths_generated_for_current_env = 0  # 重置路径计数器
        
        # 重新生成地形
        env_dir = self.generate_new_environment()
        
        # 给更多时间让地图系统处理新地形
        rospy.Timer(rospy.Duration(5.0), self.generate_next_path, oneshot=True)
    
    def start_new_environment(self, event):
        """开始新环境"""
        env_dir = self.generate_new_environment()
        # 延迟后开始生成路径
        rospy.Timer(rospy.Duration(2.0), self.generate_next_path, oneshot=True)
    
    def generate_next_path(self, event):
        """生成下一条路径"""
        # 检查系统状态，避免在不合适的时候发送位姿
        if self.waiting_for_result:
            # rospy.logwarn("Still waiting for previous result, skipping path generation")
            # 延迟重试
            rospy.Timer(rospy.Duration(1.0), self.generate_next_path, oneshot=True)
            return
        
        # 生成随机位姿对
        (start_x, start_y, start_yaw), (target_x, target_y, target_yaw) = self.generate_pose_pair()
        
        # 创建位姿消息
        start_pose = self.create_pose_stamped(start_x, start_y, start_yaw)
        target_pose = self.create_pose_stamped(target_x, target_y, target_yaw)
        
        # 设置等待状态
        self.waiting_for_result = True
        
        # 更新期望的路径ID
        self.expected_path_id = self.current_path_id
        
        env_name = f"env{self.current_env_id:06d}"
        rospy.loginfo(f"Generating path_{self.current_path_id} for {env_name} ({self.current_phase})")
        rospy.loginfo(f"  Start: [{start_x:.3f}, {start_y:.3f}, {start_yaw:.3f}]")
        rospy.loginfo(f"  Target: [{target_x:.3f}, {target_y:.3f}, {target_yaw:.3f}]")
        
        # 发布位姿
        self.start_pose_pub.publish(start_pose)
        rospy.sleep(0.1)
        self.target_pose_pub.publish(target_pose)
        
        # 添加超时检查，防止永久等待
        rospy.Timer(rospy.Duration(self.map_update_timeout), self.check_planning_timeout, oneshot=True)
    
    def check_planning_timeout(self, event):
        """检查规划超时"""
        if self.waiting_for_result:
            rospy.logwarn(f"Planning timeout detected for path_{self.current_path_id}, resetting state")
            self.waiting_for_result = False
            # 延迟重试生成路径
            rospy.Timer(rospy.Duration(1.0), self.generate_next_path, oneshot=True)
    
    def start_generation(self):
        """开始数据集生成"""
        rospy.loginfo(f"Starting terrain dataset generation:")
        rospy.loginfo(f"  Environments: {self.num_environments} (starting from {self.start_env_id})")
        rospy.loginfo(f"  Paths per environment: {self.train_paths_per_env} (train) + {self.val_paths_per_env} (val)")
        rospy.loginfo(f"  Total paths: {self.num_environments * (self.train_paths_per_env + self.val_paths_per_env)}")
        
        # 等待其他节点启动
        rospy.sleep(5.0)
        
        # 生成第一个环境
        self.start_new_environment(None)
    
    def run(self):
        """运行数据集生成器"""
        self.start_generation()
        rospy.spin()
    
    def create_occupancy_grid(self, grid_map_data=None):
        """Create a nav_msgs/OccupancyGrid based on compute_map_yaw_bins results.
        - Compute yaw_bins from uneven_map/yaw_resolution
        - Call compute_map_yaw_bins(normal_x, normal_y, normal_z, yaw_bins) if available
        - Convert per-XY yaw scores into occupancy: if max score < threshold_frac => occupied
        """
        from nav_msgs.msg import OccupancyGrid, MapMetaData

        # read parameters from uneven_map to ensure exact alignment
        map_size_x = rospy.get_param('uneven_map/map_size_x', rospy.get_param('map_size', 10.0))
        map_size_y = rospy.get_param('uneven_map/map_size_y', map_size_x)
        xy_res = rospy.get_param('uneven_map/xy_resolution', rospy.get_param('map_resolution', 0.02))
        yaw_res = rospy.get_param('uneven_map/yaw_resolution', 0.1)
        occ_threshold = rospy.get_param('uneven_map/occ_threshold', getattr(self, 'occ_threshold', 50))

        width = int(math.ceil(map_size_x / xy_res))
        height = int(math.ceil(map_size_y / xy_res))
       


        rospy.loginfo('create_occupancy_grid: using map_size_x=%s, map_size_y=%s, xy_res=%s -> width=%d, height=%d (yaw_res=%s)',
                      map_size_x, map_size_y, xy_res, width, height, yaw_res)

        occ = OccupancyGrid()
        occ.header.stamp = rospy.Time.now()
        occ.header.frame_id = 'world'
        occ.info = MapMetaData()
        occ.info.resolution = float(xy_res)
        occ.info.width = int(width)
        occ.info.height = int(height)
        occ.info.origin.position.x = -map_size_x / 2.0
        occ.info.origin.position.y = -map_size_y / 2.0
        occ.info.origin.position.z = 0.0
        occ.info.origin.orientation.x = 0.0
        occ.info.origin.orientation.y = 0.0
        occ.info.origin.orientation.z = 0.0
        occ.info.origin.orientation.w = 1.0

        total_cells = occ.info.width * occ.info.height
        data = np.full((occ.info.height, occ.info.width), -1, dtype=np.int8)

        # Extract normals/elevation from grid_map_data if it's a dict (our earlier save format)
        normal_x = normal_y = normal_z = None
        heightmap = None
        if isinstance(grid_map_data, dict):
            # grid_map_data expected keys: 'tensor' or individual maps
            try:
                if 'tensor' in grid_map_data:
                    tensor = np.array(grid_map_data['tensor'])
                    # tensor shape assumed (H, W, 4) with channels [elevation, nx, ny, nz]
                    if tensor.ndim == 3 and tensor.shape[2] >= 4:
                        heightmap = tensor[:, :, 0]
                        normal_x = tensor[:, :, 1]
                        normal_y = tensor[:, :, 2]
                        normal_z = tensor[:, :, 3]
                    else:
                        rospy.logwarn('create_occupancy_grid: tensor shape unexpected: %s', str(tensor.shape))
                else:
                    # try individual fields
                    if 'normal_x' in grid_map_data and 'normal_y' in grid_map_data and 'normal_z' in grid_map_data:
                        normal_x = np.array(grid_map_data['normal_x'])
                        normal_y = np.array(grid_map_data['normal_y'])
                        normal_z = np.array(grid_map_data['normal_z'])
                    if 'elevation' in grid_map_data:
                        heightmap = np.array(grid_map_data['elevation'])
            except Exception as e:
                rospy.logwarn('create_occupancy_grid: failed to extract normals/elevation from grid_map_data: %s', str(e))
        else:
            # treat grid_map_data as array-like heightmap
            if grid_map_data is not None:
                arr = np.array(grid_map_data)
                if arr.size == total_cells:
                    try:
                        heightmap = arr.reshape((occ.info.height, occ.info.width))
                    except Exception:
                        heightmap = np.resize(arr, (occ.info.height, occ.info.width))
                elif arr.ndim == 2 and arr.shape == (occ.info.height, occ.info.width):
                    heightmap = arr
                else:
                    rospy.logwarn('create_occupancy_grid: incoming grid_map_data shape %s does not match target (%d,%d). Resizing using numpy.resize.', str(arr.shape), occ.info.height, occ.info.width)
                    heightmap = np.resize(arr, (occ.info.height, occ.info.width))

        # compute yaw_bins from yaw_resolution
        yaw_bins = int(math.ceil(2.0 * math.pi / float(yaw_res)))
        rospy.loginfo('create_occupancy_grid: computed yaw_bins=%d from yaw_resolution=%f', yaw_bins, yaw_res)

        # 计算每个栅格的最大倾覆概率
        if normal_x is not None and normal_y is not None and normal_z is not None:
            try:
                # 使用法向量计算每个栅格在各个航向下的稳定性评分 (1.0 = stable, 0.0 = tipping)
                stability_map = compute_map_yaw_bins(normal_x, normal_y, normal_z, yaw_bins)

                # 如果返回的是 torch.Tensor，转换为 numpy
                try:
                    import torch
                    if isinstance(stability_map, torch.Tensor):
                        stability_map = stability_map.detach().cpu().numpy()
                except Exception:
                    pass

                stability_map = np.array(stability_map)

                # 规范形状为 (H, W, Y)
                H = occ.info.height
                W = occ.info.width
                Y = int(yaw_bins)

                if stability_map.ndim == 3:
                    # 尝试匹配形状 (H, W, Y) 或其它排列
                    if stability_map.shape == (H, W, Y):
                        aligned = stability_map
                    elif stability_map.shape == (Y, H, W):
                        aligned = np.transpose(stability_map, (1, 2, 0))
                    elif stability_map.shape == (W, H, Y):
                        aligned = np.transpose(stability_map, (1, 0, 2))
                    elif stability_map.shape == (H, Y, W):
                        aligned = np.transpose(stability_map, (0, 2, 1))
                    else:
                        # 无法直接匹配，尝试 resize
                        aligned = np.resize(stability_map, (H, W, Y))
                elif stability_map.ndim == 2:
                    # (H, W) -> 添加 yaw 维度
                    aligned = stability_map[..., np.newaxis]
                    if aligned.shape[2] != Y:
                        aligned = np.resize(aligned, (H, W, Y))
                elif stability_map.ndim == 1:
                    # 尝试重塑为 (H, W, Y)
                    try:
                        aligned = stability_map.reshape((H, W, -1))
                        if aligned.shape[2] != Y:
                            aligned = np.resize(aligned, (H, W, Y))
                    except Exception:
                        aligned = np.resize(stability_map, (H, W, Y))
                else:
                    aligned = np.resize(stability_map, (H, W, Y))

                stability_map = np.array(aligned, dtype=np.float32)

                # tipping 概率 = 1 - stability（保留 yaw 维度用于内部缓存）
                tipping_map = 1.0 - stability_map
                tipping_map = np.clip(tipping_map, 0.0, 1.0)

                # 对外发布的 OccupancyGrid 必须是二维 (H, W) 的展开数据。
                # 使用航向维度上的最大 tipping（保守策略）作为每个格子的倾覆概率。
                per_cell_tipping = tipping_map.max(axis=2)
                per_cell_tipping = np.clip(per_cell_tipping, 0.0, 1.0)

                # 映射到 0-100 的整数占用值（100 表示占用）
                occ_vals_2d = (per_cell_tipping * 100.0).round().astype(np.int8)

                # 确保形状为 (H, W)
                if occ_vals_2d.shape != (H, W):
                    occ_vals_2d = np.resize(occ_vals_2d, (H, W))

                # 同时构造并发布一个三维 HWY 的消息（Float32MultiArray），供期望三维输入的接收方使用
                try:
                    from std_msgs.msg import Float32MultiArray, MultiArrayDimension
                    fam = Float32MultiArray()
                    # 填充数据（浮点，0..1 表示 tipping 概率）
                    fam.data = tipping_map.flatten().astype(np.float32).tolist()
                    # 布局: dim[0]=height, dim[1]=width, dim[2]=yaw
                    dim_h = MultiArrayDimension()
                    dim_h.label = 'height'
                    dim_h.size = int(H)
                    dim_h.stride = int(W * Y)
                    dim_w = MultiArrayDimension()
                    dim_w.label = 'width'
                    dim_w.size = int(W)
                    dim_w.stride = int(Y)
                    dim_y = MultiArrayDimension()
                    dim_y.label = 'yaw'
                    dim_y.size = int(Y)
                    dim_y.stride = 1
                    fam.layout.dim = [dim_h, dim_w, dim_y]
                    fam.layout.data_offset = 0

                    # 发布（非阻塞）
                    try:
                        self.occ3d_pub.publish(fam)
                    except Exception as e:
                        rospy.logwarn(f"Failed to publish 3D HWY occupancy array: {e}")
                except Exception as e:
                    rospy.logwarn(f"Failed to build/publish 3D HWY occupancy array: {e}")

                # 发布二维 OccupancyGrid 以兼容仅接受 2D 的节点
                occ.data = occ_vals_2d.flatten().tolist()

                return occ
            except Exception as e:
                rospy.logwarn("Failed to compute occupancy data from yaw stability map: %s", str(e))

        # 若无法基于法向量计算占用，保留默认未知(-1)或之前填充的数据
        occ.data = data.flatten().tolist()
        return occ

if __name__ == '__main__':
    try:
        generator = TerrainDatasetGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        pass