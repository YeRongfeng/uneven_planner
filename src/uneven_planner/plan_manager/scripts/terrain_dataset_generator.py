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

# import sys
# # 1. 强制清理掉所有可能干扰的系统 3.8 路径
# sys.path = [p for p in sys.path if "python3.8" not in p and "dist-packages" not in p]

# # 2. 手动添加你刚刚安装了 open3d 的 3.10 路径（请确保路径与第一步输出一致）
# vim_site_packages = "/home/yrf/miniconda3/envs/vim/lib/python3.10/site-packages"
# if vim_site_packages not in sys.path:
#     sys.path.insert(0, vim_site_packages)

# # 3. 手动保留 ROS 的核心路径（为了能 import rospy）
# ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
# if ros_path not in sys.path:
#     sys.path.append(ros_path)


import rospy
from std_srvs.srv import Trigger
from plan_manager.srv import (
    PointCloudToGrid,
    PointCloudToGridRequest,
    UpdateTerrainMap,
    UpdateTerrainMapRequest,
)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，避免线程问题
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 添加3D绘图支持
import open3d as o3d
try:
    import noise
except ImportError:
    # External-map dataset generation does not use Perlin noise.  Keep that
    # workflow available on ROS/Python installations where the optional
    # ``noise`` package is not installed.
    noise = None
import random
import math
import time
import os
import sys
import tempfile
import json
import datetime
import pickle
import glob
from types import SimpleNamespace
import cv2
import tf.transformations as tf_trans
from scipy.ndimage import (
    gaussian_filter,
    zoom,
    map_coordinates,
    label as connected_component_label,
)
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.special import comb

from stability_validation import (
    build_periodic_signed_stability_esdf,
    validate_trajectory_stability,
)
from sample_laz_mother_map import (
    build_surface,
    load_surface_points,
    measure_above_surface_coverage,
    measure_classified_above_surface_coverage,
)
from terrain_map_quality import evaluate as evaluate_terrain_map
from review_slots import existing_paths_belong_to_canonical

# ROS消息类型
from geometry_msgs.msg import PoseStamped
from mpc_controller.msg import SE2Traj
from std_msgs.msg import Bool


def parse_las_class_values(value, parameter_name):
    """Parse one or more LAS classification codes from ROS parameters."""
    if isinstance(value, str):
        values = value.replace(';', ',').split(',')
    elif np.isscalar(value):
        values = (value,)
    else:
        values = value
    result = tuple(int(item) for item in values if str(item).strip())
    if not result:
        raise ValueError(f"{parameter_name} must contain at least one class code")
    if any(item < 0 for item in result):
        raise ValueError(f"{parameter_name} must contain non-negative class codes")
    return result
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

    normal_z = torch.abs(normal_z)

    device = normal_x.device
    H, W = normal_x.shape
    
    # 地形约束参数
    # h = 8.0         # 机器人高度
    # min_edge = 10.0 # 最小边长约束
    # max_edge = 20.0 # 最大边长约束
    # h = 35.0         # 机器人高度
    # min_edge = 8.0 # 最小边长约束
    # max_edge = 15.0 # 最大边长约束
    h = 15.0         # 机器人高度
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
    
    # 处理部分可达区域
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
        complex_b = b_vals[mask_complex]
        complex_nx = normal_x[mask_complex]
        complex_ny = normal_y[mask_complex]
        complex_nz = normal_z[mask_complex]
        
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

    # C++ converter 以 grid_y * width + grid_x 写入，Python reshape 后已经
    # 是 (row=y, col=x, yaw)。这里不能再转置 H/W；方形地图会让
    # shape 检查无法发现该错误。

    return yaw_stability

# def compute_map_yaw_bins(normal_x, normal_y, normal_z, yaw_bins=18):
#     """
#     计算地图上每个点的分箱角度是否会倾覆（适配RXS2空间映射）
    
#     基于倾覆安全性方法计算可达性，然后将结果从倾覆安全性SE(2)空间
#     转换到RXS2 SE(2)空间，以适配当前项目的位姿表示方法。
    
#     两种SE(2)表示虽然描述同一SE(3)位姿，但yaw角度定义不同：
#     - 倾覆安全性方法：标准水平面yaw角度
#     - RXS2方法：考虑地形contact constraint的yaw角度
    
#     Args:
#         normal_x: 地形法向量x分量 (H, W) 
#         normal_y: 地形法向量y分量 (H, W)
#         normal_z: 地形法向量z分量 (H, W)
#         yaw_bins: 朝向角度的分箱数量，默认为18

#     Returns:
#         torch.Tensor: 每个点每个角度分箱的倾覆状态，形状为(H, W, yaw_bins)，
#                      对应RXS2空间的角度分箱，1表示不会倾覆，0表示会倾覆
#     """
#     # 确保输入是torch张量
#     if not isinstance(normal_x, torch.Tensor):
#         normal_x = torch.tensor(normal_x, dtype=torch.float32)
#         normal_y = torch.tensor(normal_y, dtype=torch.float32) 
#         normal_z = torch.tensor(normal_z, dtype=torch.float32)
    
#     device = normal_x.device
#     H, W = normal_x.shape
    
#     # ========== 第一步：使用倾覆安全性方法计算可达性 ==========
    
#     # 地形约束参数
#     h = 35.0         # 机器人高度
#     min_edge = 8.0 # 最小边长约束
#     max_edge = 15.0 # 最大边长约束
    
#     # 计算地形倾斜角度对应的tan值
#     terrain_slope = torch.arctan2(torch.sqrt(normal_x**2 + normal_y**2), torch.abs(normal_z))
    
#     # 计算约束参数b
#     b_vals = h * torch.tan(terrain_slope)
    
#     # 根据b值分类地形约束类型
#     mask_reachable = b_vals < min_edge  # 完全可达
#     mask_partial = (b_vals >= min_edge) & (b_vals < max_edge)  # 部分可达
#     mask_complex = (b_vals >= max_edge) & (b_vals < torch.sqrt(torch.tensor(max_edge**2 + min_edge**2, device=device)))  # 复杂约束
#     mask_unreachable = b_vals >= torch.sqrt(torch.tensor(max_edge**2 + min_edge**2, device=device))  # 完全不可达
    
#     # 定义倾覆安全性方法的角度分箱：从-π到π均匀分布
#     safety_bin_angles = torch.linspace(-torch.pi, torch.pi, yaw_bins + 1, device=device)[:-1]  # 去掉最后一个
    
#     # 初始化倾覆安全性结果数组
#     safety_yaw_stability = torch.ones((H, W, yaw_bins), dtype=torch.float32, device=device)
    
#     def safe_arctan_transform(nz_vals, theta_local_vals):
#         """安全的arctan变换，考虑角度的正确象限"""
#         tan_vals = torch.tan(theta_local_vals)
#         arctan_result = torch.arctan(nz_vals * tan_vals)
        
#         # 如果原始角度在第二或第三象限（cos < 0），需要调整arctan结果
#         cos_local = torch.cos(theta_local_vals)
#         sin_local = torch.sin(theta_local_vals)
        
#         # 调整第二象限的角度：arctan结果需要加π
#         second_quadrant = (cos_local < 0) & (sin_local > 0)
#         arctan_result = torch.where(second_quadrant, arctan_result + torch.pi, arctan_result)
        
#         # 调整第三象限的角度：arctan结果需要减π
#         third_quadrant = (cos_local < 0) & (sin_local < 0)
#         arctan_result = torch.where(third_quadrant, arctan_result - torch.pi, arctan_result)

#         # arctan_result = normalize_angle(torch.pi * 0.5 - arctan_result)  # 作 theta=pi/2-arctan_result映射，并确保结果在[-π, π]范围内

#         return arctan_result

#     def normalize_angle(angle):
#         """标准化角度到[-π, π]"""
#         angle = torch.where(angle > torch.pi, angle - 2*torch.pi, angle)
#         angle = torch.where(angle < -torch.pi, angle + 2*torch.pi, angle)
#         return angle
    
#     def check_angle_in_range_vectorized(angles, starts, ends):
#         """向量化检查角度是否在范围内"""
#         # angles: (yaw_bins,), starts: (N,), ends: (N,)
#         # 返回: (N, yaw_bins) 布尔张量
#         angles = angles.unsqueeze(0)  # (1, yaw_bins)
#         starts = starts.unsqueeze(1)  # (N, 1)
#         ends = ends.unsqueeze(1)      # (N, 1)
        
#         angles = normalize_angle(angles)
#         starts = normalize_angle(starts)
#         ends = normalize_angle(ends)
        
#         # 正常情况：start <= end
#         normal_case = starts <= ends
#         in_range_normal = (angles >= starts) & (angles <= ends) & normal_case
        
#         # 跨越边界情况：start > end
#         cross_boundary = starts > ends
#         in_range_cross = ((angles >= starts) | (angles <= ends)) & cross_boundary
        
#         return in_range_normal | in_range_cross
    
#     # 处理完全不可达区域：所有角度都标记为会倾覆(0)
#     safety_yaw_stability[mask_unreachable] = 0.0
    
#     # 处理部分可达区域 - 向量化处理
#     if torch.any(mask_partial):
#         # 获取部分可达区域的坐标和值
#         partial_indices = torch.where(mask_partial)
#         partial_b = b_vals[mask_partial]
#         partial_nx = normal_x[mask_partial]
#         partial_ny = normal_y[mask_partial]
#         partial_nz = normal_z[mask_partial]
        
#         # 批量计算约束边界角度（局部坐标系）
#         s1_vals = torch.arcsin(min_edge / partial_b)
#         e1_vals = torch.pi - s1_vals
#         s2_vals = -s1_vals
#         e2_vals = -torch.pi + s1_vals
        
#         # 考虑地形法向量的影响：从局部坐标系转换到全局坐标系
#         normal_proj_angles = torch.arctan2(partial_ny, partial_nx)
        
#         # 计算全局坐标系下的边界参数
#         s1_transforms = safe_arctan_transform(partial_nz, s1_vals)
#         e1_transforms = safe_arctan_transform(partial_nz, e1_vals)
#         s2_transforms = safe_arctan_transform(partial_nz, s2_vals)
#         e2_transforms = safe_arctan_transform(partial_nz, e2_vals)
        
#         s1_globals = normalize_angle(normal_proj_angles + s1_transforms)
#         e1_globals = normalize_angle(normal_proj_angles + e1_transforms)
#         s2_globals = normalize_angle(normal_proj_angles + s2_transforms)
#         e2_globals = normalize_angle(normal_proj_angles + e2_transforms)
        
#         # 向量化检查每个角度分箱是否在不可达区域
#         in_unreachable_region1 = check_angle_in_range_vectorized(safety_bin_angles, s1_globals, e1_globals)  # (N, yaw_bins)
#         in_unreachable_region2 = check_angle_in_range_vectorized(safety_bin_angles, e2_globals, s2_globals)  # (N, yaw_bins)
        
#         unreachable_mask = in_unreachable_region1 | in_unreachable_region2  # (N, yaw_bins)
        
#         # 更新safety_yaw_stability
#         for idx, (i, j) in enumerate(zip(partial_indices[0], partial_indices[1])):
#             safety_yaw_stability[i, j, unreachable_mask[idx]] = 0.0
    
#     # 处理复杂约束区域 - 向量化处理
#     if torch.any(mask_complex):
#         # 获取复杂约束区域的坐标和值
#         complex_indices = torch.where(mask_complex)
#         complex_b = b_vals[complex_indices]
#         complex_nx = normal_x[complex_indices]
#         complex_ny = normal_y[complex_indices]
#         complex_nz = normal_z[complex_indices]
        
#         # 批量计算复杂约束的边界参数
#         r1_vals = torch.arcsin(min_edge / complex_b)
#         r2_vals = torch.arccos(max_edge / complex_b)
        
#         # 计算所有边界角度（局部坐标系）
#         s1_vals = -r2_vals
#         e1_vals = r2_vals
#         s2_vals = r1_vals
#         e2_vals = torch.pi - r1_vals
#         p1_vals = torch.pi - r2_vals
#         p2_vals = -torch.pi + r2_vals
#         s3_vals = -torch.pi + r1_vals
#         e3_vals = -r1_vals
        
#         # 考虑地形法向量的影响：从局部坐标系转换到全局坐标系
#         normal_proj_angles = torch.arctan2(complex_ny, complex_nx)
        
#         # 计算全局坐标系下的边界参数
#         s1_transforms = safe_arctan_transform(complex_nz, s1_vals)
#         e1_transforms = safe_arctan_transform(complex_nz, e1_vals)
#         s2_transforms = safe_arctan_transform(complex_nz, s2_vals)
#         e2_transforms = safe_arctan_transform(complex_nz, e2_vals)
#         p1_transforms = safe_arctan_transform(complex_nz, p1_vals)
#         p2_transforms = safe_arctan_transform(complex_nz, p2_vals)
#         s3_transforms = safe_arctan_transform(complex_nz, s3_vals)
#         e3_transforms = safe_arctan_transform(complex_nz, e3_vals)
        
#         s1_globals = normalize_angle(normal_proj_angles + s1_transforms)
#         e1_globals = normalize_angle(normal_proj_angles + e1_transforms)
#         s2_globals = normalize_angle(normal_proj_angles + s2_transforms)
#         e2_globals = normalize_angle(normal_proj_angles + e2_transforms)
#         p1_globals = normalize_angle(normal_proj_angles + p1_transforms)
#         p2_globals = normalize_angle(normal_proj_angles + p2_transforms)
#         s3_globals = normalize_angle(normal_proj_angles + s3_transforms)
#         e3_globals = normalize_angle(normal_proj_angles + e3_transforms)
        
#         # 向量化检查每个角度分箱是否在不可达区域
#         in_unreachable1 = check_angle_in_range_vectorized(safety_bin_angles, s1_globals, e1_globals)
#         in_unreachable2 = check_angle_in_range_vectorized(safety_bin_angles, s2_globals, e2_globals)
#         in_unreachable3 = check_angle_in_range_vectorized(safety_bin_angles, s3_globals, e3_globals)
        
#         # 处理p1和p2边界（单侧边界）
#         safety_bin_angles_expanded = safety_bin_angles.unsqueeze(0)  # (1, yaw_bins)
#         p1_expanded = p1_globals.unsqueeze(1)  # (N, 1)
#         p2_expanded = p2_globals.unsqueeze(1)  # (N, 1)
        
#         in_unreachable_p1 = safety_bin_angles_expanded > p1_expanded
#         in_unreachable_p2 = safety_bin_angles_expanded < p2_expanded
        
#         unreachable_mask = (in_unreachable1 | in_unreachable2 | in_unreachable3 | 
#                            in_unreachable_p1 | in_unreachable_p2)  # (N, yaw_bins)
        
#         # 更新safety_yaw_stability
#         for idx, (i, j) in enumerate(zip(complex_indices[0], complex_indices[1])):
#             safety_yaw_stability[i, j, unreachable_mask[idx]] = 0.0

#     # ========== 第二步：从倾覆安全性SE(2)转换到RXS2 SE(2) ==========
    
#     def normalize_angle(angle):
#         """标准化角度到[-π, π]"""
#         angle = torch.where(angle > torch.pi, angle - 2*torch.pi, angle)
#         angle = torch.where(angle < -torch.pi, angle + 2*torch.pi, angle)
#         return angle
    
#     def se2_safety_to_rxs2_yaw(yaw_safety, nx, ny, nz):
#         """
#         将倾覆安全性SE(2)的yaw角度转换为RXS2 SE(2)的yaw角度
        
#         基于地形contact constraint的转换关系：
#         - 倾覆安全性方法：标准水平面yaw
#         - RXS2方法：考虑地形法向量的yaw
        
#         转换公式基于Sherman-Morrison公式的逆变换思想
#         """
#         # 计算地形法向量在XY平面的投影
#         normal_magnitude = torch.sqrt(nx**2 + ny**2 + nz**2)
#         normalized_nx = nx / (normal_magnitude + 1e-8)
#         normalized_ny = ny / (normal_magnitude + 1e-8) 
#         normalized_nz = torch.abs(nz) / (normal_magnitude + 1e-8)
        
#         # 计算RXS2中的zb向量（地形倾斜在XY平面的投影）
#         zb_x = normalized_nx / (normalized_nz + 1e-8)
#         zb_y = normalized_ny / (normalized_nz + 1e-8)
        
#         # 计算地形法向量在XY平面的投影角度（这是两种SE(2)表示的主要差异）
#         terrain_angle = torch.arctan2(zb_y, zb_x)
        
#         # SE(2)空间转换：yaw_rxs2 = yaw_safety - terrain_correction
#         # 这里的符号根据实际的坐标系定义可能需要调整
#         yaw_rxs2 = yaw_safety - terrain_angle
        
#         return normalize_angle(yaw_rxs2)
    
#     # 定义RXS2空间的角度分箱
#     rxs2_bin_angles = torch.linspace(-torch.pi, torch.pi, yaw_bins + 1, device=device)[:-1]
    
#     # 初始化RXS2空间的结果数组
#     rxs2_yaw_stability = torch.zeros((H, W, yaw_bins), dtype=torch.float32, device=device)
    
#     # 对每个位置进行SE(2)空间转换
#     for i in range(H):
#         for j in range(W):
#             local_nx = normal_x[i, j]
#             local_ny = normal_y[i, j]
#             local_nz = normal_z[i, j]
            
#             # 对每个倾覆安全性角度分箱进行转换
#             for k in range(yaw_bins):
#                 if safety_yaw_stability[i, j, k] > 0.5:  # 如果在倾覆安全性方法中是可达的
#                     safety_yaw = safety_bin_angles[k]
                    
#                     # 转换到RXS2空间的yaw角度
#                     rxs2_yaw = se2_safety_to_rxs2_yaw(safety_yaw, local_nx, local_ny, local_nz)
                    
#                     # 找到RXS2空间中最接近的角度分箱
#                     angle_diffs = torch.abs(normalize_angle(rxs2_bin_angles - rxs2_yaw))
#                     closest_bin = torch.argmin(angle_diffs)
                    
#                     # 在RXS2空间中标记为可达
#                     rxs2_yaw_stability[i, j, closest_bin] = 1.0

#     # return rxs2_yaw_stability
#     return safety_yaw_stability



class TerrainGenerator:
    """
    地形生成器：基于generator.py的地形生成逻辑
    """
    
    def __init__(self, map_size=10.0, resolution=0.02, max_height=3.0, min_height=0.0):
        if noise is None:
            raise ImportError(
                "The optional 'noise' package is required for synthetic "
                "terrain generation, but not for external-map datasets")
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
    地图转换器：使用C++服务将点云转换为栅格地图
    调用pointcloud_to_grid_converter节点提供的服务
    """

    def __init__(self, coarse_resolution=0.2, fine_resolution=0.2, voxel_size=0.1):
        """
        初始化GridTransformer
        
        Args:
            coarse_resolution: 点云聚合分辨率（0.2m）
            fine_resolution: 输出栅格分辨率（0.2m，与粗格相同）
            voxel_size: 体素降采样大小（0.1m）
        """
        self.coarse_resolution = coarse_resolution
        self.fine_resolution = fine_resolution
        self.voxel_size = voxel_size

        # 栅格地图数据
        self.elevation_map = None
        self.normal_x_map = None
        self.normal_y_map = None
        self.normal_z_map = None

        # 地图元信息
        self.map_bounds = None
        self.map_center = None
        self.map_size = None

        # 等待C++服务可用
        rospy.loginfo("Waiting for pointcloud_to_grid service...")
        try:
            # 使用相对名称，使多个数据生成 worker 可以在独立命名空间运行。
            rospy.wait_for_service('pointcloud_to_grid', timeout=10.0)
            self.grid_service = rospy.ServiceProxy(
                'pointcloud_to_grid', PointCloudToGrid)
            rospy.loginfo("GridTransformer initialized with C++ service")
        except rospy.ROSException:
            rospy.logwarn("C++ service not available, using fallback Python implementation")
            self.grid_service = None

    def pointcloud_to_ros_message(self, pcd):
        """将Open3D点云转换为ROS PointCloud2消息"""
        import sensor_msgs.point_cloud2 as pc2
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header
        
        points = np.asarray(pcd.points)
        
        # 创建PointCloud2消息
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        
        # 定义点云字段
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)
        ]
        
        # 添加强度值（全部设为1.0）
        points_with_intensity = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
        
        # 创建PointCloud2消息
        cloud_msg = pc2.create_cloud(header, fields, points_with_intensity)
        
        return cloud_msg

    def calculate_map_bounds(self, pcd):
        """计算地图边界"""
        points = np.asarray(pcd.points)
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
        将点云转换为栅格地图（使用C++服务）
        
        Args:
            pcd: open3d.geometry.PointCloud
            map_bounds: 可选的地图边界 (min_x, max_x, min_y, max_y)
            
        Returns:
            dict: 包含栅格地图数据的字典
        """
        rospy.loginfo("Starting point cloud to grid transformation using C++ service...")

        # 计算或使用指定的地图边界
        if map_bounds is None:
            bounds = self.calculate_map_bounds(pcd)
        else:
            bounds = map_bounds
            
        self.map_bounds = bounds
        self.map_center = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2)

        # 将点云转换为ROS消息
        cloud_msg = self.pointcloud_to_ros_message(pcd)

        # 调用C++服务进行转换
        try:
            req = PointCloudToGridRequest()
            req.pointcloud = cloud_msg
            req.map_min_x = bounds[0]
            req.map_max_x = bounds[1]
            req.map_min_y = bounds[2]
            req.map_max_y = bounds[3]
            
            rospy.loginfo(f"Calling C++ grid conversion service for bounds: {bounds}")
            response = self.grid_service(req)
            
            if not response.success:
                raise Exception(f"C++ service failed: {response.message}")
                
            rospy.loginfo(f"C++ conversion successful: {response.grid_width}x{response.grid_height} grid at {response.resolution}m resolution")
            
            # C++已经返回精细分辨率的栅格，直接使用
            fine_elevation = np.array(response.elevation_grid).reshape(response.grid_height, response.grid_width)
            fine_normal_x = np.array(response.normal_x_grid).reshape(response.grid_height, response.grid_width)
            fine_normal_y = np.array(response.normal_y_grid).reshape(response.grid_height, response.grid_width)
            fine_normal_z = np.array(response.normal_z_grid).reshape(response.grid_height, response.grid_width)
            
        except Exception as e:
            rospy.logerr(f"C++ service call failed: {e}")
            raise

        # C++已经输出精细分辨率，不需要Python端插值
        # 存储结果
        self.elevation_map = fine_elevation
        self.normal_x_map = fine_normal_x
        self.normal_y_map = fine_normal_y
        self.normal_z_map = fine_normal_z
        self.map_size = (fine_elevation.shape[1], fine_elevation.shape[0])

        result = {
            'elevation': self.elevation_map,
            'normal_x': self.normal_x_map,
            'normal_y': self.normal_y_map,
            'normal_z': self.normal_z_map,
            'bounds': bounds,
            'resolution': response.resolution,  # 使用C++返回的实际分辨率
            'center': self.map_center,
            'size': self.map_size
        }

        rospy.loginfo("Point cloud to grid transformation completed!")
        return result

    def interpolate_grid(self, coarse_grids, bounds, target_resolution):
        """
        将粗糙栅格插值到精细分辨率，完全匹配grid_transformer.cpp的插值逻辑
        
        C++逻辑：
        - elevation: INTER_CUBIC (三次插值)
        - normal_x/y/z: INTER_NEAREST (最近邻插值)
        """
        elevation_coarse, normal_x_coarse, normal_y_coarse, normal_z_coarse = coarse_grids
        min_x, max_x, min_y, max_y = bounds

        target_width = int(np.ceil((max_x - min_x) / target_resolution))
        target_height = int(np.ceil((max_y - min_y) / target_resolution))

        coarse_height, coarse_width = elevation_coarse.shape

        x_coarse = np.linspace(min_x, max_x, coarse_width)
        y_coarse = np.linspace(min_y, max_y, coarse_height)

        x_fine = np.linspace(min_x, max_x, target_width)
        y_fine = np.linspace(min_y, max_y, target_height)

        rospy.loginfo("Performing grid interpolation with C++ logic...")

        # 高程使用三次插值（匹配INTER_CUBIC）
        valid_mask = ~np.isnan(elevation_coarse)
        if np.sum(valid_mask) > 3:  # 需要至少4个点进行三次插值
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

        # # 法向量使用双线性插值以获得更平滑的结果（而不是最近邻）
        # # 虽然C++使用INTER_NEAREST，但为了获得更平滑的效果，我们尝试双线性插值
        # normal_x_fine = self._interpolate_bilinear_cpp_style(normal_x_coarse, x_coarse, y_coarse, x_fine, y_fine)
        # normal_y_fine = self._interpolate_bilinear_cpp_style(normal_y_coarse, x_coarse, y_coarse, x_fine, y_fine)
        # normal_z_fine = self._interpolate_bilinear_cpp_style(normal_z_coarse, x_coarse, y_coarse, x_fine, y_fine)
        
        # 法向量使用最近邻插值
        normal_x_fine = self._interpolate_nearest(normal_x_coarse, x_coarse, y_coarse, x_fine, y_fine)
        normal_y_fine = self._interpolate_nearest(normal_y_coarse, x_coarse, y_coarse, x_fine, y_fine)
        normal_z_fine = self._interpolate_nearest(normal_z_coarse, x_coarse, y_coarse, x_fine, y_fine)

        rospy.loginfo("Grid interpolation completed with C++ logic")
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

    def _interpolate_normals_cpp_style(self, normal_x_coarse, normal_y_coarse, normal_z_coarse, 
                                       x_coarse, y_coarse, x_fine, y_fine):
        """
        法向量插值，完全匹配C++的特殊逻辑：
        在3x3邻域中选择最小normal_z值对应的法向量
        
        这对应C++代码中的以下逻辑：
        1. 初始化xm, ym, zm为当前位置的法向量
        2. 在3x3邻域(pos-0.4到pos+0.4，步长0.4)中遍历
        3. 如果发现更小的normal_z，则选择该位置的全部法向量
        """
        coarse_height, coarse_width = normal_z_coarse.shape
        target_height, target_width = len(y_fine), len(x_fine)
        
        normal_x_fine = np.full((target_height, target_width), np.nan)
        normal_y_fine = np.full((target_height, target_width), np.nan) 
        normal_z_fine = np.full((target_height, target_width), np.nan)
        
        # 记录最后有效的数据（匹配C++的last_x, last_y, last_z逻辑）
        last_x, last_y, last_z = np.nan, np.nan, np.nan
        
        for i in range(target_height):
            for j in range(target_width):
                # 找到当前精细格点对应的粗糙格点
                yi_idx = np.argmin(np.abs(y_coarse - y_fine[i]))
                xi_idx = np.argmin(np.abs(x_coarse - x_fine[j]))
                
                # 检查是否在边界内
                if not (0 <= yi_idx < coarse_height and 0 <= xi_idx < coarse_width):
                    # 超出边界，使用最后有效值
                    normal_x_fine[i, j] = last_x
                    normal_y_fine[i, j] = last_y
                    normal_z_fine[i, j] = last_z
                    continue
                
                # 初始化为当前位置的法向量值
                xm = normal_x_coarse[yi_idx, xi_idx]
                ym = normal_y_coarse[yi_idx, xi_idx]
                zm = normal_z_coarse[yi_idx, xi_idx]
                
                # 检查是否为NaN
                if np.isnan(zm):
                    normal_x_fine[i, j] = last_x
                    normal_y_fine[i, j] = last_y
                    normal_z_fine[i, j] = last_z
                    continue
                
                # 在3x3邻域中搜索最小normal_z
                # C++逻辑：从pos-0.4开始，步长0.4，共3步（对应i=1,2,3和j=1,2,3）
                for di in range(-1, 2):  # -1, 0, 1 对应C++的i=1,2,3
                    for dj in range(-1, 2):  # -1, 0, 1 对应C++的j=1,2,3
                        neighbor_yi = yi_idx + di
                        neighbor_xi = xi_idx + dj
                        
                        # 检查邻域点是否在边界内（对应C++的safeCheck）
                        if (0 <= neighbor_yi < coarse_height and 0 <= neighbor_xi < coarse_width):
                            neighbor_nz = normal_z_coarse[neighbor_yi, neighbor_xi]
                            
                            # 如果找到更小的normal_z，选择该位置的全部法向量
                            if not np.isnan(neighbor_nz) and neighbor_nz < zm:
                                zm = neighbor_nz
                                xm = normal_x_coarse[neighbor_yi, neighbor_xi]
                                ym = normal_y_coarse[neighbor_yi, neighbor_xi]
                
                # 存储结果
                normal_x_fine[i, j] = xm
                normal_y_fine[i, j] = ym
                normal_z_fine[i, j] = zm
                
                # 更新最后有效值
                if not np.isnan(zm):
                    last_x, last_y, last_z = xm, ym, zm
        
        return normal_x_fine, normal_y_fine, normal_z_fine

    def _interpolate_bilinear_cpp_style(self, data, x_coarse, y_coarse, x_fine, y_fine):
        """
        双线性插值，用于获得更平滑的法向量
        
        虽然C++使用INTER_NEAREST，但话题数据显示更平滑的效果，
        所以我们使用双线性插值来匹配这种平滑性
        """
        from scipy.interpolate import griddata
        
        coarse_height, coarse_width = data.shape
        
        # 创建粗糙网格的坐标点
        xi_coarse, yi_coarse = np.meshgrid(x_coarse, y_coarse, indexing='xy')
        
        # 过滤有效点（非NaN）
        valid_mask = ~np.isnan(data)
        if np.sum(valid_mask) == 0:
            return np.full((len(y_fine), len(x_fine)), np.nan)
        
        points = np.column_stack([xi_coarse[valid_mask], yi_coarse[valid_mask]])
        values = data[valid_mask]
        
        # 创建精细网格的坐标点
        xi_fine, yi_fine = np.meshgrid(x_fine, y_fine, indexing='xy')
        
        # 使用线性插值获得更平滑的结果
        try:
            result = griddata(points, values, (xi_fine, yi_fine), method='linear', fill_value=np.nan)
            
            # 对于线性插值无法覆盖的区域，使用最近邻填充
            nan_mask = np.isnan(result)
            if np.any(nan_mask):
                nearest_result = griddata(points, values, (xi_fine, yi_fine), method='nearest', fill_value=np.nan)
                result[nan_mask] = nearest_result[nan_mask]
                
        except Exception as e:
            rospy.logwarn(f"Bilinear interpolation failed: {e}, falling back to nearest neighbor")
            result = griddata(points, values, (xi_fine, yi_fine), method='nearest', fill_value=np.nan)
        
        return result

    def _interpolate_nearest_cpp_style(self, data, x_coarse, y_coarse, x_fine, y_fine):
        """
        最近邻插值，完全模拟C++的grid_map::InterpolationMethods::INTER_NEAREST行为
        
        关键差异分析：
        1. C++使用atPosition方法，基于连续坐标位置
        2. 需要确保坐标系统完全一致
        3. 边界处理必须匹配C++的last data逻辑
        """
        from scipy.interpolate import griddata
        
        # 使用scipy的griddata进行最近邻插值，更接近C++的atPosition行为
        coarse_height, coarse_width = data.shape
        
        # 创建粗糙网格的坐标点
        xi_coarse, yi_coarse = np.meshgrid(x_coarse, y_coarse, indexing='xy')
        
        # 过滤有效点（非NaN）
        valid_mask = ~np.isnan(data)
        if np.sum(valid_mask) == 0:
            return np.full((len(y_fine), len(x_fine)), np.nan)
        
        points = np.column_stack([xi_coarse[valid_mask], yi_coarse[valid_mask]])
        values = data[valid_mask]
        
        # 创建精细网格的坐标点
        xi_fine, yi_fine = np.meshgrid(x_fine, y_fine, indexing='xy')
        
        # 使用nearest插值
        try:
            result = griddata(points, values, (xi_fine, yi_fine), method='nearest', fill_value=np.nan)
        except Exception as e:
            rospy.logwarn(f"Griddata interpolation failed: {e}, falling back to manual method")
            # 降级到手动最近邻
            result = self._manual_nearest_neighbor(data, x_coarse, y_coarse, x_fine, y_fine)
        
        return result
    
    def _manual_nearest_neighbor(self, data, x_coarse, y_coarse, x_fine, y_fine):
        """手动最近邻插值，作为scipy方法的备选"""
        coarse_height, coarse_width = data.shape
        target_height, target_width = len(y_fine), len(x_fine)

        result = np.full((target_height, target_width), np.nan)
        last_value = np.nan

        for i in range(target_height):
            for j in range(target_width):
                # 找到最近的粗糙栅格点
                yi_idx = np.argmin(np.abs(y_coarse - y_fine[i]))
                xi_idx = np.argmin(np.abs(x_coarse - x_fine[j]))
                
                # 检查是否在边界内
                if (0 <= yi_idx < coarse_height and 0 <= xi_idx < coarse_width):
                    value = data[yi_idx, xi_idx]
                    if not np.isnan(value):
                        result[i, j] = value
                        last_value = value
                    else:
                        result[i, j] = last_value
                else:
                    result[i, j] = last_value

        return result


    def resize_map(self, map_array, target_size):
        """
        调整地图尺寸到目标大小（使用缩放插值）
        
        Args:
            map_array: 输入地图数组 (H, W)
            target_size: 目标尺寸 (整数)
            
        Returns:
            调整大小后的地图数组 (target_size, target_size)
        """
        try:
            from scipy.ndimage import zoom
            current_size = map_array.shape[0]
            scale_factor = target_size / current_size
            resized = zoom(map_array, scale_factor, order=1, prefilter=False)
            
            # 确保精确尺寸
            if resized.shape[0] != target_size or resized.shape[1] != target_size:
                final_map = np.zeros((target_size, target_size), dtype=np.float32)
                copy_size = min(resized.shape[0], target_size)
                final_map[:copy_size, :copy_size] = resized[:copy_size, :copy_size]
                resized = final_map
                
            return resized
            
        except ImportError:
            # 简单重采样
            current_size = map_array.shape[0]
            step = current_size // target_size
            if step > 1:
                return map_array[::step, ::step][:target_size, :target_size]
            else:
                # 填充
                result = np.zeros((target_size, target_size), dtype=np.float32)
                result[:current_size, :current_size] = map_array
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
        self.stop_after_train = bool(
            rospy.get_param('~stop_after_train', False))
        self.dataset_dir = rospy.get_param('~dataset_dir', 'dataset')
        self.start_env_id = rospy.get_param('~start_env_id', 0)
        self.start_phase = str(rospy.get_param(
            '~start_phase', 'train')).strip().lower()
        if self.start_phase not in ('train', 'val'):
            raise ValueError("start_phase must be 'train' or 'val'")
        
        # 外部地图参数
        self.use_external_map = rospy.get_param('~use_external_map', False)
        self.external_map_path = rospy.get_param('~external_map_path', '')
        configured_map_paths = rospy.get_param('~external_map_paths', [])
        if isinstance(configured_map_paths, str):
            configured_map_paths = configured_map_paths.replace(';', ',').split(',')
        elif not isinstance(configured_map_paths, (list, tuple)):
            raise ValueError("external_map_paths must be a list or comma-separated string")
        self.external_map_paths = []
        for map_path in configured_map_paths:
            map_path = str(map_path).strip()
            if map_path and map_path not in self.external_map_paths:
                self.external_map_paths.append(map_path)
        if not self.external_map_paths and self.external_map_path:
            self.external_map_paths = [self.external_map_path]
        if self.external_map_paths:
            self.external_map_path = self.external_map_paths[0]
        self.train_external_map_paths = self._parse_map_paths(
            rospy.get_param('~train_external_map_paths', []))
        self.val_external_map_paths = self._parse_map_paths(
            rospy.get_param('~val_external_map_paths', []))
        self.has_split_map_pools = bool(
            self.train_external_map_paths and self.val_external_map_paths)
        if self.train_external_map_paths or self.val_external_map_paths:
            if not self.has_split_map_pools:
                raise ValueError(
                    "train_external_map_paths and val_external_map_paths "
                    "must both be provided")
            self.external_map_paths = list(dict.fromkeys(
                self.train_external_map_paths + self.val_external_map_paths))
            self.external_map_path = self.train_external_map_paths[0]
        else:
            self.train_external_map_paths = list(self.external_map_paths)
            self.val_external_map_paths = list(self.external_map_paths)
        self.current_external_map_index = None
        self.external_map_format = rospy.get_param(
            '~external_map_format', 'pcd')  # 支持 pcd/ply/txt/heightmap/las/laz
        self.train_external_source_profile = str(rospy.get_param(
            '~train_external_source_profile', 'als')).strip().lower()
        self.val_external_source_profile = str(rospy.get_param(
            '~val_external_source_profile', 'als')).strip().lower()
        for profile_name, profile in (
                ('train_external_source_profile',
                 self.train_external_source_profile),
                ('val_external_source_profile',
                 self.val_external_source_profile)):
            if profile not in ('als', 'uls'):
                raise ValueError(
                    f"{profile_name} must be 'als' or 'uls', got {profile!r}")
        self.external_domain = str(rospy.get_param(
            '~external_domain', '')).strip()
        self.train_external_source_url = str(rospy.get_param(
            '~train_external_source_url', '')).strip()
        self.val_external_source_url = str(rospy.get_param(
            '~val_external_source_url', '')).strip()
        self.train_external_license = str(rospy.get_param(
            '~train_external_license', '')).strip()
        self.val_external_license = str(rospy.get_param(
            '~val_external_license', '')).strip()
        self.train_external_crs = str(rospy.get_param(
            '~train_external_crs', '')).strip()
        self.val_external_crs = str(rospy.get_param(
            '~val_external_crs', '')).strip()
        self.train_external_site_id = str(rospy.get_param(
            '~train_external_site_id', '')).strip()
        self.val_external_site_id = str(rospy.get_param(
            '~val_external_site_id', '')).strip()
        configured_grades = rospy.get_param(
            '~external_accepted_grades', 'easy,medium,hard')
        if isinstance(configured_grades, str):
            configured_grades = configured_grades.replace(';', ',').split(',')
        self.external_accepted_grades = tuple(
            str(grade).strip().lower()
            for grade in configured_grades
            if str(grade).strip())
        if not self.external_accepted_grades:
            raise ValueError("external_accepted_grades must not be empty")
        # Online LAZ/ LAS fitting parameters.  A negative value means use the
        # profile default; the resolved values are written into crop metadata.
        self.external_fit_radius = float(rospy.get_param(
            '~external_fit_radius', -1.0))
        self.external_surface_cell_size = float(rospy.get_param(
            '~external_surface_cell_size', -1.0))
        self.external_direct_fit_min_points = int(rospy.get_param(
            '~external_direct_fit_min_points', -1))
        self.external_ground_band_below = float(rospy.get_param(
            '~external_ground_band_below', 0.25))
        self.external_ground_band_above = float(rospy.get_param(
            '~external_ground_band_above', 0.35))
        self.external_envelope_outlier = float(rospy.get_param(
            '~external_envelope_outlier', 0.75))
        self.external_planner_surface_resolution = float(rospy.get_param(
            '~external_planner_surface_resolution', 0.05))
        self.external_raw_below_surface_tolerance = float(rospy.get_param(
            '~external_raw_below_surface_tolerance', 1.0))
        self.external_raw_above_surface_tolerance = float(rospy.get_param(
            '~external_raw_above_surface_tolerance', 50.0))
        self.external_above_surface_height = float(rospy.get_param(
            '~external_above_surface_height', 2.0))
        self.external_above_surface_cell_size = float(rospy.get_param(
            '~external_above_surface_cell_size', 1.0))
        self.external_max_above_surface_coverage = float(rospy.get_param(
            '~external_max_above_surface_coverage', 1.0))
        self.external_min_above_surface_component_cells = int(
            rospy.get_param('~external_min_above_surface_component_cells', 0))
        self.external_tree_class_values = parse_las_class_values(
            rospy.get_param('~external_tree_class_values', '1'),
            'external_tree_class_values')
        self.external_ground_class_values = parse_las_class_values(
            rospy.get_param('~external_ground_class_values', '2'),
            'external_ground_class_values')
        self.external_min_tree_component_cells = int(
            rospy.get_param('~external_min_tree_component_cells', 0))
        self.external_map_is_canonical = bool(
            rospy.get_param('~external_map_is_canonical', False))
        if (self.use_external_map and self.external_map_is_canonical
                and not self.has_split_map_pools):
            raise ValueError(
                "canonical maps require train_external_map_paths and "
                "val_external_map_paths so val does not reuse the train pool")
        self.canonical_maps_per_environment = int(
            rospy.get_param('~canonical_maps_per_environment', 1))
        if self.canonical_maps_per_environment <= 0:
            raise ValueError(
                "canonical_maps_per_environment must be positive")
        self.canonical_primary_scene_count = int(rospy.get_param(
            '~canonical_primary_scene_count', self.num_environments))
        self.canonical_pool_start_env_id = int(rospy.get_param(
            '~canonical_pool_start_env_id', self.start_env_id))
        if self.canonical_primary_scene_count <= 0:
            raise ValueError("canonical_primary_scene_count must be positive")
        self.canonical_replacement_round = {}
        self.canonical_map_rejections = []
        if self.external_map_is_canonical and self.has_split_map_pools:
            expected_pool_size = (
                self.canonical_primary_scene_count
                * self.canonical_maps_per_environment)
            for split_name, split_paths in (
                    ('train', self.train_external_map_paths),
                    ('val', self.val_external_map_paths)):
                if len(split_paths) < expected_pool_size:
                    raise ValueError(
                        f"{split_name} canonical pool has {len(split_paths)} "
                        f"maps; expected at least {expected_pool_size}")
        self.target_map_size = float(rospy.get_param('~target_map_size', 20.0))
        self.target_resolution = float(rospy.get_param('~target_resolution', 0.2))
        self.external_map_fixed_yaw_deg = float(
            rospy.get_param('~external_map_fixed_yaw_deg', 180.0))
        self.external_map_physical_size = float(
            rospy.get_param('~external_map_physical_size', 0.0))
        self.external_map_min_physical_size = float(
            rospy.get_param('~external_map_min_physical_size', 0.0))
        self.scale_external_map_z = bool(
            rospy.get_param('~scale_external_map_z', False))
        self.crop_rotation_min_deg = float(
            rospy.get_param('~crop_rotation_min_deg', -180.0))
        self.crop_rotation_max_deg = float(
            rospy.get_param('~crop_rotation_max_deg', 180.0))
        self.crop_padding = float(rospy.get_param('~crop_padding', 1.0))
        self.crop_min_points = int(rospy.get_param('~crop_min_points', 50))
        self.crop_min_coverage = float(
            rospy.get_param('~crop_min_coverage', 0.85))
        self.crop_max_attempts = int(rospy.get_param('~crop_max_attempts', 50))
        self.crop_random_seed = int(rospy.get_param('~crop_random_seed', -1))
        self.crop_rng = np.random.default_rng(
            None if self.crop_random_seed < 0 else self.crop_random_seed)
        self.current_crop_info = None
        self.current_source_transform = None
        self.current_canonical_grid = None
        self.external_source_metadata = {}
        self.external_source_classifications = {}
        # 每个worker对每个源地图只读取并规范化一次。后续环境以及地图重生成
        # 都只执行随机裁剪，避免反复加载点云。
        self.cached_external_source_maps = {}

        if self.target_map_size <= 0.0 or self.target_resolution <= 0.0:
            raise ValueError("target_map_size and target_resolution must be positive")
        target_cells = self.target_map_size / self.target_resolution
        if not np.isclose(target_cells, round(target_cells), atol=1e-6):
            raise ValueError(
                "target_map_size must be an integer multiple of target_resolution")
        if self.crop_padding < 0.0:
            raise ValueError("crop_padding must be non-negative")
        if self.external_map_min_physical_size < 0.0:
            raise ValueError("external_map_min_physical_size must be non-negative")
        if self.external_above_surface_height < 0.0:
            raise ValueError(
                "external_above_surface_height must be non-negative")
        if self.external_above_surface_cell_size <= 0.0:
            raise ValueError(
                "external_above_surface_cell_size must be positive")
        if not 0.0 <= self.external_max_above_surface_coverage <= 1.0:
            raise ValueError(
                "external_max_above_surface_coverage must be in [0, 1]")
        if self.external_min_above_surface_component_cells < 0:
            raise ValueError(
                "external_min_above_surface_component_cells must be "
                "non-negative")
        if self.external_min_tree_component_cells < 0:
            raise ValueError(
                "external_min_tree_component_cells must be non-negative")
        if not 0.0 <= self.crop_min_coverage <= 1.0:
            raise ValueError("crop_min_coverage must be in [0, 1]")
        if self.crop_rotation_max_deg < self.crop_rotation_min_deg:
            raise ValueError(
                "crop_rotation_max_deg must be >= crop_rotation_min_deg")
        
        # 验证外部地图配置
        if self.use_external_map:
            if not self.external_map_paths:
                raise ValueError(
                    "external_map_paths or external_map_path must be specified "
                    "when use_external_map is True")
            for map_path in self.external_map_paths:
                if not os.path.exists(map_path):
                    raise FileNotFoundError(
                        f"External map file not found: {map_path}")
        
        # 地形生成参数（仅在不使用外部地图时有效）
        self.map_size = rospy.get_param('~map_size', 10.0)
        self.map_resolution = rospy.get_param('~map_resolution', 0.02)
        self.max_height = rospy.get_param('~max_height', 3.0)
        self.min_height = rospy.get_param('~min_height', 0.0)
        
        # 地图转换参数（完全匹配grid_transformer.cpp）
        self.coarse_resolution = rospy.get_param('~coarse_resolution', 0.2)
        self.fine_resolution = rospy.get_param('~fine_resolution', 0.2)
        self.voxel_size = rospy.get_param('~voxel_size', 0.1)
        
        # 路径生成参数（动态计算，在地图生成后更新）
        self.effective_map_size = self.map_size  # 有效地图尺寸，外部地图时会更新
        self.map_x_min = -self.map_size / 2 + 1.0
        self.map_x_max = self.map_size / 2 - 1.0
        self.map_y_min = -self.map_size / 2 + 1.0
        self.map_y_max = self.map_size / 2 - 1.0
        self.min_distance = rospy.get_param('~min_distance', 2.0)
        self.publish_delay = rospy.get_param('~publish_delay', 0.1)
        self.generation_random_seed = int(
            rospy.get_param('~generation_random_seed', -1))
        if self.generation_random_seed >= 0:
            random.seed(self.generation_random_seed)
            np.random.seed(self.generation_random_seed)
            torch.manual_seed(self.generation_random_seed)
        self.prefilter_stable_poses = bool(
            rospy.get_param('~prefilter_stable_poses', True))
        # 保存前的最终验收严格复用 MPT 当前 stability contract。它与
        # planner/pose prefilter 的 binary yaw map 分离，避免二者语义混用。
        self.trajectory_stability_d_safe = float(
            rospy.get_param('~trajectory_stability_d_safe', 0.15))
        self.trajectory_stability_yaw_weight = float(
            rospy.get_param('~trajectory_stability_yaw_weight', 1.4))
        self.trajectory_stability_yaw_bins = int(
            rospy.get_param('~trajectory_stability_yaw_bins', 36))
        if self.trajectory_stability_d_safe < 0.0:
            raise ValueError("trajectory_stability_d_safe must be non-negative")
        if self.trajectory_stability_yaw_weight <= 0.0:
            raise ValueError("trajectory_stability_yaw_weight must be positive")
        if self.trajectory_stability_yaw_bins <= 1:
            raise ValueError("trajectory_stability_yaw_bins must be greater than one")
        self.pose_sampling_max_attempts = int(
            rospy.get_param('~pose_sampling_max_attempts', 1000))
        if self.pose_sampling_max_attempts <= 0:
            raise ValueError("pose_sampling_max_attempts must be positive")
        self.medium_distance_fraction = float(
            rospy.get_param('~medium_distance_fraction', 0.4))
        self.long_distance_fraction = float(
            rospy.get_param('~long_distance_fraction', 0.4))
        self.moderate_complexity_fraction = float(
            rospy.get_param('~moderate_complexity_fraction', 0.4))
        self.high_complexity_fraction = float(
            rospy.get_param('~high_complexity_fraction', 0.4))
        for fraction_name in (
                'medium_distance_fraction', 'long_distance_fraction',
                'moderate_complexity_fraction',
                'high_complexity_fraction'):
            fraction = getattr(self, fraction_name)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"{fraction_name} must be in [0, 1]")
        if (self.medium_distance_fraction +
                self.long_distance_fraction > 1.0):
            raise ValueError(
                "medium_distance_fraction + long_distance_fraction "
                "must be <= 1")
        if (self.moderate_complexity_fraction +
                self.high_complexity_fraction > 1.0):
            raise ValueError(
                "moderate_complexity_fraction + "
                "high_complexity_fraction must be <= 1")
        self.map_ready_delay = float(
            rospy.get_param('~map_ready_delay', 1.0))
        self.max_path_retries_before_regenerate = int(
            rospy.get_param('~max_path_retries_before_regenerate', 30))
        self.mark_unplannable_canonical = bool(
            rospy.get_param('~mark_unplannable_canonical', False))
        if self.map_ready_delay < 0.0:
            raise ValueError("map_ready_delay must be non-negative")
        
        # 状态变量
        self.current_env_id = self.start_env_id
        self.current_path_id = 0
        self.current_phase = self.start_phase  # 'train' or 'val'
        self.paths_generated_for_current_env = 0
        self.scene_failed_attempts = 0
        self._env_skip_status = None
        self._resume_existing_map = False
        self.waiting_for_result = False
        self.current_trajectory = None
        self.map_update_timeout = rospy.get_param('~map_update_timeout', 10.0)
        self.planner_connection_timeout = float(
            rospy.get_param('~planner_connection_timeout', 60.0))
        if self.planner_connection_timeout <= 0.0:
            raise ValueError("planner_connection_timeout must be positive")

        # 新增：用于轨迹稳定性验证的当前地图缓存
        self.current_normal_x = None
        self.current_normal_y = None
        self.current_normal_z = None
        self.current_map_bounds = None
        self.current_resolution = None
        self.current_yaw_scores = None
        self.current_yaw_bins = None
        self.current_trajectory_stability_esdf = None
        self.current_stable_pose_candidates = None
        self.current_endpoint_obstacle_mask = None
        self.current_endpoint_obstacle_stats = None
        self.max_pose_distance = None
        self.current_pose_sampling_profile = None
        self.current_path_profile_retry_round = 0

        # 一个路径只有一个重试管理者；定时器也只能有一个有效实例。
        self.planning_attempt_id = 0
        self.active_planning_attempt_id = None
        self.planning_timeout_timer = None
        self.next_path_timer = None
        self.next_path_schedule_id = 0
        self.path_retry_count = 0
        self.retry_statistics = {
            'planning_failed': 0,
            'trajectory_unstable': 0,
            'empty_trajectory': 0,
            'watchdog_timeout': 0,
            'environment_regenerated': 0,
            'canonical_path_resampled': 0,
            'canonical_map_rejected': 0,
            'canonical_map_replaced': 0,
        }
        self.map_update_request_id = 0
        self.current_planner_map_version = None

        # 添加轨迹标识，防止地图切换时的轨迹混乱
        self.expected_env_id = self.start_env_id
        self.expected_path_id = 0
        self.map_generation_timestamp = 0.0
        
        # 创建数据集目录结构
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.val_dir = os.path.join(self.dataset_dir, 'val')
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        experiment_manifest_name = str(rospy.get_param(
            '~experiment_manifest_name', 'experiment_manifest.json')).strip()
        if (not experiment_manifest_name or
                os.path.basename(experiment_manifest_name) != experiment_manifest_name):
            raise ValueError(
                "experiment_manifest_name must be a non-empty filename")
        self.experiment_manifest_path = os.path.join(
            self.dataset_dir, experiment_manifest_name)
        self.experiment_started_utc = datetime.datetime.now(
            datetime.timezone.utc).isoformat()
        self.experiment_status = 'initialized'
        self.attempt_records = []
        self.active_attempt_record = None
        self.map_application_records = []
        
        # 初始化组件
        if not self.use_external_map:
            # 只有在不使用外部地图时才初始化地形生成器
            self.terrain_generator = TerrainGenerator(
                map_size=self.map_size,
                resolution=self.map_resolution,
                max_height=self.max_height,
                min_height=self.min_height
            )
        else:
            self.terrain_generator = None
            rospy.loginfo(
                "Using %d external source map(s): %s",
                len(self.external_map_paths), self.external_map_paths)
        
        self.grid_transformer = GridTransformer(
            coarse_resolution=self.coarse_resolution,
            fine_resolution=self.fine_resolution,
            voxel_size=self.voxel_size
        )
        
        # ROS通信
        # 全部使用相对ROS名称；单worker时仍解析到原有根话题，多worker时
        # 自动隔离到各自命名空间。
        self.pointcloud_pub = rospy.Publisher('uneven_map/pointcloud', PointCloud2, queue_size=1, latch=True)
        # Occupancy map publisher (external occ map to uneven_map)
        self.occ_pub = rospy.Publisher('external_occ_map', OccupancyGrid, queue_size=1, latch=True)
        # 3D occupancy grid publisher (HWY layout) for receivers that expect HWY flattened
        from std_msgs.msg import Float32MultiArray
        self.occ3d_pub = rospy.Publisher('external_occ_grid_hwy', Float32MultiArray, queue_size=1, latch=True)
        self.terrain_update_service = rospy.ServiceProxy(
            'update_terrain_map', UpdateTerrainMap)
        self.start_pose_pub = rospy.Publisher('data_generate_node/start_pose', PoseStamped, queue_size=1)
        self.target_pose_pub = rospy.Publisher('data_generate_node/target_pose', PoseStamped, queue_size=1)
        
        self.traj_sub = rospy.Subscriber('data_generate_node/optimized_traj', SE2Traj, self.trajectory_callback)
        self.result_sub = rospy.Subscriber('data_generate_node/planning_result', Bool, self.planning_result_callback)
        self.map_regen_sub = rospy.Subscriber('data_generate_node/map_regeneration_request', Bool, self.map_regeneration_callback)
        self.write_experiment_manifest()
        rospy.on_shutdown(self.finalize_experiment_manifest)
        
        rospy.loginfo(f"TerrainDatasetGenerator initialized:")
        rospy.loginfo(f"  - num_environments: {self.num_environments}")
        rospy.loginfo(f"  - train_paths_per_env: {self.train_paths_per_env}")
        rospy.loginfo(f"  - val_paths_per_env: {self.val_paths_per_env}")
        rospy.loginfo(f"  - dataset_dir: {self.dataset_dir}")
        rospy.loginfo(f"  - start_env_id: {self.start_env_id}")
        if self.use_external_map:
            map_mode = ('canonical audited maps' if self.external_map_is_canonical
                        else 'random external crops')
            rospy.loginfo(
                f"  - {map_mode}: {self.target_map_size:.2f}m x "
                f"{self.target_map_size:.2f}m, {self.target_resolution:.3f}m/cell, "
                f"{int(round(target_cells))}x{int(round(target_cells))}")
        
        rospy.loginfo("TerrainDatasetGenerator ready to start!")
        
        # 注释掉等待外部服务的代码，直接使用内部流程
        # rospy.loginfo("Waiting for /start_data_generation service...")
        # rospy.wait_for_service('/start_data_generation', timeout=30)
        # try:
        #     trigger = rospy.ServiceProxy('/start_data_generation', Trigger)
        #     resp = trigger()
        #     rospy.loginfo("start_data_generation called: success=%s msg=%s", resp.success, resp.message)
        # except Exception as e:
        #     rospy.logerr("Failed to call /start_data_generation: %s", e)

    @staticmethod
    def _parse_map_paths(configured_paths):
        if isinstance(configured_paths, str):
            configured_paths = configured_paths.replace(';', ',').split(',')
        elif not isinstance(configured_paths, (list, tuple)):
            raise ValueError(
                "map paths must be a list or comma-separated string")
        return list(dict.fromkeys(
            str(path).strip() for path in configured_paths
            if str(path).strip()))

    @staticmethod
    def _file_record(path):
        """Record local file provenance without reading the full file."""
        absolute = os.path.abspath(path)
        if not os.path.isfile(absolute):
            return {'path': absolute, 'missing': True}
        stat = os.stat(absolute)
        return {
            'path': absolute,
            'size_bytes': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns),
        }

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(key): TerrainDatasetGenerator._json_safe(item)
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [TerrainDatasetGenerator._json_safe(item)
                    for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    def _ros_parameter_snapshot(self):
        snapshot = {}
        relevant_prefixes = (
            rospy.get_name() + '/', '/uneven_map/', '/alm_traj_opt/',
            '/kino_astar/', '/manager/', '/only_planner/')
        for name in sorted(rospy.get_param_names()):
            if name == rospy.get_name() or name.startswith(relevant_prefixes):
                try:
                    snapshot[name] = self._json_safe(rospy.get_param(name))
                except (KeyError, rospy.ROSException) as exc:
                    snapshot[name] = {'read_error': str(exc)}
        return snapshot

    def write_experiment_manifest(self):
        """Atomically persist parameters and outcomes throughout the run."""
        source_maps = [
            self._file_record(path) for path in self.external_map_paths]
        path_files = sorted(glob.glob(os.path.join(
            self.dataset_dir, '*', 'env*', 'path_*.p')))
        map_files = sorted(glob.glob(os.path.join(
            self.dataset_dir, '*', 'env*', 'map.p')))
        manifest = {
            'schema_version': 'terrain-planner-experiment-v2',
            'status': self.experiment_status,
            'started_utc': self.experiment_started_utc,
            'updated_utc': datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            'node_name': rospy.get_name(),
            'python_version': sys.version,
            'dependency_versions': {
                'numpy': np.__version__,
                'open3d': getattr(o3d, '__version__', 'unknown'),
                'torch': getattr(torch, '__version__', 'unknown'),
                'opencv': getattr(cv2, '__version__', 'unknown'),
            },
            'implementation_file': os.path.abspath(__file__),
            'source_maps': source_maps,
            'ros_parameters': self._ros_parameter_snapshot(),
            'attempts': self.attempt_records,
            'map_applications': self.map_application_records,
            'canonical_map_rejections': self.canonical_map_rejections,
            'retry_statistics': dict(self.retry_statistics),
            'outputs': {
                'maps': len(map_files),
                'paths': len(path_files),
                'train_paths': len([p for p in path_files
                                    if os.sep + 'train' + os.sep in p]),
                'val_paths': len([p for p in path_files
                                  if os.sep + 'val' + os.sep in p]),
            },
        }
        temporary = self.experiment_manifest_path + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(self._json_safe(manifest), stream, indent=2,
                      ensure_ascii=False, sort_keys=True)
            stream.write('\n')
        os.replace(temporary, self.experiment_manifest_path)

    def finalize_experiment_manifest(self):
        if self.experiment_status in ('initialized', 'running'):
            self.experiment_status = 'interrupted'
        try:
            self.write_experiment_manifest()
        except Exception as exc:
            rospy.logerr("Failed to finalize experiment manifest: %s", exc)
    
    def pointcloud_to_ros_message(self, pcd):
        """将Open3D点云转换为ROS PointCloud2消息"""
        import sensor_msgs.point_cloud2 as pc2
        from sensor_msgs.msg import PointField
        
        points = np.asarray(pcd.points, dtype=np.float32)
        # # 将点云绕中心旋转180度
        # # 计算旋转矩阵
        # R = np.array([[-1, 0, 0],
        #               [0, -1, 0],
        #               [0, 0, 1]])
        # points = points.dot(R.T)
        
        # 创建ROS PointCloud2消息
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        
        # 使用更简单可靠的方法创建点云
        ros_pointcloud = pc2.create_cloud_xyz32(header, points)

        return ros_pointcloud

    def select_external_map_path(self):
        """从当前 split 的地图池中确定性轮换 canonical 场景。"""
        phase_paths = (self.train_external_map_paths
                       if self.current_phase == 'train'
                       else self.val_external_map_paths)
        source_count = len(phase_paths)
        if self.external_map_is_canonical:
            if not self.has_split_map_pools:
                raise RuntimeError(
                    "canonical generation requires split train/val map pools")
            local_environment = (
                self.current_env_id - self.canonical_pool_start_env_id)
            replacement_round = self.canonical_replacement_round.get(
                (self.current_phase, self.current_env_id), 0)
            source_index = (
                local_environment
                + replacement_round * self.canonical_primary_scene_count)
            if source_index >= source_count:
                raise RuntimeError(
                    f"Canonical {self.current_phase} map pool exhausted: "
                    f"need index {source_index}, have {source_count} maps")
        else:
            source_index = self.current_env_id % source_count
            if (not self.has_split_map_pools and
                    self.current_phase == 'val' and source_count > 1):
                source_index = (source_index + 1) % source_count
        self.current_external_map_index = source_index
        self.external_map_path = phase_paths[source_index]
        rospy.loginfo(
            "Selected external source map %d/%d for env%06d (%s): %s",
            source_index + 1, source_count, self.current_env_id,
            self.current_phase, self.external_map_path)
        return self.external_map_path

    def prepare_external_map_frame(self, pcd):
        """应用外部地图的固定坐标系旋转（默认保持原有的180度变换）。"""
        points = np.asarray(pcd.points, dtype=np.float64)
        angle = math.radians(self.external_map_fixed_yaw_deg)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        rotated_points = points.copy()
        rotated_points[:, 0] = cos_angle * points[:, 0] - sin_angle * points[:, 1]
        rotated_points[:, 1] = sin_angle * points[:, 0] + cos_angle * points[:, 1]
        pcd.points = o3d.utility.Vector3dVector(rotated_points)
        return pcd

    def normalize_external_map_scale(self, pcd):
        """
        将外部点云的源坐标单位统一转换为米，并将XY中心移到原点。

        external_map_physical_size > 0 时把较长的横向跨度缩放到该尺寸；
        否则保留米制坐标，但可通过external_map_min_physical_size仅放大过小
        的源地图。高度默认不跟随XY缩放，只有显式启用scale_external_map_z
        时才会缩放。
        """
        points = np.asarray(pcd.points, dtype=np.float64)
        finite_mask = np.isfinite(points).all(axis=1)
        if not np.any(finite_mask):
            raise ValueError("External map contains no finite points")

        finite_xy = points[finite_mask, :2]
        raw_min = finite_xy.min(axis=0)
        raw_max = finite_xy.max(axis=0)
        raw_center = (raw_min + raw_max) / 2.0
        raw_extent = raw_max - raw_min
        max_horizontal_extent = float(np.max(raw_extent))
        min_horizontal_extent = float(np.min(raw_extent))
        if max_horizontal_extent <= 0.0:
            raise ValueError("External map has zero horizontal extent")

        if self.external_map_physical_size > 0.0:
            scale = self.external_map_physical_size / max_horizontal_extent
        elif (self.external_map_min_physical_size > 0.0 and
              min_horizontal_extent < self.external_map_min_physical_size):
            scale = (
                self.external_map_min_physical_size / min_horizontal_extent)
        else:
            scale = 1.0

        normalized_points = points.copy()
        normalized_points[:, 0] = (points[:, 0] - raw_center[0]) * scale
        normalized_points[:, 1] = (points[:, 1] - raw_center[1]) * scale
        if self.scale_external_map_z:
            normalized_points[:, 2] = points[:, 2] * scale
        pcd.points = o3d.utility.Vector3dVector(normalized_points)

        transform_info = {
            'raw_bounds': (
                float(raw_min[0]), float(raw_max[0]),
                float(raw_min[1]), float(raw_max[1])),
            'raw_center': (float(raw_center[0]), float(raw_center[1])),
            'raw_horizontal_extent': (
                float(raw_extent[0]), float(raw_extent[1])),
            'meters_per_source_unit': float(scale),
            'physical_size': float(
                max_horizontal_extent * scale),
            'scale_z': self.scale_external_map_z
        }
        rospy.loginfo(
            "Normalized external map: raw extent=(%.3f, %.3f), "
            "scale=%.8f m/source-unit, physical extent=(%.3f, %.3f)m",
            raw_extent[0], raw_extent[1], scale,
            raw_extent[0] * scale, raw_extent[1] * scale)
        finite_z = normalized_points[finite_mask, 2]
        rospy.loginfo(
            "External map height range after normalization: [%.3f, %.3f]m "
            "(scale_z=%s)",
            float(np.min(finite_z)), float(np.max(finite_z)),
            self.scale_external_map_z)
        return pcd, transform_info

    def sample_rotated_external_crop(self, source_pcd, max_attempts=None,
                                     coverage_resolution=None,
                                     source_classification=None,
                                     return_classification=False):
        """
        从外部点云随机选取一个可旋转的正方形区域，并转换到局部地图坐标系。

        返回严格位于目标范围内的点云、仅供栅格转换使用的padding点云，
        以及裁剪元数据。两份点云中心均位于原点，目标区域在
        [-target_map_size/2, target_map_size/2) 内。
        """
        all_source_points = np.asarray(source_pcd.points, dtype=np.float64)
        finite_mask = np.isfinite(all_source_points).all(axis=1)
        all_points_are_finite = bool(np.all(finite_mask))
        if source_classification is not None:
            all_source_classification = np.asarray(source_classification)
            if (all_source_classification.ndim != 1 or
                    len(all_source_classification) != len(all_source_points)):
                raise ValueError(
                    "source_classification must align with source point cloud")
            source_classification = (
                all_source_classification if all_points_are_finite
                else all_source_classification[finite_mask])
        elif return_classification:
            raise ValueError(
                "return_classification requires source_classification")
        # 常见PCD全部为有限值，此时保留Open3D的零拷贝视图；原先每次
        # 裁剪都会复制约500万点，既耗时又放大多worker的内存峰值。
        if all_points_are_finite:
            source_points = all_source_points
        else:
            source_points = all_source_points[finite_mask]
        if len(source_points) == 0:
            raise ValueError("External map contains no finite points")

        source_colors = None
        colors = np.asarray(source_pcd.colors)
        if len(colors) == len(all_source_points):
            source_colors = (
                colors if all_points_are_finite else colors[finite_mask])

        source_min = source_points[:, :2].min(axis=0)
        source_max = source_points[:, :2].max(axis=0)
        crop_half_size = self.target_map_size / 2.0
        selection_half_size = crop_half_size + self.crop_padding

        attempt_limit = (self.crop_max_attempts if max_attempts is None
                         else int(max_attempts))
        if attempt_limit <= 0:
            raise ValueError("max_attempts must be positive")
        support_resolution = (
            self.coarse_resolution if coverage_resolution is None
            else float(coverage_resolution))
        if support_resolution <= 0.0:
            raise ValueError("coverage_resolution must be positive")

        for attempt in range(1, attempt_limit + 1):
            crop_yaw_deg = float(self.crop_rng.uniform(
                self.crop_rotation_min_deg, self.crop_rotation_max_deg))
            crop_yaw = math.radians(crop_yaw_deg)
            cos_yaw = math.cos(crop_yaw)
            sin_yaw = math.sin(crop_yaw)

            # 旋转正方形在源坐标轴上的半宽。以此约束中心，保证整个
            # 目标区域（含法向量估计 padding）都落在源地图包围盒内。
            source_axis_extent = selection_half_size * (
                abs(cos_yaw) + abs(sin_yaw))
            center_min = source_min + source_axis_extent
            center_max = source_max - source_axis_extent
            if np.any(center_min > center_max):
                continue

            center = self.crop_rng.uniform(center_min, center_max)
            delta_x = source_points[:, 0] - center[0]
            delta_y = source_points[:, 1] - center[1]

            # source -> crop local: R(-yaw) * (point - center)
            local_x = cos_yaw * delta_x + sin_yaw * delta_y
            local_y = -sin_yaw * delta_x + cos_yaw * delta_y
            exact_mask = (
                (np.abs(local_x) < crop_half_size) &
                (np.abs(local_y) < crop_half_size))
            exact_point_count = int(np.count_nonzero(exact_mask))
            if exact_point_count < self.crop_min_points:
                continue

            # 用C++聚合所采用的粗栅格分辨率检查空间覆盖率。仅检查点数
            # 会让扫描线状的稀疏裁剪通过，随后在100x100地图中产生大量NaN。
            coarse_grid_size = int(math.ceil(
                self.target_map_size / support_resolution))
            exact_x = local_x[exact_mask]
            exact_y = local_y[exact_mask]
            coarse_ix = np.floor(
                (exact_x + crop_half_size) / support_resolution).astype(int)
            coarse_iy = np.floor(
                (exact_y + crop_half_size) / support_resolution).astype(int)
            valid_indices = (
                (coarse_ix >= 0) & (coarse_ix < coarse_grid_size) &
                (coarse_iy >= 0) & (coarse_iy < coarse_grid_size))
            occupied_linear = np.unique(
                coarse_iy[valid_indices] * coarse_grid_size +
                coarse_ix[valid_indices])
            coarse_coverage = (
                len(occupied_linear) / float(coarse_grid_size ** 2))
            if coarse_coverage < self.crop_min_coverage:
                continue

            padded_mask = (
                (np.abs(local_x) < selection_half_size) &
                (np.abs(local_y) < selection_half_size))
            cropped_points = source_points[padded_mask].copy()
            cropped_points[:, 0] = local_x[padded_mask]
            cropped_points[:, 1] = local_y[padded_mask]

            cropped_pcd = o3d.geometry.PointCloud()
            cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)
            if source_colors is not None:
                cropped_pcd.colors = o3d.utility.Vector3dVector(
                    source_colors[padded_mask])

            # 对外发布和保存可视化时必须严格保持20m目标范围；带padding
            # 的版本只供C++法向量估计使用，不能当成最终裁剪点云。
            target_points = source_points[exact_mask].copy()
            target_points[:, 0] = local_x[exact_mask]
            target_points[:, 1] = local_y[exact_mask]
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_points)
            if source_colors is not None:
                target_pcd.colors = o3d.utility.Vector3dVector(
                    source_colors[exact_mask])

            fixed_yaw = math.radians(self.external_map_fixed_yaw_deg)
            fixed_cos = math.cos(fixed_yaw)
            fixed_sin = math.sin(fixed_yaw)
            center_in_original_frame = (
                fixed_cos * center[0] + fixed_sin * center[1],
                -fixed_sin * center[0] + fixed_cos * center[1])
            raw_center = self.current_source_transform['raw_center']
            source_scale = self.current_source_transform[
                'meters_per_source_unit']
            center_in_raw_source_frame = (
                center_in_original_frame[0] / source_scale + raw_center[0],
                center_in_original_frame[1] / source_scale + raw_center[1])
            yaw_in_original_frame_deg = (
                crop_yaw_deg - self.external_map_fixed_yaw_deg + 180.0
            ) % 360.0 - 180.0

            crop_info = {
                'center_in_source_frame': (float(center[0]), float(center[1])),
                'center_in_original_frame': (
                    float(center_in_original_frame[0]),
                    float(center_in_original_frame[1])),
                'center_in_raw_source_frame': (
                    float(center_in_raw_source_frame[0]),
                    float(center_in_raw_source_frame[1])),
                'yaw_deg': crop_yaw_deg,
                'yaw_rad': crop_yaw,
                'yaw_in_original_frame_deg': yaw_in_original_frame_deg,
                'fixed_source_yaw_deg': self.external_map_fixed_yaw_deg,
                'target_map_size': self.target_map_size,
                'target_resolution': self.target_resolution,
                'local_bounds': (
                    -crop_half_size, crop_half_size,
                    -crop_half_size, crop_half_size),
                'padding': self.crop_padding,
                'points_in_target': exact_point_count,
                'points_with_padding': int(len(cropped_points)),
                'coarse_coverage': float(coarse_coverage),
                'coarse_coverage_resolution': float(support_resolution),
                'sampling_attempt': attempt,
                'source_transform': dict(self.current_source_transform),
                'source_bounds': (
                    float(source_min[0]), float(source_max[0]),
                    float(source_min[1]), float(source_max[1]))
            }
            rospy.loginfo(
                "Selected external crop: center=(%.3f, %.3f), yaw=%.2fdeg, "
                "points=%d (+padding: %d), coarse coverage=%.1f%%, attempt=%d",
                center[0], center[1], crop_yaw_deg, exact_point_count,
                len(cropped_points), coarse_coverage * 100.0, attempt)
            if return_classification:
                return (target_pcd, cropped_pcd, crop_info,
                        source_classification[exact_mask].copy(),
                        source_classification[padded_mask].copy())
            return target_pcd, cropped_pcd, crop_info

        raise RuntimeError(
            "Unable to sample a valid rotated external-map crop after "
            f"{attempt_limit} attempts. Source bounds are "
            f"x[{source_min[0]:.3f}, {source_max[0]:.3f}], "
            f"y[{source_min[1]:.3f}, {source_max[1]:.3f}]. "
            "Check external_map_physical_size, or reduce crop_padding/"
            "crop_min_points/crop_min_coverage.")

    def external_source_profile_for_current_phase(self):
        """Return the quality profile used by the active dataset split."""
        profile = (self.train_external_source_profile
                   if self.current_phase == 'train'
                   else self.val_external_source_profile)
        if profile not in ('als', 'uls'):
            raise ValueError(
                "external source profile must be 'als' or 'uls', got "
                f"{profile!r}")
        return profile

    def external_source_description_for_current_phase(self):
        """Return split-specific provenance fields for saved map metadata."""
        if self.current_phase == 'train':
            return {
                'source_url': self.train_external_source_url,
                'license': self.train_external_license,
                'source_crs': self.train_external_crs,
                'site_id': self.train_external_site_id,
            }
        return {
            'source_url': self.val_external_source_url,
            'license': self.val_external_license,
            'source_crs': self.val_external_crs,
            'site_id': self.val_external_site_id,
        }

    def online_surface_fit_args(self):
        """Resolve profile defaults and explicit online fitting parameters."""
        profile = self.external_source_profile_for_current_phase()
        defaults = {
            'als': {
                'fit_radius': 0.90,
                'surface_cell_size': 1.00,
                'direct_fit_min_points': 5,
                'coverage_resolution': 1.00,
            },
            'uls': {
                'fit_radius': 0.35,
                'surface_cell_size': 0.50,
                'direct_fit_min_points': 8,
                'coverage_resolution': 0.20,
            },
        }[profile]

        fit_radius = (self.external_fit_radius
                      if self.external_fit_radius > 0.0
                      else defaults['fit_radius'])
        surface_cell_size = (
            self.external_surface_cell_size
            if self.external_surface_cell_size > 0.0
            else defaults['surface_cell_size'])
        direct_fit_min_points = (
            self.external_direct_fit_min_points
            if self.external_direct_fit_min_points > 0
            else defaults['direct_fit_min_points'])
        if self.external_ground_band_below < 0.0:
            raise ValueError("external_ground_band_below must be non-negative")
        if self.external_ground_band_above < 0.0:
            raise ValueError("external_ground_band_above must be non-negative")
        if self.external_envelope_outlier <= 0.0:
            raise ValueError("external_envelope_outlier must be positive")
        if self.external_planner_surface_resolution <= 0.0:
            raise ValueError(
                "external_planner_surface_resolution must be positive")
        if self.external_raw_below_surface_tolerance < 0.0:
            raise ValueError(
                "external_raw_below_surface_tolerance must be non-negative")
        if self.external_raw_above_surface_tolerance < 0.0:
            raise ValueError(
                "external_raw_above_surface_tolerance must be non-negative")

        return SimpleNamespace(
            size=self.target_map_size,
            resolution=self.target_resolution,
            fit_radius=fit_radius,
            surface_cell_size=surface_cell_size,
            ground_band_below=self.external_ground_band_below,
            ground_band_above=self.external_ground_band_above,
            envelope_outlier=self.external_envelope_outlier,
            direct_fit_min_points=direct_fit_min_points,
            coverage_resolution=defaults['coverage_resolution'],
            planner_surface_resolution=(
                self.external_planner_surface_resolution),
            raw_below_surface_tolerance=(
                self.external_raw_below_surface_tolerance),
            raw_above_surface_tolerance=(
                self.external_raw_above_surface_tolerance),
            above_surface_height=self.external_above_surface_height,
            above_surface_cell_size=self.external_above_surface_cell_size,
            max_above_surface_coverage=(
                self.external_max_above_surface_coverage),
            min_above_surface_component_cells=(
                self.external_min_above_surface_component_cells),
            tree_class_values=self.external_tree_class_values,
            ground_class_values=self.external_ground_class_values,
            min_tree_component_cells=self.external_min_tree_component_cells,
            source_profile=profile,
        )

    def evaluate_online_surface(self, surface, exact_point_count,
                                crop_info, fit_args):
        """Run the existing terrain quality policy on an in-memory crop.

        The quality module is intentionally shared with the offline mother-map
        sampler.  Its public interface is NPZ-based, so this creates one
        short-lived scratch pair per candidate; rejected candidates never
        enter the dataset directory or a sampled-scene pool.
        """
        map_size_area = self.target_map_size ** 2
        metadata = {
            'source_surface_density_points_per_m2': float(
                exact_point_count / map_size_area),
            'planner_point_density_points_per_m2': float(
                len(surface['planner_xyz']) / map_size_area),
            'source_support_coverage': float(
                crop_info['coarse_coverage']),
            'source_support_resolution_m': float(
                crop_info.get('coarse_coverage_resolution',
                              fit_args.coverage_resolution)),
            'processing': {
                'source_profile': fit_args.source_profile,
                'source_support_resolution_m': float(
                    fit_args.coverage_resolution),
                'fit_radius_m': float(fit_args.fit_radius),
                'surface_cell_size_m': float(fit_args.surface_cell_size),
                'ground_band_below_m': float(fit_args.ground_band_below),
                'ground_band_above_m': float(fit_args.ground_band_above),
                'envelope_outlier_m': float(fit_args.envelope_outlier),
                'direct_fit_minimum_points': int(
                    fit_args.direct_fit_min_points),
                'planner_surface_resolution_m': float(
                    fit_args.planner_surface_resolution),
                'raw_below_surface_tolerance_m': float(
                    fit_args.raw_below_surface_tolerance),
                'raw_above_surface_tolerance_m': float(
                    fit_args.raw_above_surface_tolerance),
                'above_surface_height_m': float(
                    fit_args.above_surface_height),
                'above_surface_cell_size_m': float(
                    fit_args.above_surface_cell_size),
                'max_above_surface_coverage': float(
                    fit_args.max_above_surface_coverage),
                'min_above_surface_component_cells': int(
                    fit_args.min_above_surface_component_cells),
                'tree_class_values': list(fit_args.tree_class_values),
                'ground_class_values': list(fit_args.ground_class_values),
                'min_tree_component_cells': int(
                    fit_args.min_tree_component_cells),
            },
        }
        normals = np.asarray(surface['normals'], dtype=np.float32)
        with tempfile.TemporaryDirectory(prefix='terrain-quality-') as temp_dir:
            npz_path = os.path.join(temp_dir, 'candidate.npz')
            np.savez_compressed(
                npz_path,
                elevation=np.asarray(surface['elevation'], dtype=np.float32),
                normal_x=normals[:, :, 0],
                normal_y=normals[:, :, 1],
                normal_z=normals[:, :, 2],
                valid_mask=np.asarray(surface['valid'], dtype=bool),
                observed_mask=np.asarray(surface['observed'], dtype=bool),
                resolution=np.float32(self.target_resolution),
                size=np.float32(self.target_map_size),
            )
            with open(os.path.splitext(npz_path)[0] + '.json', 'w',
                      encoding='utf-8') as stream:
                json.dump(metadata, stream, ensure_ascii=False)
            quality = evaluate_terrain_map(npz_path)
        quality.pop('path', None)
        return quality, metadata

    def prepare_online_laz_crop(self, source_pcd, source_classification=None):
        """Sample, fit, grade, and return one accepted LAZ crop.

        A crop that fails fitting or the quality gate is discarded immediately;
        the next attempt starts from the same cached mother map.  Nothing is
        written until this method returns an accepted fitted surface.
        """
        fit_args = self.online_surface_fit_args()
        last_reason = 'no candidate was sampled'
        for quality_attempt in range(1, self.crop_max_attempts + 1):
            try:
                sampled_crop = self.sample_rotated_external_crop(
                    source_pcd,
                    max_attempts=1,
                    coverage_resolution=fit_args.coverage_resolution,
                    source_classification=source_classification,
                    return_classification=source_classification is not None)
                if source_classification is None:
                    target_pcd, padded_pcd, crop_info = sampled_crop
                    target_classification = None
                else:
                    (target_pcd, padded_pcd, crop_info,
                     target_classification, _) = sampled_crop
            except RuntimeError as exc:
                last_reason = str(exc)
                continue

            exact = np.asarray(target_pcd.points, dtype=np.float64)
            padded = np.asarray(padded_pcd.points, dtype=np.float64)
            try:
                surface = build_surface(exact, padded, fit_args)
                quality, quality_metadata = self.evaluate_online_surface(
                    surface, len(exact), crop_info, fit_args)
                above_surface = measure_above_surface_coverage(
                    surface,
                    self.target_map_size,
                    self.target_resolution,
                    fit_args.above_surface_height,
                    fit_args.above_surface_cell_size)
                classified_above_surface = None
                if target_classification is not None:
                    classified_above_surface = (
                        measure_classified_above_surface_coverage(
                            surface,
                            exact,
                            target_classification,
                            self.target_map_size,
                            self.target_resolution,
                            fit_args.above_surface_height,
                            fit_args.above_surface_cell_size,
                            fit_args.tree_class_values,
                            fit_args.ground_class_values))
            except Exception as exc:
                last_reason = f'online surface fitting failed: {exc}'
                rospy.logdebug(
                    "Rejecting LAZ crop %d/%d for fitting failure: %s",
                    quality_attempt, self.crop_max_attempts, exc)
                continue

            crop_info['online_surface_fitting'] = True
            crop_info['quality_attempt'] = quality_attempt
            crop_info['quality'] = quality
            crop_info['quality_metadata'] = quality_metadata
            crop_info['above_surface'] = above_surface
            if classified_above_surface is not None:
                crop_info['classified_above_surface'] = (
                    classified_above_surface)
            crop_info['fit_parameters'] = quality_metadata['processing']
            crop_info['source_raw_points_in_patch'] = int(len(exact))
            crop_info['planner_point_count'] = int(len(surface['planner_xyz']))
            crop_info['ground_candidate_count'] = int(
                surface['ground_candidate_count'])
            crop_info['observed_fraction'] = float(
                np.mean(surface['observed']))
            crop_info['valid_fraction'] = float(np.mean(surface['valid']))
            crop_info['vertical_origin_m'] = float(surface['vertical_origin'])
            source_metadata = self.external_source_metadata.get(
                self.external_map_path, {})

            grade = str(quality.get('grade') or '').lower()
            above_surface_coverage = float(
                above_surface['above_surface_coverage_fraction'])
            density_accepted = (
                above_surface_coverage
                <= fit_args.max_above_surface_coverage)
            component_accepted = (
                above_surface['above_surface_largest_component_cells']
                >= fit_args.min_above_surface_component_cells)
            tree_component_accepted = (
                classified_above_surface is not None
                and classified_above_surface[
                    'tree_candidate_largest_component_cells']
                >= fit_args.min_tree_component_cells)
            if fit_args.min_tree_component_cells == 0:
                tree_component_accepted = True
            accepted = (
                quality.get('quality') == 'pass'
                and grade in self.external_accepted_grades
                and density_accepted
                and component_accepted
                and tree_component_accepted)
            if not accepted:
                reasons = quality.get('reasons') or []
                if not density_accepted:
                    reasons = list(reasons) + [
                        'above_surface_coverage %.3f > max %.3f' % (
                            above_surface_coverage,
                            fit_args.max_above_surface_coverage)]
                if not component_accepted:
                    reasons = list(reasons) + [
                        'above_surface_largest_component %d < min %d' % (
                            above_surface[
                                'above_surface_largest_component_cells'],
                            fit_args.min_above_surface_component_cells)]
                if not tree_component_accepted:
                    tree_largest = (0 if classified_above_surface is None else
                                    classified_above_surface[
                                        'tree_candidate_largest_component_cells'])
                    reasons = list(reasons) + [
                        'tree_candidate_largest_component %d < min %d' % (
                            tree_largest, fit_args.min_tree_component_cells)]
                last_reason = (
                    ','.join(reasons) if reasons
                    else f"grade {grade or 'none'} not accepted")
                rospy.loginfo(
                    "Rejected online LAZ crop %d/%d: quality=%s grade=%s "
                    "score=%s reasons=%s",
                    quality_attempt, self.crop_max_attempts,
                    quality.get('quality'), quality.get('grade'),
                    quality.get('geometry_score'), last_reason)
                continue

            source_description = (
                self.external_source_description_for_current_phase())
            mother_map_sample = dict(source_metadata)
            mother_map_sample.update(source_description)
            mother_map_sample.update({
                'source_file': os.path.abspath(self.external_map_path),
                'domain': self.external_domain,
                'patch_size_m': float(self.target_map_size),
                'resolution_m': float(self.target_resolution),
                'grid_shape': list(surface['elevation'].shape),
                'source_surface': 'all_finite_XYZ_no_LAS_class_filter',
                'source_raw_points_in_patch': int(len(exact)),
                'raw_returns_retained_in_planner_pcd': int(
                    len(surface['raw_xyz'])),
                'planner_point_count': int(len(surface['planner_xyz'])),
                'source_support_coverage': float(
                    crop_info['coarse_coverage']),
                'source_support_resolution_m': float(
                    crop_info['coarse_coverage_resolution']),
                'crop_center_in_source_frame': list(
                    crop_info['center_in_source_frame']),
                'crop_yaw_deg': float(crop_info['yaw_deg']),
                'quality': dict(quality),
                'above_surface': dict(above_surface),
                'classified_above_surface': (
                    dict(classified_above_surface)
                    if classified_above_surface is not None else None),
                'processing': dict(quality_metadata['processing']),
            })
            crop_info['mother_map_sample'] = mother_map_sample
            if source_metadata:
                crop_info['mother_map_source'] = dict(source_metadata)

            grid_map_data = {
                'elevation': np.asarray(surface['elevation'], dtype=np.float32),
                'normal_x': np.asarray(
                    surface['normals'][:, :, 0], dtype=np.float32),
                'normal_y': np.asarray(
                    surface['normals'][:, :, 1], dtype=np.float32),
                'normal_z': np.asarray(
                    surface['normals'][:, :, 2], dtype=np.float32),
                'valid_mask': np.asarray(surface['valid'], dtype=bool),
                'observed_mask': np.asarray(surface['observed'], dtype=bool),
                'bounds': (
                    -self.target_map_size / 2.0,
                    self.target_map_size / 2.0,
                    -self.target_map_size / 2.0,
                    self.target_map_size / 2.0),
                'resolution': self.target_resolution,
            }
            planner_pcd = o3d.geometry.PointCloud()
            planner_pcd.points = o3d.utility.Vector3dVector(
                np.asarray(surface['planner_xyz'], dtype=np.float64))
            planner_pcd.normals = o3d.utility.Vector3dVector(
                np.asarray(surface['planner_normals'], dtype=np.float64))
            rospy.loginfo(
                "Accepted online LAZ crop: grade=%s score=%.2f, "
                "raw=%d planner=%d valid=%.1f%% observed=%.1f%%",
                grade, float(quality.get('geometry_score', float('nan'))),
                len(exact), len(surface['planner_xyz']),
                100.0 * crop_info['valid_fraction'],
                100.0 * crop_info['observed_fraction'])
            return planner_pcd, planner_pcd, crop_info, grid_map_data

        raise RuntimeError(
            "Unable to obtain an accepted online LAZ crop after "
            f"{self.crop_max_attempts} candidates; last rejection: "
            f"{last_reason}")

    def validate_canonical_external_map(self, source_pcd):
        """Validate an already prepared metric map without augmenting it.

        Canonical maps are immutable base maps: no scale, rotation, random
        crop, or recentering is allowed here.  Their cell centers must cover
        the configured target bounds at the configured resolution.
        """
        points = np.asarray(source_pcd.points, dtype=np.float64)
        if len(points) == 0 or not np.isfinite(points).all():
            raise ValueError("Canonical external map contains invalid points")
        half_size = self.target_map_size / 2.0
        tolerance = max(1e-6, self.target_resolution * 0.05)
        if (np.any(points[:, 0] < -half_size - tolerance) or
                np.any(points[:, 0] >= half_size + tolerance) or
                np.any(points[:, 1] < -half_size - tolerance) or
                np.any(points[:, 1] >= half_size + tolerance)):
            raise ValueError(
                "Canonical external map has points outside configured bounds")

        xy_min = points[:, :2].min(axis=0)
        xy_max = points[:, :2].max(axis=0)
        expected_min = -half_size + 0.5 * self.target_resolution
        expected_max = half_size - 0.5 * self.target_resolution
        coverage_tolerance = self.target_resolution * 0.1
        if (np.any(xy_min > expected_min + coverage_tolerance) or
                np.any(xy_max < expected_max - coverage_tolerance)):
            raise ValueError(
                "Canonical external map does not cover the expected cell "
                f"centers [{expected_min:.3f}, {expected_max:.3f}]; got "
                f"min={xy_min}, max={xy_max}")

        grid_size = int(round(self.target_map_size / self.target_resolution))
        ix = np.floor(
            (points[:, 0] + half_size) / self.target_resolution).astype(int)
        iy = np.floor(
            (points[:, 1] + half_size) / self.target_resolution).astype(int)
        inside = ((ix >= 0) & (ix < grid_size) &
                  (iy >= 0) & (iy < grid_size))
        occupied = np.unique(iy[inside] * grid_size + ix[inside])
        coverage = len(occupied) / float(grid_size * grid_size)
        if coverage < 0.97:
            raise ValueError(
                f"Canonical external map coverage {coverage:.3f} < 0.97")

        z_min = float(points[:, 2].min())
        z_max = float(points[:, 2].max())
        # The canonical PCD now intentionally retains all raw returns in the
        # crop, including canopy/rock returns above the historical 5 m fitted
        # surface range. The 4-channel NPZ remains the bounded terrain grid;
        # only the lower offset contract applies to the raw PCD here.
        # No z-range gate is applied to the retained raw PCD. The 4-channel
        # NPZ is the bounded terrain grid; raw returns may lie below or above
        # its fitted surface and are kept for later runtime masking.

        source_transform = {
            'raw_bounds': (
                float(xy_min[0]), float(xy_max[0]),
                float(xy_min[1]), float(xy_max[1])),
            'raw_center': (0.0, 0.0),
            'raw_horizontal_extent': (
                float(xy_max[0] - xy_min[0]),
                float(xy_max[1] - xy_min[1])),
            'meters_per_source_unit': 1.0,
            'physical_size': self.target_map_size,
            'scale_z': False,
            'canonical': True,
        }
        crop_info = {
            'mode': 'canonical',
            'center_in_source_frame': (0.0, 0.0),
            'center_in_original_frame': (0.0, 0.0),
            'center_in_raw_source_frame': (0.0, 0.0),
            'yaw_deg': 0.0,
            'yaw_rad': 0.0,
            'yaw_in_original_frame_deg': 0.0,
            'fixed_source_yaw_deg': 0.0,
            'target_map_size': self.target_map_size,
            'target_resolution': self.target_resolution,
            'local_bounds': (
                -half_size, half_size, -half_size, half_size),
            'padding': 0.0,
            'points_in_target': int(len(points)),
            'points_with_padding': int(len(points)),
            'coarse_coverage': float(coverage),
            'sampling_attempt': 0,
            'source_transform': dict(source_transform),
            'source_bounds': (
                float(xy_min[0]), float(xy_max[0]),
                float(xy_min[1]), float(xy_max[1])),
        }
        rospy.loginfo(
            "Accepted canonical external map: points=%d, coverage=%.1f%%, "
            "z=[%.3f, %.3f]m; augmentation disabled",
            len(points), coverage * 100.0, z_min, z_max)
        return source_pcd, source_transform, crop_info

    def load_canonical_grid(self, map_path):
        sidecar_path = os.path.splitext(map_path)[0] + '.npz'
        if not os.path.exists(sidecar_path):
            raise FileNotFoundError(
                "Canonical map requires an NPZ grid sidecar: " +
                sidecar_path)
        sidecar = np.load(sidecar_path)
        required = {
            'elevation', 'normal_x', 'normal_y', 'normal_z',
            'valid_mask', 'resolution', 'size'}
        missing = sorted(required.difference(sidecar.files))
        if missing:
            raise ValueError(
                "Canonical NPZ is missing fields: " + ', '.join(missing))

        grid_size = int(round(self.target_map_size / self.target_resolution))
        expected_shape = (grid_size, grid_size)
        result = {}
        for field in ('elevation', 'normal_x', 'normal_y', 'normal_z'):
            value = np.asarray(sidecar[field], dtype=np.float32)
            if value.shape != expected_shape or not np.isfinite(value).all():
                raise ValueError(
                    f"Canonical NPZ field {field} must be finite with "
                    f"shape {expected_shape}, got {value.shape}")
            result[field] = value
        valid_mask = np.asarray(sidecar['valid_mask'], dtype=bool)
        if valid_mask.shape != expected_shape:
            raise ValueError(
                f"Canonical valid_mask has shape {valid_mask.shape}, "
                f"expected {expected_shape}")
        if 'observed_mask' in sidecar.files:
            observed_mask = np.asarray(sidecar['observed_mask'], dtype=bool)
        else:
            observed_mask = valid_mask.copy()
        if observed_mask.shape != expected_shape:
            raise ValueError(
                f"Canonical observed_mask has shape {observed_mask.shape}, "
                f"expected {expected_shape}")
        if valid_mask.mean() < 1.0:
            raise ValueError(
                "Canonical completed surface still contains invalid cells")
        if not math.isclose(
                float(sidecar['resolution']), self.target_resolution,
                rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Canonical NPZ resolution mismatch")
        if not math.isclose(
                float(sidecar['size']), self.target_map_size,
                rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Canonical NPZ size mismatch")
        half_size = 0.5 * self.target_map_size
        result['valid_mask'] = valid_mask
        result['observed_mask'] = observed_mask
        result['bounds'] = (-half_size, half_size, -half_size, half_size)
        result['resolution'] = self.target_resolution
        rospy.loginfo(
            "Loaded canonical grid sidecar %s: shape=%s, valid=%.2f%%, "
            "observed=%.2f%%",
            sidecar_path, expected_shape, valid_mask.mean() * 100.0,
            observed_mask.mean() * 100.0)
        return result

    @staticmethod
    def load_canonical_metadata(map_path):
        metadata_path = os.path.splitext(map_path)[0] + '.json'
        if not os.path.exists(metadata_path):
            return None
        with open(metadata_path, encoding='utf-8') as stream:
            return json.load(stream)
    
    @staticmethod
    def _remove_env_path_files(env_dir):
        removed = 0
        for path_file in glob.glob(os.path.join(env_dir, "path_*.p")):
            try:
                os.remove(path_file)
                removed += 1
            except OSError:
                pass
        return removed

    @staticmethod
    def _compact_env_path_files(env_dir):
        """Rename path_*.p to a contiguous path_0..path_{n-1} sequence."""
        files = []
        for path_file in glob.glob(os.path.join(env_dir, "path_*.p")):
            stem = os.path.splitext(os.path.basename(path_file))[0]
            try:
                files.append((int(stem.split("_", 1)[1]), path_file))
            except (IndexError, ValueError):
                continue
        files.sort()
        for new_id, (_, path_file) in enumerate(files):
            dest = os.path.join(env_dir, f"path_{new_id}.p")
            if os.path.abspath(path_file) == os.path.abspath(dest):
                continue
            if os.path.exists(dest):
                raise RuntimeError(
                    f"Cannot compact path files in {env_dir}: {dest} exists")
            os.rename(path_file, dest)
        return len(files)

    def generate_new_environment(self):
        """生成新的地形环境"""
        env_name = f"env{self.current_env_id:06d}"
        
        # 选择输出目录
        if self.current_phase == 'train':
            env_dir = os.path.join(self.train_dir, env_name)
        else:
            env_dir = os.path.join(self.val_dir, env_name)
        
        os.makedirs(env_dir, exist_ok=True)
        self._env_skip_status = None
        self._resume_existing_map = False
        if self.external_map_is_canonical and self.mark_unplannable_canonical:
            # Unchecked test generation must re-try the current canonical
            # map. Old 打回 markers and the previous 1 path belong to the
            # last test, not this map version.
            self.clear_canonical_needs_return(env_dir)
            self._remove_env_path_files(env_dir)
            self.current_path_id = 0
            self.paths_generated_for_current_env = 0
        elif self.external_map_is_canonical:
            if os.path.isfile(os.path.join(env_dir, "needs_return.json")):
                skip_status = "needs_return"
            else:
                if self.use_external_map:
                    selected_map_path = self.select_external_map_path()
                    if not existing_paths_belong_to_canonical(
                            env_dir, selected_map_path,
                            expected_split=self.current_phase):
                        discarded = self._remove_env_path_files(env_dir)
                        if discarded:
                            rospy.logwarn(
                                "Discarding %d stale %s %s trajectories; "
                                "they were not generated on %s",
                                discarded, self.current_phase, env_name,
                                selected_map_path)
                skip_status = self.canonical_env_skip_status(env_dir)
            if skip_status:
                self._env_skip_status = skip_status
                existing = glob.glob(os.path.join(env_dir, "path_*.p"))
                self.current_path_id = len(existing)
                self.paths_generated_for_current_env = len(existing)
                rospy.loginfo(
                    "Skipping %s %s (%s, existing paths=%d)",
                    self.current_phase, env_name, skip_status, len(existing))
                return env_dir
            existing_count = self._compact_env_path_files(env_dir)
            if existing_count:
                self.current_path_id = existing_count
                self.paths_generated_for_current_env = existing_count
                self._resume_existing_map = True
                rospy.loginfo(
                    "Resuming %s %s from path_%d",
                    self.current_phase, env_name, self.current_path_id)
            else:
                self.current_path_id = 0
                self.paths_generated_for_current_env = 0
        
        # 如果重新生成，清理旧的路径文件但保留目录结构
        if not (self.external_map_is_canonical
                and self.paths_generated_for_current_env > 0):
            removed = self._remove_env_path_files(env_dir)
            if removed:
                rospy.loginfo(
                    "Removed %d old path file(s) in %s", removed, env_name)
        
        rospy.loginfo(f"Generating environment {env_name} for {self.current_phase} phase")
        
        # 生成或加载地形
        if self.use_external_map:
            online_grid_map_data = None
            selected_map_path = self.select_external_map_path()
            cache_entry = self.cached_external_source_maps.get(
                selected_map_path)
            if cache_entry is None:
                rospy.loginfo("Loading and caching external source map...")
                source_pcd = self.load_external_map()
                if self.external_map_is_canonical:
                    source_pcd, source_transform, canonical_info = (
                        self.validate_canonical_external_map(source_pcd))
                    source_metadata = self.load_canonical_metadata(
                        selected_map_path)
                    if source_metadata is not None:
                        canonical_info['mother_map_sample'] = source_metadata
                        if isinstance(source_metadata.get('quality'), dict):
                            canonical_info['quality'] = dict(
                                source_metadata['quality'])
                    canonical_grid = self.load_canonical_grid(
                        selected_map_path)
                    cache_entry = (
                        source_pcd, source_transform, canonical_info,
                        canonical_grid,
                        self.external_source_classifications.get(
                            selected_map_path))
                else:
                    source_pcd, source_transform = (
                        self.normalize_external_map_scale(source_pcd))
                    source_pcd = self.prepare_external_map_frame(source_pcd)
                    cache_entry = (
                        source_pcd, source_transform, None, None,
                        self.external_source_classifications.get(
                            selected_map_path))
                self.cached_external_source_maps[selected_map_path] = cache_entry
            else:
                rospy.loginfo(
                    "Reusing cached external source map: %s",
                    selected_map_path)
            (source_pcd, self.current_source_transform, canonical_info,
             canonical_grid, source_classification) = cache_entry
            if self.external_map_is_canonical:
                pcd = source_pcd
                grid_input_pcd = source_pcd
                self.current_crop_info = dict(canonical_info)
                self.current_canonical_grid = canonical_grid
            else:
                self.current_canonical_grid = None
                if self.external_map_format.lower() in ('las', 'laz'):
                    (pcd, grid_input_pcd, self.current_crop_info,
                     online_grid_map_data) = self.prepare_online_laz_crop(
                         source_pcd, source_classification)
                else:
                    pcd, grid_input_pcd, self.current_crop_info = (
                        self.sample_rotated_external_crop(source_pcd))

            # 为外部地图创建基本的terrain_info
            points = np.asarray(pcd.points)
            terrain_info = {
                'map_size': self.target_map_size,
                'resolution': self.target_resolution,
                'max_height': np.max(points[:, 2]) if len(points) > 0 else self.max_height,
                'min_height': np.min(points[:, 2]) if len(points) > 0 else self.min_height,
                'num_points': len(points),
                'source': 'external_map',
                'external_map_path': self.external_map_path,
                'external_map_index': self.current_external_map_index,
                'external_map_format': self.external_map_format,
                'crop': dict(self.current_crop_info)
            }
            
            # 对于外部地图，不需要heightmap，设置为None
            heightmap = None
        else:
            # 生成地形
            rospy.loginfo("Generating new terrain...")
            heightmap, terrain_info = self.terrain_generator.generate_terrain()
            
            # 转换为点云
            pcd = self.terrain_generator.heightmap_to_pointcloud(heightmap)
        
        # 转换为栅格地图
        if self.use_external_map:
            if self.external_map_is_canonical:
                # The NPZ sidecar is the already audited network/occupancy
                # grid. Keep the dense PCD exclusively for UnevenMap fitting.
                grid_map_data = dict(self.current_canonical_grid)
            elif online_grid_map_data is not None:
                # LAZ/ LAS crops are fitted and graded before they reach the
                # planner.  Keep the fitted grid instead of re-rasterizing
                # the raw mother-map returns with the generic converter.
                grid_map_data = dict(online_grid_map_data)
            else:
                # 点云已经被旋转、裁剪并移到局部坐标系。只对20m目标区域
                # 栅格化，避免先处理整张大地图再做固定中心裁剪。
                half_size = self.target_map_size / 2.0
                map_bounds = (-half_size, half_size, -half_size, half_size)
                grid_map_data = self.grid_transformer.transform_pointcloud_to_grid(
                    grid_input_pcd, map_bounds)

            # 不再通过 resize 隐式改变物理分辨率；服务必须直接输出100x100。
            expected_grid_size = int(round(
                self.target_map_size / self.target_resolution))
            actual_grid_shape = grid_map_data['elevation'].shape
            actual_resolution = float(grid_map_data['resolution'])
            if actual_grid_shape != (expected_grid_size, expected_grid_size):
                raise RuntimeError(
                    f"Grid converter returned {actual_grid_shape}, expected "
                    f"({expected_grid_size}, {expected_grid_size})")
            if not math.isclose(
                    actual_resolution, self.target_resolution,
                    rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(
                    f"Grid converter returned resolution {actual_resolution}, "
                    f"expected {self.target_resolution}. Check "
                    "grid_fine_resolution in the launch file.")

            # 验证处理结果
            self.validate_external_map_processing(
                grid_map_data, self.target_map_size, self.target_resolution)
            
            # 更新位姿生成范围以匹配外部地图尺寸
            self.update_pose_generation_bounds(self.target_map_size)
        else:
            # 对于生成的地形，使用原来的逻辑
            map_bounds = (-self.map_size/2, self.map_size/2, -self.map_size/2, self.map_size/2)
            grid_map_data = self.grid_transformer.transform_pointcloud_to_grid(pcd, map_bounds)
            
            # 更新位姿生成范围
            self.update_pose_generation_bounds(self.map_size)

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
                valid_mask = np.asarray(
                    grid_map_data.get(
                        'valid_mask',
                        np.ones(self.current_normal_x.shape, dtype=bool)),
                    dtype=bool)
                if valid_mask.shape != self.current_normal_x.shape:
                    raise ValueError(
                        f"valid_mask shape {valid_mask.shape} does not match "
                        f"normal grid {self.current_normal_x.shape}")
                yaw_scores[~valid_mask, :] = 0.0
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

            # Final save admission uses a separate map with MPT's frozen yaw
            # discretization and continuous signed-distance semantics. Keep the
            # planner/pose prefilter above unchanged and binary.
            try:
                validation_scores = compute_map_yaw_bins(
                    self.current_normal_x,
                    self.current_normal_y,
                    self.current_normal_z,
                    self.trajectory_stability_yaw_bins)
                if isinstance(validation_scores, torch.Tensor):
                    validation_scores = (
                        validation_scores.detach().cpu().numpy())
                validation_scores = np.asarray(validation_scores)
                validation_scores[~valid_mask, :] = 0.0
                expected_shape = (
                    self.current_normal_x.shape[0],
                    self.current_normal_x.shape[1],
                    self.trajectory_stability_yaw_bins,
                )
                if validation_scores.shape != expected_shape:
                    raise ValueError(
                        "MPT validation yaw map shape mismatch: "
                        f"{validation_scores.shape} != {expected_shape}")
                self.current_trajectory_stability_esdf = (
                    build_periodic_signed_stability_esdf(
                        validation_scores,
                        voxel_size_xy=self.current_resolution,
                        yaw_weight=self.trajectory_stability_yaw_weight,
                    ))
                rospy.loginfo(
                    "Prepared MPT trajectory stability ESDF %s "
                    "(d_safe=%.3fm, yaw_weight=%.3f)",
                    self.current_trajectory_stability_esdf.shape,
                    self.trajectory_stability_d_safe,
                    self.trajectory_stability_yaw_weight)
            except Exception as e:
                rospy.logwarn(
                    f"Failed to build MPT trajectory stability ESDF: {e}")
                self.current_trajectory_stability_esdf = None
        except Exception as e:
            rospy.logwarn(f"Failed to cache map normals for trajectory validation: {e}")
            self.current_normal_x = self.current_normal_y = self.current_normal_z = None
            self.current_map_bounds = None
            self.current_resolution = None
            self.current_yaw_scores = None
            self.current_yaw_bins = None
            self.current_trajectory_stability_esdf = None
        self.prepare_endpoint_obstacle_mask(pcd, grid_map_data)
        self.prepare_stable_pose_candidates()

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
                nx = -np.array(nx, dtype=np.float32)
            if ny is not None:
                ny = -np.array(ny, dtype=np.float32)
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
            
            # # 应用180度旋转以匹配ROS坐标系（如果需要）
            # # 这是为了解决map.p文件与GridMap话题之间的坐标系差异
            # rospy.loginfo("Applying 180-degree rotation to match ROS coordinate system...")
            # grid_map_tensor = np.rot90(grid_map_tensor, k=2, axes=(0, 1))  # 180度旋转
            # rospy.loginfo(f"After rotation, grid shape: {grid_map_tensor.shape}")
            
        except Exception as e:
            rospy.logwarn(f"Failed to build grid_map_tensor from grid_map_data: {e}")
            grid_map_tensor = np.zeros((1, 1, 4), dtype=np.float32)

        # 保存地图数据，按照正确的格式
        if self.use_external_map:
            # 对于外部地图，确保使用目标参数
            target_bounds = (
                -self.target_map_size / 2.0, self.target_map_size / 2.0,
                -self.target_map_size / 2.0, self.target_map_size / 2.0)
            valid_mask = np.asarray(
                grid_map_data.get(
                    'valid_mask',
                    np.ones(grid_map_tensor.shape[:2], dtype=bool)),
                dtype=bool)
            observed_mask = np.asarray(
                grid_map_data.get('observed_mask', valid_mask), dtype=bool)
            if (valid_mask.shape != grid_map_tensor.shape[:2] or
                    observed_mask.shape != grid_map_tensor.shape[:2]):
                raise RuntimeError(
                    "External map masks must match the saved grid tensor: "
                    f"valid={valid_mask.shape}, observed={observed_mask.shape}, "
                    f"tensor={grid_map_tensor.shape[:2]}")
            sample_metadata = (
                (self.current_crop_info or {}).get('mother_map_sample') or {})
            quality = (self.current_crop_info or {}).get('quality')
            if quality is None:
                quality = sample_metadata.get('quality')

            map_data = {
                'tensor': grid_map_tensor,
                'bounds': target_bounds,
                'resolution': self.target_resolution,
                'map_name': env_name,
                'channels': ['elevation', 'normal_x', 'normal_y', 'normal_z'],
                'shape': grid_map_tensor.shape,
                'source': 'external_map',
                'dataset_phase': self.current_phase,
                'external_map_path': os.path.abspath(self.external_map_path),
                'original_map_path': (
                    None if self.external_map_is_canonical
                    else self.external_map_path),
                'original_map_index': self.current_external_map_index,
                'source_map': {
                    'scene': os.path.splitext(
                        os.path.basename(self.external_map_path))[0],
                    'split': self.current_phase,
                },
                'crop': dict(self.current_crop_info),
                'valid_mask': valid_mask,
                'observed_mask': observed_mask,
                'quality': dict(quality) if isinstance(quality, dict) else quality,
                'endpoint_obstacle_filter': (
                    dict(self.current_endpoint_obstacle_stats)
                    if self.current_endpoint_obstacle_stats is not None else None),
                'target_grid_size': f"{grid_map_tensor.shape[0]}x{grid_map_tensor.shape[1]}"
            }
            rospy.loginfo(
                f"External map: saved with target bounds {target_bounds}, "
                f"resolution {self.target_resolution}, "
                f"grid shape {grid_map_tensor.shape}")
        else:
            # 对于生成的地形，使用原来的逻辑
            map_data = {
                'tensor': grid_map_tensor,
                'bounds': grid_map_data['bounds'],
                'resolution': grid_map_data['resolution'],
                'map_name': env_name,
                'channels': ['elevation', 'normal_x', 'normal_y', 'normal_z'],
                'shape': grid_map_tensor.shape
            }
        
        # Build the exact HWY map consumed by the embedded UnevenMap. The
        # update service is the authoritative synchronization barrier: terrain
        # reconstruction and occupancy replacement both finish before it
        # returns, so no pose can be sent against an old or mixed map.
        try:
            occ_msg, occ_hwy_msg = self.create_occupancy_grid(grid_map_data)
        except Exception as e:
            rospy.logfatal(f"Failed to build planner occupancy grid: {e}")
            rospy.signal_shutdown("invalid planner occupancy grid")
            raise

        planner_pcd = grid_input_pcd if self.use_external_map else pcd
        ros_pointcloud = self.pointcloud_to_ros_message(planner_pcd)
        bounds = grid_map_data.get('bounds')
        resolution = float(grid_map_data.get('resolution'))
        if bounds is None or len(bounds) != 4:
            raise RuntimeError(
                f"Planner map bounds must be (min_x,max_x,min_y,max_y), "
                f"got {bounds}")
        self.map_update_request_id += 1
        update_request = UpdateTerrainMapRequest()
        update_request.request_id = self.map_update_request_id
        update_request.environment_id = self.current_env_id
        update_request.pointcloud = ros_pointcloud
        update_request.occupancy_hwy = occ_hwy_msg
        update_request.min_x = float(bounds[0])
        update_request.max_x = float(bounds[1])
        update_request.min_y = float(bounds[2])
        update_request.max_y = float(bounds[3])
        update_request.resolution = resolution
        try:
            update_response = self.terrain_update_service(update_request)
        except rospy.ServiceException as exc:
            rospy.logfatal(f"Atomic planner map update call failed: {exc}")
            rospy.signal_shutdown("planner map update service failed")
            raise RuntimeError("planner map update service failed") from exc
        if (not update_response.success or
                update_response.request_id != self.map_update_request_id or
                update_response.environment_id != self.current_env_id):
            raise RuntimeError(
                "Planner rejected or mis-correlated map update: "
                f"success={update_response.success}, "
                f"request={update_response.request_id}, "
                f"env={update_response.environment_id}, "
                f"message={update_response.message}")
        self.current_planner_map_version = int(update_response.map_version)
        map_data['planner_map_version'] = self.current_planner_map_version
        map_data['planner_map_request_id'] = self.map_update_request_id
        map_data['planner_point_count'] = int(update_response.point_count)
        map_application = {
            'environment_id': self.current_env_id,
            'phase': self.current_phase,
            'map_version': self.current_planner_map_version,
            'map_request_id': self.map_update_request_id,
            'point_count': int(update_response.point_count),
            'occupied_se2': int(update_response.occupied_voxel_count),
            'occupied_xy': int(update_response.occupied_xy_count),
            'source_yaw_bins': int(update_response.source_yaw_bins),
            'internal_yaw_bins': int(update_response.internal_yaw_bins),
        }
        self.map_application_records.append(map_application)
        self.write_experiment_manifest()
        rospy.loginfo(
            "Planner atomically applied %s: version=%d, points=%d, "
            "occupied_se2=%d, occupied_xy=%d, yaw=%d->%d",
            env_name, self.current_planner_map_version,
            update_response.point_count,
            update_response.occupied_voxel_count,
            update_response.occupied_xy_count,
            update_response.source_yaw_bins,
            update_response.internal_yaw_bins)

        map_file = os.path.join(env_dir, 'map.p')
        if self._resume_existing_map and os.path.isfile(map_file):
            rospy.loginfo(
                "Keeping existing map.p while resuming paths: %s", map_file)
        else:
            with open(map_file, 'wb') as f:
                pickle.dump(map_data, f)
            rospy.loginfo(f"Saved map data: {map_file}")
            self.save_terrain_visualizations(
                heightmap, terrain_info, env_dir, pointcloud=pcd,
                grid_map_data=grid_map_data)
        
        # These latched topics are diagnostics/backward compatibility only.
        # Planning synchronization is exclusively the service response above.
        self.pointcloud_pub.publish(ros_pointcloud)
        self.occ3d_pub.publish(occ_hwy_msg)
        self.occ_pub.publish(occ_msg)
        
        rospy.loginfo(f"Published terrain point cloud for {env_name}")
        rospy.loginfo(f"Map bounds: {grid_map_data.get('bounds', 'Unknown')}")
        rospy.loginfo(f"Map resolution: {grid_map_data.get('resolution', 'Unknown')}")
        rospy.loginfo(f"Map shape: {grid_map_tensor.shape}")
        
        # 更新地图生成时间戳，用于轨迹验证
        self.map_generation_timestamp = rospy.Time.now().to_sec()
        
        # 清理任何待处理的轨迹回调状态，防止旧地图的轨迹被保存到新地图
        self.finish_active_planning_attempt()
        self.cancel_timer('next_path_timer')
        self.current_trajectory = None
        self.path_retry_count = 0
        
        # 更新期望的环境和路径ID
        self.expected_env_id = self.current_env_id
        self.expected_path_id = self.current_path_id
        
        # 不在这里进行固定阻塞等待。调用方会使用单个可配置的短定时器；
        # 若地图确实尚未就绪，规划器会立即返回失败并由统一重试逻辑处理。
        
        if not (self.external_map_is_canonical
                and self.paths_generated_for_current_env > 0):
            self.current_path_id = 0
            self.paths_generated_for_current_env = 0
        self.current_path_profile_retry_round = 0
        self.scene_failed_attempts = 0
        self.expected_path_id = self.current_path_id

        return env_dir
    
    def save_terrain_visualizations(
            self, heightmap, terrain_info, env_dir, pointcloud=None,
            grid_map_data=None):
        """保存地形可视化图片"""
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        
        try:
            plt.ioff()  # 关闭交互模式
            
            if heightmap is not None:
                # 对于生成的地形，创建2D高度图
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(heightmap.T, origin='lower', 
                              extent=[-self.map_size/2, self.map_size/2, -self.map_size/2, self.map_size/2],
                              cmap='terrain', interpolation='bilinear')
                plt.colorbar(im, ax=ax, label='Height (m)')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title(f'Terrain Height Map\n(Seed: {terrain_info.get("rng_seed", "N/A")})')
                plt.tight_layout()
                plt.savefig(os.path.join(env_dir, 'terrain_2d.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)
                plt.clf()
            else:
                # 外部地图直接显示最终写入map.p的100x100高程栅格。这样
                # 图片与规划器/训练实际使用的数据一致，不会被点云抽样纹理误导。
                crop_info = terrain_info.get('crop', {})
                elevation = None
                if grid_map_data is not None:
                    elevation = grid_map_data.get('elevation')
                if elevation is None:
                    raise ValueError(
                        "External-map elevation grid is unavailable")
                elevation = np.asarray(elevation, dtype=np.float32)
                bounds = grid_map_data.get(
                    'bounds',
                    (-self.target_map_size / 2.0,
                     self.target_map_size / 2.0,
                     -self.target_map_size / 2.0,
                     self.target_map_size / 2.0))

                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(
                    elevation, origin='lower',
                    extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
                    cmap='terrain', interpolation='nearest',
                    aspect='equal')
                plt.colorbar(im, ax=ax, label='Height (m)')
                ax.set_xlabel('Local X (m)')
                ax.set_ylabel('Local Y (m)')
                ax.set_title(
                    'External Map Crop: saved 100x100 elevation grid\n'
                    f'raw source center='
                    f'{crop_info.get("center_in_raw_source_frame", "N/A")}, '
                    f'raw source yaw='
                    f'{crop_info.get("yaw_in_original_frame_deg", float("nan")):.2f} deg')
                plt.tight_layout()
                plt.savefig(os.path.join(env_dir, 'terrain_2d.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)
                plt.clf()
            
            # 3D可视化
            if heightmap is not None:
                # 对于生成的地形，创建3D表面图
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
                ax.set_title(f'3D Terrain View\n(Seed: {terrain_info.get("rng_seed", "N/A")})')
            else:
                # 外部地图绘制最终保存的规则栅格，而不是随机抽样原始点云。
                try:
                    if grid_map_data is None:
                        raise ValueError("Grid map is unavailable")
                    elevation = np.asarray(
                        grid_map_data.get('elevation'), dtype=np.float32)
                    if elevation.ndim != 2:
                        raise ValueError(
                            f"Invalid elevation shape: {elevation.shape}")
                    bounds = grid_map_data.get(
                        'bounds',
                        (-self.target_map_size / 2.0,
                         self.target_map_size / 2.0,
                         -self.target_map_size / 2.0,
                         self.target_map_size / 2.0))
                    resolution = float(grid_map_data.get(
                        'resolution', self.target_resolution))
                    rows, cols = elevation.shape
                    x = bounds[0] + (
                        np.arange(cols, dtype=np.float64) + 0.5) * resolution
                    y = bounds[2] + (
                        np.arange(rows, dtype=np.float64) + 0.5) * resolution
                    X, Y = np.meshgrid(x, y)
                    
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    surf = ax.plot_surface(
                        X, Y, elevation, cmap='terrain',
                        linewidth=0, antialiased=True, alpha=0.9)
                    ax.set_xlim(bounds[0], bounds[1])
                    ax.set_ylim(bounds[2], bounds[3])
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_zlabel('Height (m)')
                    ax.set_title(
                        '3D External Map Crop (saved 100x100 grid)')
                    plt.colorbar(surf, ax=ax, label='Height (m)')
                    
                except Exception as e:
                    rospy.logwarn(f"Failed to create 3D visualization for external map: {e}")
                    # 创建一个简单的说明图
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.text(0.5, 0.5, 'External Map\n(3D visualization unavailable)',
                            transform=ax.transAxes, fontsize=16, ha='center', va='center')
                    ax.set_title('3D External Map View')
                    ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(env_dir, 'terrain_3d.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
            
        except Exception as e:
            rospy.logerr(f"Failed to save terrain visualization: {e}")
        
        plt.ion()  # 重新启用交互模式
    
    def update_pose_generation_bounds(self, map_size):
        """
        更新位姿生成的边界范围
        
        Args:
            map_size: 地图尺寸（米）
        """
        self.effective_map_size = map_size
        
        # 为外部地图使用更保守的边界，确保位姿在安全区域内
        safety_margin = 2.0  # 安全边距
        # if self.use_external_map:
        #     # 对于外部地图，使用更大的安全边距
        #     safety_margin = 5.0
        
        self.map_x_min = -map_size / 2 + safety_margin
        self.map_x_max = map_size / 2 - safety_margin
        self.map_y_min = -map_size / 2 + safety_margin
        self.map_y_max = map_size / 2 - safety_margin
        
        # 计算有效区域的对角线长度
        effective_width = self.map_x_max - self.map_x_min
        effective_height = self.map_y_max - self.map_y_min
        max_possible_distance = math.sqrt(effective_width**2 + effective_height**2)
        self.max_pose_distance = max_possible_distance
        
        # 动态调整最小距离
        if self.use_external_map:
            # 对于外部地图，使用对角线长度的30%作为最小距离
            self.min_distance = max_possible_distance * 0.3
        else:
            # 对于生成地图，保持原来的逻辑
            self.min_distance = rospy.get_param('~min_distance', 2.0)
        
        rospy.loginfo(f"Updated pose generation bounds: X[{self.map_x_min:.1f}, {self.map_x_max:.1f}], Y[{self.map_y_min:.1f}, {self.map_y_max:.1f}]")
        rospy.loginfo(f"Updated min_distance: {self.min_distance:.1f}m (max possible: {max_possible_distance:.1f}m)")
    
    def generate_random_pose(self):
        """生成随机位姿"""
        x = random.uniform(self.map_x_min, self.map_x_max)
        y = random.uniform(self.map_y_min, self.map_y_max)
        yaw = random.uniform(-np.pi, np.pi)
        return (x, y, yaw)

    def prepare_stable_pose_candidates(self):
        """
        从当前稳定性图中预计算可用端点。

        只保留XY可通行区域中最大的连通分量，并且每个候选姿态对应的
        yaw bin 都必须稳定。这个检查与最终轨迹验证使用同一份 yaw_scores，
        因此不会放宽数据要求，只会在调用昂贵规划器前拒绝明显无效的端点。
        对外部点云，再排除明显高于拟合地面的原始回波所在单元，避免把
        树冠/树顶当成可站立的地面端点。
        """
        self.current_stable_pose_candidates = None
        if (not self.prefilter_stable_poses or
                self.current_yaw_scores is None or
                self.current_map_bounds is None or
                self.current_resolution is None):
            return

        scores = np.asarray(self.current_yaw_scores)
        if scores.ndim != 3:
            rospy.logwarn("Cannot prefilter poses: yaw stability map is not 3-D")
            return

        stable = np.isfinite(scores) & (scores > 0.5)
        height, width, _ = stable.shape
        min_x, _, min_y, _ = self.current_map_bounds
        resolution = float(self.current_resolution)
        x_centers = min_x + (np.arange(width) + 0.5) * resolution
        y_centers = min_y + (np.arange(height) + 0.5) * resolution
        inside_x = ((x_centers >= self.map_x_min) &
                    (x_centers <= self.map_x_max))
        inside_y = ((y_centers >= self.map_y_min) &
                    (y_centers <= self.map_y_max))
        allowed_xy = inside_y[:, None] & inside_x[None, :]

        def candidates_in_largest_component(stable_mask):
            traversable = np.any(stable_mask, axis=2) & allowed_xy
            labels, component_count = connected_component_label(
                traversable, structure=np.ones((3, 3), dtype=np.uint8))
            if component_count == 0:
                return None, 0, 0
            component_sizes = np.bincount(labels.ravel())
            component_sizes[0] = 0
            largest_label = int(np.argmax(component_sizes))
            largest_xy = labels == largest_label
            candidates = np.argwhere(
                stable_mask & largest_xy[:, :, None])
            return candidates, int(component_sizes[largest_label]), int(
                np.count_nonzero(traversable))

        filtered_stable = stable
        obstacle_mask = self.current_endpoint_obstacle_mask
        obstacle_mask_applied = (
            obstacle_mask is not None and
            obstacle_mask.shape == stable.shape[:2])
        if obstacle_mask_applied:
            filtered_stable = stable & ~obstacle_mask[:, :, None]

        candidates, largest_size, traversable_count = (
            candidates_in_largest_component(filtered_stable))
        if candidates is None or len(candidates) < 2:
            if obstacle_mask_applied:
                # The obstacle mask is an endpoint prefilter, not a map
                # acceptance gate. Keep the manually accepted map usable if a
                # future threshold/configuration removes all stable cells.
                rospy.logwarn(
                    "Endpoint obstacle mask left fewer than two stable "
                    "candidates; using the unmasked stable surface for this "
                    "map")
                obstacle_mask_applied = False
                candidates, largest_size, traversable_count = (
                    candidates_in_largest_component(stable))
        if candidates is None or len(candidates) < 2:
            rospy.logwarn(
                "Stable-pose prefilter found fewer than two candidates")
            return

        self.current_stable_pose_candidates = candidates.astype(
            np.int32, copy=False)
        rospy.loginfo(
            "Stable-pose prefilter prepared %d poses in largest component "
            "(%d/%d traversable cells, obstacle mask=%s)",
            len(candidates), largest_size, traversable_count,
            "on" if obstacle_mask_applied else "off")

    def prepare_endpoint_obstacle_mask(self, pointcloud, grid_map_data):
        """Mark cells containing returns clearly above the fitted surface.

        The planner PCD contains both the dense fitted ground surface and the
        retained raw returns. Comparing every return with the saved fitted
        elevation makes canopy/top returns visible without relying on LAS
        classifications, while low returns/noise do not enter this mask.
        """
        self.current_endpoint_obstacle_mask = None
        self.current_endpoint_obstacle_stats = None
        if not self.use_external_map or pointcloud is None:
            return

        elevation = np.asarray(
            grid_map_data.get('elevation'), dtype=np.float64)
        bounds = grid_map_data.get('bounds')
        resolution = float(grid_map_data.get('resolution', 0.0))
        points = np.asarray(pointcloud.points, dtype=np.float64)
        if (elevation.ndim != 2 or len(points) == 0 or
                bounds is None or len(bounds) != 4 or resolution <= 0.0):
            rospy.logwarn(
                "Cannot prepare endpoint obstacle mask: incomplete external "
                "point cloud/grid data")
            return

        min_x, _, min_y, _ = (float(value) for value in bounds)
        columns = (points[:, 0] - min_x) / resolution - 0.5
        rows = (points[:, 1] - min_y) / resolution - 0.5
        inside = (
            np.isfinite(points).all(axis=1) &
            (columns >= 0.0) &
            (columns <= elevation.shape[1] - 1) &
            (rows >= 0.0) &
            (rows <= elevation.shape[0] - 1))
        fitted_z = np.full(len(points), np.nan, dtype=np.float64)
        if np.any(inside):
            fitted_z[inside] = map_coordinates(
                elevation,
                np.vstack((rows[inside], columns[inside])),
                order=1,
                mode='nearest')
        residual = points[:, 2] - fitted_z
        above = np.isfinite(residual) & (
            residual > self.external_above_surface_height)

        obstacle_mask = np.zeros(elevation.shape, dtype=bool)
        if np.any(above):
            ix = np.floor((points[above, 0] - min_x) / resolution).astype(
                np.int64)
            iy = np.floor((points[above, 1] - min_y) / resolution).astype(
                np.int64)
            in_cells = (
                (ix >= 0) & (ix < obstacle_mask.shape[1]) &
                (iy >= 0) & (iy < obstacle_mask.shape[0]))
            obstacle_mask[iy[in_cells], ix[in_cells]] = True

        self.current_endpoint_obstacle_mask = obstacle_mask
        self.current_endpoint_obstacle_stats = {
            'height_threshold_m': float(self.external_above_surface_height),
            'point_count': int(len(points)),
            'above_surface_point_count': int(np.count_nonzero(above)),
            'occupied_cell_count': int(np.count_nonzero(obstacle_mask)),
            'total_cell_count': int(obstacle_mask.size),
            'occupied_cell_fraction': float(obstacle_mask.mean()),
        }
        rospy.loginfo(
            "Endpoint obstacle mask: returns>%0.2fm=%d, cells=%d/%d "
            "(%0.1f%%)",
            self.external_above_surface_height,
            self.current_endpoint_obstacle_stats['above_surface_point_count'],
            self.current_endpoint_obstacle_stats['occupied_cell_count'],
            self.current_endpoint_obstacle_stats['total_cell_count'],
            100.0 * self.current_endpoint_obstacle_stats[
                'occupied_cell_fraction'])

    def pose_from_stable_candidate(self, candidate):
        """把 (iy, ix, yaw_bin) 候选项转换为栅格内部的连续位姿。"""
        iy, ix, yaw_index = (int(candidate[0]), int(candidate[1]),
                             int(candidate[2]))
        min_x, _, min_y, _ = self.current_map_bounds
        resolution = float(self.current_resolution)
        # 在同一个栅格内部抖动，增加数据多样性且不改变稳定性索引。
        x = min_x + (ix + random.uniform(0.05, 0.95)) * resolution
        y = min_y + (iy + random.uniform(0.05, 0.95)) * resolution
        yaw_bins = self.current_yaw_scores.shape[2]
        bin_width = 2.0 * math.pi / float(yaw_bins)
        yaw = (-math.pi + (yaw_index + 0.5) * bin_width +
               random.uniform(-0.45, 0.45) * bin_width)
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
    
    def pose_sampling_profile_for_current_path(self):
        """
        为当前路径确定距离和朝向难度档位。

        初始档位由环境、阶段和path id确定，因此普通重试不会改变数据配额。
        canonical 地图上的某条路径连续失败后，调用方只为这条路径增加
        retry round，逐级选择更容易的档位；成功后下一条路径恢复原始配额。
        两个互质排列让距离与难度档位彼此解耦。
        """
        phase_offset = 17 if self.current_phase == 'val' else 0
        distance_slot = (
            self.current_path_id * 37 +
            self.current_env_id * 13 + phase_offset) % 100 / 100.0
        complexity_slot = (
            self.current_path_id * 53 +
            self.current_env_id * 29 + phase_offset) % 100 / 100.0

        if distance_slot < self.long_distance_fraction:
            base_distance_class = 'long'
        elif distance_slot < (
                self.long_distance_fraction +
                self.medium_distance_fraction):
            base_distance_class = 'medium'
        else:
            base_distance_class = 'short'

        if complexity_slot < self.high_complexity_fraction:
            base_complexity_class = 'high'
        elif complexity_slot < (
                self.high_complexity_fraction +
                self.moderate_complexity_fraction):
            base_complexity_class = 'moderate'
        else:
            base_complexity_class = 'simple'

        retry_round = max(0, int(self.current_path_profile_retry_round))
        distance_fallbacks = {
            'long': ('long', 'medium', 'short'),
            'medium': ('medium', 'short'),
            'short': ('short',),
        }
        complexity_fallbacks = {
            'high': ('high', 'moderate', 'simple'),
            'moderate': ('moderate', 'simple'),
            'simple': ('simple',),
        }
        distance_class = distance_fallbacks[base_distance_class][min(
            retry_round, len(distance_fallbacks[base_distance_class]) - 1)]
        complexity_class = complexity_fallbacks[base_complexity_class][min(
            retry_round, len(complexity_fallbacks[base_complexity_class]) - 1)]

        max_distance = float(self.max_pose_distance)
        distance_ratio_ranges = {
            'short': (0.30, 0.50),
            'medium': (0.50, 0.70),
            'long': (0.70, 0.94),
        }
        ratio_min, ratio_max = distance_ratio_ranges[distance_class]
        distance_min = max(self.min_distance, ratio_min * max_distance)
        distance_max = ratio_max * max_distance

        # 两端朝向相对起点->终点连线的误差之和。实测该指标与轨迹
        # 曲折率的相关系数约0.86，因而可在调用昂贵规划器前控制复杂度。
        complexity_ranges = {
            'simple': (0.0, 2.0),
            'moderate': (2.0, 4.0),
            'high': (4.0, 2.0 * math.pi + 1e-6),
        }
        mismatch_min, mismatch_max = complexity_ranges[complexity_class]
        return {
            'distance_class': distance_class,
            'complexity_class': complexity_class,
            'base_distance_class': base_distance_class,
            'base_complexity_class': base_complexity_class,
            'profile_retry_round': retry_round,
            'profile_relaxed': bool(retry_round > 0),
            'target_distance_range': (distance_min, distance_max),
            'target_heading_mismatch_range_rad': (
                mismatch_min, mismatch_max),
        }

    @staticmethod
    def pose_pair_sampling_metrics(start, target):
        """计算端点直线距离和两端相对直连方向的朝向误差之和。"""
        delta_x = target[0] - start[0]
        delta_y = target[1] - start[1]
        distance = math.hypot(delta_x, delta_y)
        bearing = math.atan2(delta_y, delta_x)

        def angular_error(angle):
            return abs(math.atan2(
                math.sin(angle - bearing), math.cos(angle - bearing)))

        mismatch_sum = angular_error(start[2]) + angular_error(target[2])
        return distance, mismatch_sum

    @staticmethod
    def interval_penalty(value, lower, upper):
        """数值落在闭区间内返回0，否则返回到最近边界的距离。"""
        if value < lower:
            return lower - value
        if value > upper:
            return value - upper
        return 0.0

    def generate_pose_pair(self):
        """
        按固定配额生成距离、朝向难度均满足要求的稳定端点对。

        默认比例为20%短/40%中/40%长和20%简单/40%中等/40%高难。
        这里只增加低成本端点筛选，后续规划和逐点稳定性验证保持严格。
        """
        candidates = self.current_stable_pose_candidates
        profile = self.pose_sampling_profile_for_current_path()
        distance_min, distance_max = profile['target_distance_range']
        mismatch_min, mismatch_max = profile[
            'target_heading_mismatch_range_rad']
        best_pair = None
        best_penalty = float('inf')

        for attempt in range(1, self.pose_sampling_max_attempts + 1):
            if candidates is not None:
                indices = np.random.randint(0, len(candidates), size=2)
                start = self.pose_from_stable_candidate(candidates[indices[0]])
                target = self.pose_from_stable_candidate(
                    candidates[indices[1]])
            else:
                start = self.generate_random_pose()
                target = self.generate_random_pose()

            distance, mismatch_sum = self.pose_pair_sampling_metrics(
                start, target)
            distance_penalty = self.interval_penalty(
                distance, distance_min, distance_max)
            mismatch_penalty = self.interval_penalty(
                mismatch_sum, mismatch_min, mismatch_max)
            # 归一化后记录最接近目标档位的稳定端点对，作为极少数采样
            # 不收敛时的保底；不会回退到固定的中短距离位姿。
            penalty = (
                distance_penalty / max(self.max_pose_distance, 1e-6) +
                mismatch_penalty / (2.0 * math.pi))
            if penalty < best_penalty:
                best_penalty = penalty
                best_pair = (start, target, distance, mismatch_sum)

            if distance_penalty == 0.0 and mismatch_penalty == 0.0:
                profile.update({
                    'endpoint_distance': distance,
                    'heading_mismatch_sum_rad': mismatch_sum,
                    'sampling_attempt': attempt,
                    'strict_profile_match': True,
                })
                self.current_pose_sampling_profile = profile
                rospy.loginfo(
                    "Pose sampling profile: distance=%s %.2fm "
                    "(target %.2f-%.2fm), complexity=%s %.1fdeg "
                    "(attempt %d)",
                    profile['distance_class'], distance,
                    distance_min, distance_max,
                    profile['complexity_class'],
                    math.degrees(mismatch_sum), attempt)
                return start, target

        start, target, distance, mismatch_sum = best_pair
        profile.update({
            'endpoint_distance': distance,
            'heading_mismatch_sum_rad': mismatch_sum,
            'sampling_attempt': self.pose_sampling_max_attempts,
            'strict_profile_match': False,
        })
        self.current_pose_sampling_profile = profile
        rospy.logwarn(
            "Could not exactly match pose profile after %d cheap attempts; "
            "using closest stable pair: distance=%s %.2fm, complexity=%s "
            "%.1fdeg, normalized penalty=%.4f",
            self.pose_sampling_max_attempts,
            profile['distance_class'], distance,
            profile['complexity_class'], math.degrees(mismatch_sum),
            best_penalty)
        return start, target

    def cancel_timer(self, attribute):
        """取消并清空一个 rospy.Timer；重复调用是安全的。"""
        if attribute == 'next_path_timer':
            # shutdown 无法撤回已经排进回调队列的事件，用代次使其逻辑失效。
            self.next_path_schedule_id += 1
        timer = getattr(self, attribute, None)
        setattr(self, attribute, None)
        if timer is not None:
            try:
                timer.shutdown()
            except Exception:
                pass

    def finish_active_planning_attempt(self):
        """结束当前规划尝试，并确保其超时定时器不会影响后续尝试。"""
        self.cancel_timer('planning_timeout_timer')
        self.active_planning_attempt_id = None
        self.waiting_for_result = False

    def finish_attempt_record(self, outcome, **details):
        if self.active_attempt_record is None:
            return
        self.active_attempt_record.update(
            outcome=outcome,
            finished_utc=datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            **self._json_safe(details))
        self.attempt_records.append(self.active_attempt_record)
        self.active_attempt_record = None
        self.write_experiment_manifest()

    def schedule_next_path(self, delay=None, reason=""):
        """保证任意时刻最多只有一个待执行的路径生成定时器。"""
        if delay is None:
            delay = self.publish_delay
        self.cancel_timer('next_path_timer')
        schedule_id = self.next_path_schedule_id
        expected_state = (
            self.current_env_id, self.current_phase, self.current_path_id)

        def callback(event):
            if schedule_id != self.next_path_schedule_id:
                rospy.logdebug(
                    "Ignoring stale next-path timer generation %d (active=%d)",
                    schedule_id, self.next_path_schedule_id)
                return
            self.next_path_timer = None
            current_state = (
                self.current_env_id, self.current_phase, self.current_path_id)
            if current_state != expected_state:
                rospy.logdebug(
                    "Ignoring stale next-path timer: expected=%s current=%s",
                    expected_state, current_state)
                return
            self.generate_next_path(event)

        # rospy.Timer cannot represent a zero period. A 1 ms one-shot keeps
        # the callback asynchronous while allowing map_ready_delay=0 now that
        # the synchronous map-update service is the readiness barrier.
        timer_delay = max(1e-3, float(delay))
        self.next_path_timer = rospy.Timer(
            rospy.Duration(timer_delay), callback, oneshot=True)
        if reason:
            rospy.logdebug("Scheduled next path in %.3fs (%s)", delay, reason)

    def current_phase_path_target(self):
        return (self.train_paths_per_env if self.current_phase == 'train'
                else self.val_paths_per_env)

    def canonical_env_skip_status(self, env_dir):
        """Return why a canonical env should not be planned, or None."""
        if os.path.isfile(os.path.join(env_dir, "needs_return.json")):
            return "needs_return"
        existing = len(glob.glob(os.path.join(env_dir, "path_*.p")))
        if existing >= self.current_phase_path_target():
            return "complete"
        if self.mark_unplannable_canonical and existing >= 1:
            return "already_tested"
        return None

    def current_env_dir(self):
        env_name = f"env{self.current_env_id:06d}"
        phase_dir = (
            self.train_dir if self.current_phase == 'train' else self.val_dir)
        return os.path.join(phase_dir, env_name)

    def mark_canonical_needs_return(self, reason):
        """Write a skip marker so 8765 can ask for a human return."""
        env_dir = self.current_env_dir()
        os.makedirs(env_dir, exist_ok=True)
        payload = {
            "needs_return": True,
            "reason": reason,
            "split": self.current_phase,
            "env_id": int(self.current_env_id),
            "map_path": self.external_map_path,
            "paths_saved": int(self.paths_generated_for_current_env),
            "scene_failed_attempts": int(self.scene_failed_attempts),
            "path_retries": int(self.path_retry_count),
        }
        marker = os.path.join(env_dir, "needs_return.json")
        with open(marker, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        map_path = self.external_map_path
        if map_path:
            meta_path = os.path.splitext(map_path)[0] + ".json"
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as stream:
                    metadata = json.load(stream)
                metadata["needs_return"] = True
                metadata["needs_return_reason"] = reason
                with open(meta_path, "w", encoding="utf-8") as stream:
                    json.dump(metadata, stream, indent=2, ensure_ascii=False)
                    stream.write("\n")

    def clear_canonical_needs_return(self, env_dir=None):
        env_dir = env_dir or self.current_env_dir()
        marker = os.path.join(env_dir, "needs_return.json")
        if os.path.isfile(marker):
            try:
                os.remove(marker)
            except OSError:
                pass
        map_path = getattr(self, "external_map_path", None)
        if not map_path:
            return
        meta_path = os.path.splitext(map_path)[0] + ".json"
        if not os.path.isfile(meta_path):
            return
        with open(meta_path, encoding="utf-8") as stream:
            metadata = json.load(stream)
        if not metadata.get("needs_return"):
            return
        metadata["needs_return"] = False
        metadata.pop("needs_return_reason", None)
        with open(meta_path, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)
            stream.write("\n")

    def finish_current_phase(self):
        """Move train→val or env→next after this phase is done or skipped."""
        if self.current_phase == 'train':
            if self.stop_after_train:
                self.finish_active_planning_attempt()
                self.current_trajectory = None
                self.experiment_status = 'completed'
                self.write_experiment_manifest()
                rospy.loginfo(
                    "Train-only generation completed! Retry statistics: %s",
                    self.retry_statistics)
                rospy.signal_shutdown("train-only generation completed")
                return
            self.finish_active_planning_attempt()
            self.current_trajectory = None
            self.current_phase = 'val'
            self.current_path_id = 0
            self.paths_generated_for_current_env = 0
            self.current_path_profile_retry_round = 0
            self.path_retry_count = 0
            self.scene_failed_attempts = 0
            self.generate_new_environment()
            if self._env_skip_status:
                self.finish_current_phase()
                return
            self.schedule_next_path(
                delay=self.map_ready_delay,
                reason="independent validation environment ready")
            return

        self.finish_active_planning_attempt()
        self.current_trajectory = None
        self.current_env_id += 1
        self.current_phase = 'train'
        self.current_path_id = 0
        self.paths_generated_for_current_env = 0
        self.current_path_profile_retry_round = 0
        self.path_retry_count = 0
        self.scene_failed_attempts = 0
        if self.current_env_id >= self.start_env_id + self.num_environments:
            self.experiment_status = 'completed'
            self.write_experiment_manifest()
            rospy.loginfo(
                "Dataset generation completed! Retry statistics: %s",
                self.retry_statistics)
            rospy.signal_shutdown("Dataset generation completed")
            return
        self.cancel_timer('next_path_timer')
        rospy.Timer(rospy.Duration(1.0), self.start_new_environment, oneshot=True)

    def skip_unplannable_canonical_map(self):
        """Stop retrying a canonical map that produced no trajectory."""
        self.retry_statistics['canonical_map_rejected'] += 1
        reason = "no_valid_trajectory"
        rospy.logwarn(
            "Canonical %s env%06d still has 0 trajectories after %d "
            "start-goal attempts; marking for human return and skipping",
            self.current_phase, self.current_env_id,
            self.scene_failed_attempts)
        self.finish_active_planning_attempt()
        self.cancel_timer('next_path_timer')
        self.current_trajectory = None
        self.mark_canonical_needs_return(reason)
        self.write_experiment_manifest()
        self.finish_current_phase()

    def regenerate_if_retry_limit_reached(self):
        """
        当前裁剪区域长期无法生成合格路径时，重新裁剪整个环境。

        这不会放宽轨迹要求，也不会保存失败结果；它只是避免在一个几乎
        不可规划的裁剪上无限消耗时间。
        """
        limit = self.max_path_retries_before_regenerate
        if limit <= 0 or self.path_retry_count < limit:
            return False

        if self.external_map_is_canonical:
            if (self.mark_unplannable_canonical
                    and self.paths_generated_for_current_env == 0):
                # Test generation only: this scene swapped start/goal pairs
                # for a full retry budget and still saved nothing.
                self.skip_unplannable_canonical_map()
                return True
            # The scene can produce trajectories. Keep replacing failed
            # pairs and do not skip it later.
            self.retry_statistics['canonical_path_resampled'] += 1
            self.current_path_profile_retry_round = min(
                self.current_path_profile_retry_round + 1, 2)
            rospy.loginfo(
                "Canonical env%06d already has %d trajectories; "
                "continuing after %d failed pairs on path_%d",
                self.current_env_id, self.paths_generated_for_current_env,
                self.path_retry_count, self.current_path_id)
            self.finish_active_planning_attempt()
            self.cancel_timer('next_path_timer')
            self.current_trajectory = None
            self.path_retry_count = 0
            self.current_pose_sampling_profile = None
            self.write_experiment_manifest()
            self.schedule_next_path(
                delay=self.map_ready_delay,
                reason="canonical scene kept after a failed pair batch")
            return True

        self.retry_statistics['environment_regenerated'] += 1
        rospy.logwarn(
            "Path_%d failed %d consecutive attempts; regenerating "
            "env%06d with a new crop instead of retrying indefinitely",
            self.current_path_id, self.path_retry_count, self.current_env_id)
        self.finish_active_planning_attempt()
        self.cancel_timer('next_path_timer')
        rospy.Timer(
            rospy.Duration(self.publish_delay),
            self.regenerate_current_environment, oneshot=True)
        return True
    
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
            self.retry_statistics['planning_failed'] += 1
            self.path_retry_count += 1
            self.scene_failed_attempts += 1
            self.finish_attempt_record('planning_failed')
            rospy.logwarn(
                "Planning failed for path_%d (retry %d); sampling a new "
                "prevalidated pose pair",
                self.current_path_id, self.path_retry_count)
            self.finish_active_planning_attempt()
            if self.regenerate_if_retry_limit_reached():
                return
            self.schedule_next_path(reason="planning failed")
    
    def map_regeneration_callback(self, msg):
        """地图重新生成请求回调函数"""
        if not msg.data:
            rospy.loginfo("Received map regeneration request with data=False, ignoring")
            return
        
        rospy.logwarn(f"Received map regeneration request for env{self.current_env_id:06d}")
        rospy.logwarn(f"Current state: waiting_for_result={self.waiting_for_result}, current_path_id={self.current_path_id}")
        
        # 清理所有待处理状态，防止旧轨迹干扰
        self.finish_active_planning_attempt()
        self.cancel_timer('next_path_timer')
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
        if traj_points is None or len(traj_points) == 0:
            rospy.logwarn(f"Invalid or empty trajectory for path_{self.current_path_id}, retrying...")
            self.retry_statistics['empty_trajectory'] += 1
            self.path_retry_count += 1
            self.scene_failed_attempts += 1
            self.finish_attempt_record('empty_trajectory')
            self.finish_active_planning_attempt()
            if self.regenerate_if_retry_limit_reached():
                return
            self.schedule_next_path(reason="empty trajectory")
            return

        # 保存前使用与 MPT 完全一致的连续 stability hard gate。planner 和
        # pose prefilter 仍可使用原 binary yaw map，不参与此最终验收。
        invalid_found = False
        unstable_count = 0
        total_points = len(traj_points)
        
        # 缺少最终验收 ESDF 时直接拒绝，不能回退到 floor+binary 判定。
        if (self.current_trajectory_stability_esdf is None or
                self.current_map_bounds is None or
                self.current_resolution is None):
            rospy.logwarn(
                "Rejecting trajectory because MPT stability ESDF or map "
                "bounds/resolution is unavailable")
            invalid_found = True
        else:
            validation = validate_trajectory_stability(
                traj_points,
                self.current_trajectory_stability_esdf,
                self.current_map_bounds,
                self.current_resolution,
                d_safe=self.trajectory_stability_d_safe,
            )
            invalid_found = not validation['valid']
            unstable_count = validation['invalid_count']
            rospy.loginfo(
                "Validating %d trajectory points against continuous MPT "
                "stability ESDF %s (d_safe=%.3fm)",
                total_points,
                self.current_trajectory_stability_esdf.shape,
                self.trajectory_stability_d_safe)
            if invalid_found:
                point_idx = validation['first_invalid_index']
                reason = validation['first_invalid_reason']
                margin = (
                    float('nan') if point_idx is None
                    else float(validation['margins'][point_idx]))
                point = (
                    [float('nan')] * 3 if point_idx is None
                    else traj_points[point_idx])
                rospy.logwarn(
                    "Point %s fails continuous stability validation: "
                    "reason=%s pose=[%.3f,%.3f,%.3f] margin=%.4fm "
                    "required=%.4fm",
                    point_idx, reason,
                    float(point[0]), float(point[1]), float(point[2]),
                    margin, self.trajectory_stability_d_safe)
            else:
                rospy.loginfo(
                    "Continuous stability minimum margin: %.4fm",
                    validation['minimum_margin'])
                if self.active_attempt_record is not None:
                    self.active_attempt_record['minimum_stability_margin_m'] = (
                        float(validation['minimum_margin']))

        # 输出验证结果统计
        if not invalid_found:
            rospy.loginfo(
                f"Trajectory validation PASSED: {total_points} points, "
                "all margins satisfy d_safe")
        else:
            rospy.logwarn(
                "Trajectory validation FAILED: unstable=%d/%d",
                unstable_count, total_points)
            
        if invalid_found:
            self.retry_statistics['trajectory_unstable'] += 1
            self.path_retry_count += 1
            self.scene_failed_attempts += 1
            self.finish_attempt_record(
                'trajectory_unstable', unstable_points=unstable_count,
                trajectory_points=total_points)
            rospy.logwarn(
                "Trajectory for path_%d failed hard validation (retry %d); "
                "sampling another pair",
                self.current_path_id, self.path_retry_count)
            self.finish_active_planning_attempt()
            if self.regenerate_if_retry_limit_reached():
                return
            # 继续生成新的路径尝试
            self.schedule_next_path(reason="unstable trajectory")
            return

        # 验证通过，接受轨迹并继续处理
        rospy.loginfo(f"Trajectory validation successful, saving path_{self.current_path_id}")
        self.current_trajectory = msg
        self.finish_active_planning_attempt()
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
    
    def fit_spline_curve(self, points, num_control_points=25):
        """
        使用样条曲线拟合路径点，返回固定数量的控制点
        
        样条插值的优势：
        - 保证经过所有控制点
        - 可导性好，适合梯度优化
        - 曲线平滑连续
        
        Args:
            points: 原始路径点，形状为(N, 3)，包含[x, y, yaw]
            num_control_points: 样条曲线的控制点数量，建议20-30
        
        Returns:
            样条控制点，形状为(num_control_points, 3)
        """
        if points is None or len(points) < 2:
            rospy.logwarn("Insufficient points for spline fitting")
            return None
        
        n_points = len(points)
        
        # 如果原始点数小于等于目标控制点数，直接返回原始点
        if n_points <= num_control_points:
            rospy.loginfo(f"Number of input points ({n_points}) <= target control points ({num_control_points}), returning original points")
            return points
        
        rospy.loginfo(f"Fitting spline curve: {n_points} points -> {num_control_points} control points")
        
        try:
            # 分开处理xy坐标和yaw角
            xy_points = points[:, :2]
            yaw_points = points[:, 2]
            
            # 对xy坐标进行样条拟合
            xy_control_points = self._fit_spline_xy(xy_points, num_control_points)
            
            # 对yaw角进行周期性感知的样条插值
            yaw_control_points = self._fit_spline_yaw(yaw_points, num_control_points)
            
            # 组合xy和yaw
            control_points = np.column_stack([xy_control_points, yaw_control_points])
            
            rospy.loginfo(f"Spline fitting completed: control points shape = {control_points.shape}")
            return control_points
            
        except Exception as e:
            rospy.logwarn(f"Spline fitting failed with error: {e}, using fallback uniform sampling")
            # 降级到均匀采样
            indices = np.linspace(0, n_points - 1, num_control_points).astype(int)
            return points[indices]
    
    def _fit_spline_xy(self, xy_points, num_control_points):
        """
        使用样条插值对xy坐标进行拟合
        
        方法：基于路径长度参数化，使用自然三次样条选择控制点
        
        Args:
            xy_points: 原始路径的xy坐标，形状为(N, 2)
            num_control_points: 目标控制点数量
        
        Returns:
            样条控制点的xy坐标，形状为(num_control_points, 2)
        """
        n = len(xy_points)
        
        # 计算累积路径长度作为参数
        distances = np.sqrt(np.sum(np.diff(xy_points, axis=0)**2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        total_distance = cumulative_distances[-1]
        
        if total_distance < 1e-6:
            # 路径太短，直接均匀采样索引
            indices = np.linspace(0, n - 1, num_control_points).astype(int)
            return xy_points[indices]
        
        # 归一化到[0, 1]作为参数t
        t_original = cumulative_distances / total_distance
        
        # 使用自定义的样条插值（与grad_optimizer.py一致）
        t_new = np.linspace(0, 1, num_control_points)
        
        # 对x和y分别进行样条插值
        x_new = self._evaluate_scalar_spline(xy_points[:, 0], t_new, t_original)
        y_new = self._evaluate_scalar_spline(xy_points[:, 1], t_new, t_original)
        
        return np.column_stack([x_new, y_new])
    
    def _fit_spline_yaw(self, yaw_points, num_control_points):
        """
        对yaw角进行周期性感知的样条插值
        
        Args:
            yaw_points: 原始的yaw角数组，形状为(N,)
            num_control_points: 目标控制点数量
        
        Returns:
            插值后的yaw角，形状为(num_control_points,)
        """
        n = len(yaw_points)
        
        # 创建参数t（均匀参数化）
        t_original = np.linspace(0, 1, n)
        t_new = np.linspace(0, 1, num_control_points)
        
        # 使用自定义的yaw样条插值（与grad_optimizer.py一致）
        yaw_interpolated = self._evaluate_yaw_spline(yaw_points, t_new, t_original)
        
        return yaw_interpolated
    
    def _solve_natural_cubic_M(self, y_values, t_control):
        """
        求解自然三次样条的二阶导数 M
        使用三对角矩阵算法（Thomas algorithm）
        
        Args:
            y_values: 控制点的y值，形状为(N,)
            t_control: 控制点的参数，形状为(N,)
        
        Returns:
            二阶导数M，形状为(N,)
        """
        N = len(y_values)
        if N < 2:
            return np.zeros(N)
        
        # 构建三对角系统 A*M = b
        # 自然边界条件：M[0] = M[-1] = 0
        h = np.diff(t_control)  # 根据实际参数计算间隔
        h = np.clip(h, 1e-6, None)  # 避免除零
        
        # 构建对角线
        diag = 2 * (h[:-1] + h[1:])
        diag = np.concatenate([[1], diag, [1]])  # 边界条件
        
        # 构建上下对角线
        upper = np.concatenate([[0], h[1:], [0]])
        lower = np.concatenate([[0], h[:-1], [0]])
        
        # 构建右侧向量
        b = np.zeros(N)
        for i in range(1, N - 1):
            b[i] = 6 * ((y_values[i + 1] - y_values[i]) / h[i] - 
                        (y_values[i] - y_values[i - 1]) / h[i - 1])
        
        # 边界条件（自然样条）
        b[0] = 0
        b[-1] = 0
        
        # 使用 Thomas 算法求解三对角系统
        M = self._solve_tridiagonal(lower, diag, upper, b)
        
        return M
    
    def _solve_tridiagonal(self, lower, diag, upper, b):
        """
        使用 Thomas 算法求解三对角线性系统
        
        Args:
            lower: 下对角线
            diag: 主对角线
            upper: 上对角线
            b: 右侧向量
        
        Returns:
            解向量 x
        """
        N = len(b)
        c_prime = np.zeros(N - 1)
        d_prime = np.zeros(N)
        x = np.zeros(N)
        
        # 前向消元
        c_prime[0] = upper[0] / diag[0]
        d_prime[0] = b[0] / diag[0]
        
        for i in range(1, N - 1):
            denom = diag[i] - lower[i] * c_prime[i - 1]
            c_prime[i] = upper[i] / denom
            d_prime[i] = (b[i] - lower[i] * d_prime[i - 1]) / denom
        
        d_prime[-1] = (b[-1] - lower[-1] * d_prime[-2]) / (diag[-1] - lower[-1] * c_prime[-2])
        
        # 回代
        x[-1] = d_prime[-1]
        for i in range(N - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]
        
        return x
    
    def _evaluate_scalar_spline(self, y_control, t_eval, t_control):
        """
        对一维标量序列进行三次样条插值
        
        Args:
            y_control: 控制点的y值，形状为(N,)
            t_eval: 评估点的参数，形状为(M,)
            t_control: 控制点的参数，形状为(N,)
        
        Returns:
            插值后的y值，形状为(M,)
        """
        N = len(y_control)
        M_values = self._solve_natural_cubic_M(y_control, t_control)  # 传入 t_control
        
        h = np.diff(t_control)
        h = np.clip(h, 1e-6, None)  # 避免除零
        
        # 找到每个评估点所在的区间
        idx = np.searchsorted(t_control[1:], t_eval, side='left')
        idx = np.clip(idx, 0, N - 2)
        
        # 获取区间端点
        t_k = t_control[idx]
        t_k1 = t_control[idx + 1]
        h_k = h[idx]
        dt = t_eval - t_k
        
        y_k = y_control[idx]
        y_k1 = y_control[idx + 1]
        M_k = M_values[idx]
        M_k1 = M_values[idx + 1]
        
        # 三次样条插值公式
        term1 = M_k * (t_k1 - t_eval)**3 / (6 * h_k)
        term2 = M_k1 * dt**3 / (6 * h_k)
        term3 = (y_k - M_k * h_k**2 / 6) * (t_k1 - t_eval) / h_k
        term4 = (y_k1 - M_k1 * h_k**2 / 6) * dt / h_k
        
        S = term1 + term2 + term3 + term4
        
        return S
    
    def _evaluate_yaw_spline(self, yaw_control, t_eval, t_control):
        """
        对yaw角进行周期性感知的三次样条插值
        
        Args:
            yaw_control: 控制点的yaw角，形状为(N,)
            t_eval: 评估点的参数，形状为(M,)
            t_control: 控制点的参数，形状为(N,)
        
        Returns:
            插值后的yaw角，形状为(M,)，规范化到[-pi, pi]
        """
        # 展开角度序列，消除周期性跳跃
        yaw_unwrapped = self._unwrap_angles(yaw_control)
        
        # 对展开后的角度进行标量样条插值
        yaw_interpolated_unwrapped = self._evaluate_scalar_spline(yaw_unwrapped, t_eval, t_control)
        
        # 将插值结果重新规范化到 [-π, π]
        yaw_interpolated = np.arctan2(np.sin(yaw_interpolated_unwrapped), 
                                       np.cos(yaw_interpolated_unwrapped))
        
        return yaw_interpolated
    
    def _unwrap_angles(self, angles):
        """
        展开角度序列，消除周期性跳跃
        
        Args:
            angles: 角度序列，形状为(N,)
        
        Returns:
            展开后的角度序列，形状为(N,)
        """
        if len(angles) <= 1:
            return angles.copy()
        
        unwrapped = np.zeros_like(angles)
        unwrapped[0] = angles[0]
        
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i-1]
            # 规范化角度差到 [-π, π]
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            unwrapped[i] = unwrapped[i-1] + diff
        
        return unwrapped
    
    def _fit_bezier_xy_stable(self, xy_points, num_control_points):
        """
        使用改进的算法拟合贝塞尔曲线的xy坐标，提高数值稳定性
        
        方法：使用分段低阶贝塞尔曲线，而不是一条高阶曲线
        
        Args:
            xy_points: 原始路径的xy坐标，形状为(N, 2)
            num_control_points: 目标控制点数量
        
        Returns:
            贝塞尔控制点的xy坐标，形状为(num_control_points, 2)
        """
        n = len(xy_points)
        
        # 方法1：直接使用基于路径长度的均匀采样作为控制点
        # 这比高阶贝塞尔拟合更稳定，同时保持路径形状
        
        # 计算累积路径长度
        distances = np.sqrt(np.sum(np.diff(xy_points, axis=0)**2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        total_distance = cumulative_distances[-1]
        
        if total_distance < 1e-6:
            # 路径太短，直接均匀采样索引
            indices = np.linspace(0, n - 1, num_control_points).astype(int)
            return xy_points[indices]
        
        # 生成目标距离点
        target_distances = np.linspace(0, total_distance, num_control_points)
        
        # 对每个目标距离进行插值
        control_points = []
        for target_dist in target_distances:
            # 使用numpy的插值函数
            x_interp = np.interp(target_dist, cumulative_distances, xy_points[:, 0])
            y_interp = np.interp(target_dist, cumulative_distances, xy_points[:, 1])
            control_points.append([x_interp, y_interp])
        
        return np.array(control_points)
    
    def _interpolate_yaw_smooth(self, yaw_points, num_control_points):
        """
        对yaw角进行平滑插值处理，考虑角度连续性
        
        Args:
            yaw_points: 原始的yaw角数组，形状为(N,)
            num_control_points: 目标控制点数量
        
        Returns:
            插值后的yaw角，形状为(num_control_points,)
        """
        n = len(yaw_points)
        
        # 处理角度连续性（解决-pi到pi的跳变）
        yaw_unwrapped = np.unwrap(yaw_points)
        
        # 使用线性插值
        indices_old = np.arange(n)
        indices_new = np.linspace(0, n - 1, num_control_points)
        yaw_interpolated = np.interp(indices_new, indices_old, yaw_unwrapped)
        
        # 将角度规范化到[-pi, pi]
        yaw_interpolated = (yaw_interpolated + np.pi) % (2 * np.pi) - np.pi
        
        return yaw_interpolated
    
    def _fit_bezier_xy(self, xy_points, num_control_points):
        """
        使用最小二乘法拟合贝塞尔曲线的xy坐标
        
        Args:
            xy_points: 原始路径的xy坐标，形状为(N, 2)
            num_control_points: 目标控制点数量
        
        Returns:
            贝塞尔控制点的xy坐标，形状为(num_control_points, 2)
        """
        n = len(xy_points)
        degree = num_control_points - 1
        
        # 生成参数t：根据路径长度分布
        t = self._compute_path_parameter(xy_points)
        
        # 构建贝塞尔基函数矩阵
        B = np.array([[self._bernstein_poly(degree, i, ti) for i in range(num_control_points)] for ti in t])
        
        # 使用最小二乘求解控制点
        # B @ P = xy_points
        # P = (B^T B)^-1 B^T xy_points
        try:
            control_points = np.linalg.lstsq(B, xy_points, rcond=None)[0]
        except np.linalg.LinAlgError:
            rospy.logwarn("Bezier fitting failed, using uniform sampling as fallback")
            # 备用方案：均匀采样原始点
            indices = np.linspace(0, n-1, num_control_points).astype(int)
            control_points = xy_points[indices]
        
        return control_points
    
    def _bernstein_poly(self, n, i, t):
        """
        计算贝塞尔基函数
        B_i^n(t) = C(n,i) * t^i * (1-t)^(n-i)
        
        Args:
            n: 曲线阶数
            i: 基函数索引
            t: 参数值
        
        Returns:
            基函数值
        """
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    def _compute_path_parameter(self, points):
        """
        根据路径长度计算参数t的分布
        
        Args:
            points: 路径点，形状为(N, 2)
        
        Returns:
            参数t的数组，形状为(N,)，范围从0到1
        """
        n = len(points)
        if n < 2:
            return np.array([0.0])
        
        # 计算累积距离
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        
        # 归一化到[0, 1]
        total_distance = cumulative_distances[-1]
        if total_distance < 1e-6:
            # 如果路径长度几乎为0，均匀分布
            return np.linspace(0, 1, n)
        
        t = cumulative_distances / total_distance
        return t
    
    def _interpolate_yaw(self, yaw_points, num_control_points):
        """
        对yaw角进行插值处理
        
        Args:
            yaw_points: 原始的yaw角数组，形状为(N,)
            num_control_points: 目标控制点数量
        
        Returns:
            插值后的yaw角，形状为(num_control_points,)
        """
        n = len(yaw_points)
        
        # 处理角度连续性（解决-pi到pi的跳变）
        yaw_unwrapped = np.unwrap(yaw_points)
        
        # 使用线性插值
        indices_old = np.linspace(0, n-1, n)
        indices_new = np.linspace(0, n-1, num_control_points)
        yaw_interpolated = np.interp(indices_new, indices_old, yaw_unwrapped)
        
        # 将角度规范化到[-pi, pi]
        yaw_interpolated = (yaw_interpolated + np.pi) % (2 * np.pi) - np.pi
        
        return yaw_interpolated
    
    def process_trajectory(self):
        """处理接收到的轨迹"""
        trajectory_path = self.sample_trajectory(self.current_trajectory)
        
        if trajectory_path is None:
            rospy.logwarn(f"Invalid trajectory for path_{self.current_path_id}, retrying...")
            self.retry_statistics['empty_trajectory'] += 1
            self.path_retry_count += 1
            self.scene_failed_attempts += 1
            self.schedule_next_path(reason="trajectory processing failed")
            return
        
        rospy.loginfo(f"Received trajectory with {len(trajectory_path)} points from C++")
        
        # 对轨迹进行样条曲线拟合，获得固定数量的控制点
        # 使用轨迹实际长度作为控制点数量，保留所有原始点
        # num_control_points = 20 + 2 # 控制点个数+起点/终点，一共22个点
        # spline_control_points = self.fit_spline_curve(trajectory_path, num_control_points)
        
        # if spline_control_points is None:
        #     rospy.logwarn(f"Spline fitting failed for path_{self.current_path_id}, retrying...")
        #     rospy.Timer(rospy.Duration(self.publish_delay), self.generate_next_path, oneshot=True)
        #     return
        
        # rospy.loginfo(f"Spline fitting completed: {len(trajectory_path)} points -> {len(spline_control_points)} control points")
        
        # 确定当前环境目录
        env_name = f"env{self.current_env_id:06d}"
        if self.current_phase == 'train':
            env_dir = os.path.join(self.train_dir, env_name)
        else:
            env_dir = os.path.join(self.val_dir, env_name)
        
        # 确保目录存在
        os.makedirs(env_dir, exist_ok=True)
        
        # 创建路径数据，直接使用完整轨迹（不进行样条拟合）
        path_data = {
            'path': trajectory_path,  # 使用完整的原始轨迹
            'map_name': env_name,
            'planner_map_version': self.current_planner_map_version,
            'planner_map_request_id': self.map_update_request_id,
        }
        
        # 保存路径文件
        path_file = os.path.join(env_dir, f"path_{self.current_path_id}.p")
        with open(path_file, 'wb') as f:
            pickle.dump(path_data, f)
        self.finish_attempt_record(
            'saved', path_file=os.path.abspath(path_file),
            trajectory_points=len(trajectory_path))
        
        rospy.loginfo(f"Saved {path_file} with {len(trajectory_path)} original trajectory points")
        
        # 更新计数
        self.current_path_id += 1
        self.paths_generated_for_current_env += 1
        self.path_retry_count = 0
        self.current_path_profile_retry_round = 0
        if self.paths_generated_for_current_env == 1:
            self.clear_canonical_needs_return()
        
        # 检查当前环境是否完成
        target_paths = self.train_paths_per_env if self.current_phase == 'train' else self.val_paths_per_env
        
        if self.paths_generated_for_current_env >= target_paths:
            rospy.loginfo(f"Completed {self.current_phase} phase for environment {self.current_env_id}")
            self.finish_current_phase()
            return
        
        # 生成当前环境的下一条路径
        self.schedule_next_path(reason="previous path saved")
    
    def regenerate_current_environment(self, event):
        """重新生成当前环境的地形"""
        rospy.logwarn(f"Regenerating terrain for env{self.current_env_id:06d}")
        
        # 清理待处理状态，防止旧轨迹干扰
        self.finish_active_planning_attempt()
        self.cancel_timer('next_path_timer')
        self.current_trajectory = None

        # train/val地图相互独立。失败重建只清理并重做当前split，避免已经
        # 完成的另一split被无关失败覆盖。
        env_name = f"env{self.current_env_id:06d}"
        phase_dir = (
            self.train_dir if self.current_phase == 'train' else self.val_dir)
        env_dir = os.path.join(phase_dir, env_name)
        for pattern in ('path_*.p', 'map.p', 'terrain_2d.png',
                        'terrain_3d.png'):
            for old_file in glob.glob(os.path.join(env_dir, pattern)):
                try:
                    os.remove(old_file)
                except OSError as exc:
                    rospy.logwarn(
                        "Failed to remove stale generated file %s: %s",
                        old_file, exc)
        
        # 重置当前路径ID为0，重新开始生成该环境的路径
        self.current_path_id = 0
        self.paths_generated_for_current_env = 0  # 重置路径计数器
        
        # 重新生成地形
        env_dir = self.generate_new_environment()
        
        self.schedule_next_path(
            delay=self.map_ready_delay, reason="environment regenerated")
    
    def start_new_environment(self, event):
        """开始新环境"""
        self.generate_new_environment()
        if self._env_skip_status:
            self.finish_current_phase()
            return
        self.schedule_next_path(
            delay=self.map_ready_delay, reason="new environment ready")
    
    def generate_next_path(self, event):
        """生成下一条路径"""
        # 检查系统状态，避免在不合适的时候发送位姿
        if self.waiting_for_result:
            rospy.logdebug(
                "Still waiting for planning attempt %s; ignoring duplicate "
                "path-generation trigger", self.active_planning_attempt_id)
            return

        if not self.planner_is_connected():
            rospy.logfatal(
                "Planner connections were lost before path_%d; stopping this "
                "worker instead of waiting forever", self.current_path_id)
            rospy.signal_shutdown("only_planner disconnected")
            return
        
        # 生成随机位姿对
        (start_x, start_y, start_yaw), (target_x, target_y, target_yaw) = self.generate_pose_pair()
        
        # 创建位姿消息
        start_pose = self.create_pose_stamped(start_x, start_y, start_yaw)
        target_pose = self.create_pose_stamped(target_x, target_y, target_yaw)
        
        # 设置等待状态
        self.waiting_for_result = True
        self.planning_attempt_id += 1
        attempt_id = self.planning_attempt_id
        self.active_planning_attempt_id = attempt_id
        self.active_attempt_record = {
            'attempt_id': attempt_id,
            'started_utc': datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            'environment_id': self.current_env_id,
            'phase': self.current_phase,
            'path_id': self.current_path_id,
            'retry_index': self.path_retry_count,
            'start_pose': [start_x, start_y, start_yaw],
            'target_pose': [target_x, target_y, target_yaw],
            'sampling_profile': self._json_safe(
                self.current_pose_sampling_profile),
            'planner_map_version': self.current_planner_map_version,
            'planner_map_request_id': self.map_update_request_id,
        }
        self.experiment_status = 'running'
        self.write_experiment_manifest()
        
        # 更新期望的路径ID
        self.expected_path_id = self.current_path_id
        
        env_name = f"env{self.current_env_id:06d}"
        rospy.loginfo(
            "Generating path_%d for %s (%s), planning attempt=%d, retry=%d",
            self.current_path_id, env_name, self.current_phase,
            attempt_id, self.path_retry_count)
        rospy.loginfo(f"  Pose bounds: X[{self.map_x_min:.1f}, {self.map_x_max:.1f}], Y[{self.map_y_min:.1f}, {self.map_y_max:.1f}]")
        rospy.loginfo(f"  Start: [{start_x:.3f}, {start_y:.3f}, {start_yaw:.3f}]")
        rospy.loginfo(f"  Target: [{target_x:.3f}, {target_y:.3f}, {target_yaw:.3f}]")
        
        # 发布位姿
        start_pose.header.seq = attempt_id
        target_pose.header.seq = attempt_id
        self.start_pose_pub.publish(start_pose)
        rospy.sleep(0.1)
        self.target_pose_pub.publish(target_pose)
        
        # 看门狗只监控本次尝试。规划器不能被ROS定时器真正取消，因此超时后
        # 不发送并发请求；否则旧请求完成时会被误认为新请求的结果。
        self.cancel_timer('planning_timeout_timer')
        self.planning_timeout_timer = rospy.Timer(
            rospy.Duration(self.map_update_timeout),
            lambda timer_event: self.check_planning_timeout(
                timer_event, attempt_id),
            oneshot=True)

    def check_planning_timeout(self, event, attempt_id=None):
        """规划看门狗：报告慢规划，但不与尚未结束的规划并发重试。"""
        if (not self.waiting_for_result or
                attempt_id != self.active_planning_attempt_id):
            return
        if not self.planner_is_connected():
            self.finish_active_planning_attempt()
            rospy.logfatal(
                "Planner disconnected during attempt %d for path_%d; "
                "stopping this worker instead of waiting forever",
                attempt_id, self.current_path_id)
            rospy.signal_shutdown("only_planner disconnected")
            return
        self.retry_statistics['watchdog_timeout'] += 1
        rospy.logwarn(
            "Planning attempt %d for path_%d has run for %.1fs; still "
            "waiting instead of launching an overlapping retry",
            attempt_id, self.current_path_id, self.map_update_timeout)
        self.planning_timeout_timer = rospy.Timer(
            rospy.Duration(self.map_update_timeout),
            lambda timer_event: self.check_planning_timeout(
                timer_event, attempt_id),
            oneshot=True)

    def planner_is_connected(self):
        """规划请求与结果两个方向都建立连接后才认为规划器可用。"""
        return (
            self.start_pose_pub.get_num_connections() > 0 and
            self.target_pose_pub.get_num_connections() > 0 and
            self.result_sub.get_num_connections() > 0 and
            self.traj_sub.get_num_connections() > 0)

    def wait_for_planner(self):
        """等待规划器完成地图/KD-tree初始化，超时则明确退出。"""
        deadline = time.monotonic() + self.planner_connection_timeout
        rospy.loginfo(
            "Waiting up to %.1fs for only_planner connections...",
            self.planner_connection_timeout)
        try:
            rospy.wait_for_service(
                'update_terrain_map',
                timeout=self.planner_connection_timeout)
        except rospy.ROSException:
            rospy.logfatal(
                "only_planner did not advertise update_terrain_map within "
                "%.1fs", self.planner_connection_timeout)
            rospy.signal_shutdown("planner map update service timeout")
            return False
        while not rospy.is_shutdown():
            if self.planner_is_connected():
                rospy.loginfo("only_planner is connected; starting generation")
                return True
            if time.monotonic() >= deadline:
                rospy.logfatal(
                    "only_planner did not establish all request/result "
                    "connections within %.1fs",
                    self.planner_connection_timeout)
                rospy.signal_shutdown("only_planner connection timeout")
                return False
            rospy.sleep(0.2)
        return False

    def start_generation(self):
        """开始数据集生成"""
        rospy.loginfo(f"Starting terrain dataset generation:")
        rospy.loginfo(f"  Environments: {self.num_environments} (starting from {self.start_env_id})")
        rospy.loginfo(f"  Paths per environment: {self.train_paths_per_env} (train) + {self.val_paths_per_env} (val)")
        rospy.loginfo(f"  Total paths: {self.num_environments * (self.train_paths_per_env + self.val_paths_per_env)}")
        
        # 规划器只有在地图和KD-tree初始化完毕后才创建订阅/发布连接。
        if not self.wait_for_planner():
            return
        
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

        # 对于外部地图，使用目标参数；对于生成地图，使用默认参数
        if self.use_external_map:
            # 使用外部地图的目标参数
            map_size_x = self.target_map_size
            map_size_y = map_size_x
            xy_res = self.target_resolution
        else:
            # 使用默认的地图参数
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

                # 规范合同只有 (H=row=y, W=column=x, Y=yaw)。这里禁止
                # transpose/resize 猜测，否则方形生产地图会静默掩盖H/W错误。
                H = occ.info.height
                W = occ.info.width
                Y = int(yaw_bins)
                expected_shape = (H, W, Y)
                if stability_map.shape != expected_shape:
                    raise ValueError(
                        "stability map must use (H=row=y,W=column=x,Y=yaw): "
                        f"{stability_map.shape} != {expected_shape}")
                stability_map = np.asarray(
                    stability_map, dtype=np.float32)
                valid_mask = np.asarray(
                    grid_map_data.get(
                        'valid_mask', np.ones((H, W), dtype=bool)),
                    dtype=bool)
                if valid_mask.shape != (H, W):
                    raise ValueError(
                        f"valid_mask shape {valid_mask.shape} != {(H, W)}")
                stability_map[~valid_mask, :] = 0.0
                
                # 将x和y进行互换
                # stability_map = np.transpose(stability_map, (1, 0, 2))
                # # 将x翻转
                # stability_map = np.flip(stability_map, axis=0)
                # # 将y翻转
                # stability_map = np.flip(stability_map, axis=1)

                # tipping 概率 = 1 - stability（保留 yaw 维度用于内部缓存）
                tipping_map = 1.0 - stability_map
                tipping_map = np.clip(tipping_map, 0.0, 1.0)

                # 对外发布的 OccupancyGrid 必须是二维 (H, W) 的展开数据。
                # 使用航向维度上的最大 tipping（保守策略）作为每个格子的倾覆概率。
                per_cell_tipping = tipping_map.max(axis=2)
                per_cell_tipping = np.clip(per_cell_tipping, 0.0, 1.0)

                # 映射到 0-100 的整数占用值（100 表示占用）
                occ_vals_2d = (per_cell_tipping * 100.0).round().astype(np.int8)

                from std_msgs.msg import (
                    Float32MultiArray,
                    MultiArrayDimension,
                )
                fam = Float32MultiArray()
                fam.data = tipping_map.flatten().astype(
                    np.float32).tolist()
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

                # 发布二维 OccupancyGrid 以兼容仅接受 2D 的节点
                occ.data = occ_vals_2d.flatten().tolist()

                return occ, fam
            except Exception as e:
                raise RuntimeError(
                    "Failed to compute strict HWY occupancy: "
                    f"{e}") from e

        raise RuntimeError(
            "normal_x, normal_y and normal_z are required for planner "
            "occupancy")

    def load_external_map(self):
        """
        加载外部地图文件
        
        Returns:
            o3d.geometry.PointCloud: 加载的点云数据
        """
        try:
            if self.external_map_format.lower() == 'pcd':
                # 加载PCD格式文件
                pcd = o3d.io.read_point_cloud(self.external_map_path)
                rospy.loginfo(f"Loaded PCD file with {len(pcd.points)} points")
                
            elif self.external_map_format.lower() == 'ply':
                # 加载PLY格式文件
                pcd = o3d.io.read_point_cloud(self.external_map_path)
                rospy.loginfo(f"Loaded PLY file with {len(pcd.points)} points")
                
            elif self.external_map_format.lower() == 'txt':
                # 加载TXT格式文件 (假设格式为: x y z)
                points = np.loadtxt(self.external_map_path)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                rospy.loginfo(f"Loaded TXT file with {len(pcd.points)} points")
                
            elif self.external_map_format.lower() == 'heightmap':
                # 加载高度图格式 (假设为numpy数组或图像)
                if self.external_map_path.endswith('.npy'):
                    heightmap = np.load(self.external_map_path)
                else:
                    # 假设是图像格式
                    import cv2
                    heightmap = cv2.imread(self.external_map_path, cv2.IMREAD_GRAYSCALE)
                    heightmap = heightmap.astype(np.float32) / 255.0 * self.max_height
                
                # 将高度图转换为点云
                h, w = heightmap.shape
                x = np.linspace(-self.map_size/2, self.map_size/2, w)
                y = np.linspace(-self.map_size/2, self.map_size/2, h)
                xx, yy = np.meshgrid(x, y)
                
                points = np.stack([xx.flatten(), yy.flatten(), heightmap.flatten()], axis=1)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                rospy.loginfo(f"Converted heightmap to point cloud with {len(pcd.points)} points")

            elif self.external_map_format.lower() in ('las', 'laz'):
                # Keep every finite XYZ return.  Preserve the aligned LAS
                # labels separately for crop-level forest selection; they do
                # not filter or alter the raw planner point cloud.
                (points, source_count, source_bounds,
                 source_class_counts, source_classification) = (
                     load_surface_points(
                         self.external_map_path,
                         return_classification=True))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                self.external_source_classifications[
                    self.external_map_path] = source_classification
                self.external_source_metadata[self.external_map_path] = {
                    'source_point_count': int(source_count),
                    'source_bounds': source_bounds,
                    'source_class_counts': source_class_counts,
                    'source_classification_field': 'LAS classification',
                    'source_surface': 'all_finite_XYZ_no_LAS_class_filter',
                }
                rospy.loginfo(
                    "Loaded raw %s source with %d finite XYZ points "
                    "and aligned LAS classification labels",
                    self.external_map_format.upper(), len(points))

            else:
                raise ValueError(f"Unsupported external map format: {self.external_map_format}")
            
            # 验证点云是否为空
            if len(pcd.points) == 0:
                raise ValueError("Loaded point cloud is empty")
                
            return pcd
            
        except Exception as e:
            rospy.logerr(f"Failed to load external map: {e}")
            raise
    
    def validate_external_map_processing(self, grid_map_data, target_map_size, target_resolution):
        """
        验证外部地图处理结果
        
        Args:
            grid_map_data: 处理后的栅格地图数据
            target_map_size: 目标地图尺寸（米）
            target_resolution: 目标分辨率（米）
        """
        expected_grid_size = int(round(target_map_size / target_resolution))
        actual_shape = grid_map_data['elevation'].shape
        
        rospy.loginfo("=== External Map Processing Validation ===")
        rospy.loginfo(f"Target map size: {target_map_size}m x {target_map_size}m")
        rospy.loginfo(f"Target resolution: {target_resolution}m")
        rospy.loginfo(f"Expected grid size: {expected_grid_size} x {expected_grid_size}")
        rospy.loginfo(f"Actual grid shape: {actual_shape}")
        rospy.loginfo(f"Bounds: {grid_map_data.get('bounds', 'Unknown')}")
        rospy.loginfo(f"Resolution: {grid_map_data.get('resolution', 'Unknown')}")
        if self.external_map_is_canonical:
            processing_method = "Audited canonical fitted surface"
        elif self.external_map_format.lower() in ('las', 'laz'):
            processing_method = "Online fitted and quality-gated mother-map crop"
        else:
            processing_method = "Random translated/rotated point-cloud crop"
        rospy.loginfo("Processing method: %s", processing_method)
        
        # 验证栅格尺寸
        if actual_shape == (expected_grid_size, expected_grid_size):
            rospy.loginfo("✓ Grid size matches rotated crop target")
        else:
            rospy.logwarn(f"✗ Grid size mismatch: expected ({expected_grid_size}, {expected_grid_size}), got {actual_shape}")
        
        # 验证边界
        expected_bounds = (-target_map_size/2, target_map_size/2, -target_map_size/2, target_map_size/2)
        actual_bounds = grid_map_data.get('bounds')
        if actual_bounds and np.allclose(actual_bounds, expected_bounds, rtol=1e-3):
            rospy.loginfo("✓ Bounds match target")
        else:
            rospy.logwarn(f"✗ Bounds mismatch: expected {expected_bounds}, got {actual_bounds}")
        
        # 验证分辨率
        actual_resolution = grid_map_data.get('resolution')
        if actual_resolution and abs(actual_resolution - target_resolution) < 1e-6:
            rospy.loginfo("✓ Resolution matches target")
        else:
            rospy.logwarn(f"✗ Resolution mismatch: expected {target_resolution}, got {actual_resolution}")

        # 训练和规划都不应接收到空高程或空法向量。尤其在聚合分辨率
        # 改为真实0.2m后，这项检查可以及时暴露点云覆盖不足。
        for channel_name in (
                'elevation', 'normal_x', 'normal_y', 'normal_z'):
            channel = np.asarray(grid_map_data.get(channel_name))
            if channel.shape != actual_shape:
                raise RuntimeError(
                    f"External map channel {channel_name} has shape "
                    f"{channel.shape}, expected {actual_shape}")
            invalid_count = int(channel.size - np.isfinite(channel).sum())
            if invalid_count:
                raise RuntimeError(
                    f"External map channel {channel_name} contains "
                    f"{invalid_count}/{channel.size} invalid cells")
        rospy.loginfo("✓ All elevation and normal cells are finite")
        
        rospy.loginfo("=" * 45)
        
if __name__ == '__main__':
    try:
        generator = TerrainDatasetGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        pass
