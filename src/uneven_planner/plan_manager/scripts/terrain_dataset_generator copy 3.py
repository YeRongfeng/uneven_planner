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
from plan_manager.srv import PointCloudToGrid, PointCloudToGridRequest

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
import shutil
import glob
import cv2
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

    # 将x和y进行转置
    yaw_stability = yaw_stability.permute(1, 0, 2)  # (W, H, yaw_bins) -> (H, W, yaw_bins)
    
    # # 对x翻转，y翻转以匹配地图坐标系
    # yaw_stability = torch.flip(yaw_stability, dims=[0])  # 沿H轴翻转
    # yaw_stability = torch.flip(yaw_stability, dims=[1])  # 沿W轴翻转

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

    def __init__(self, coarse_resolution=0.4, fine_resolution=0.2, voxel_size=0.2):
        """
        初始化GridTransformer
        
        Args:
            coarse_resolution: 粗糙网格分辨率（0.4m）- C++服务返回此分辨率
            fine_resolution: 精细网格分辨率（0.2m）- Python端插值到此分辨率
            voxel_size: 体素降采样大小（0.2m）- C++端使用
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
            rospy.wait_for_service('/pointcloud_to_grid', timeout=10.0)
            self.grid_service = rospy.ServiceProxy('/pointcloud_to_grid', PointCloudToGrid)
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
            
            # 将一维数组转换为二维栅格
            coarse_elevation = np.array(response.elevation_grid).reshape(response.grid_height, response.grid_width)
            coarse_normal_x = np.array(response.normal_x_grid).reshape(response.grid_height, response.grid_width)
            coarse_normal_y = np.array(response.normal_y_grid).reshape(response.grid_height, response.grid_width)
            coarse_normal_z = np.array(response.normal_z_grid).reshape(response.grid_height, response.grid_width)
            
            coarse_grids = (coarse_elevation, coarse_normal_x, coarse_normal_y, coarse_normal_z)
            
        except Exception as e:
            rospy.logerr(f"C++ service call failed: {e}")
            raise

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

    
    # def process_grid_cells(self, points, normals, grid_dict, map_size):
    #     """处理栅格单元，计算高程和法向量"""
    #     width, height = map_size

    #     if CUPY_AVAILABLE:
    #         elevation_grid = cp.full((height, width), cp.nan, dtype=cp.float32)
    #         normal_x_grid = cp.full((height, width), cp.nan, dtype=cp.float32)
    #         normal_y_grid = cp.full((height, width), cp.nan, dtype=cp.float32)
    #         normal_z_grid = cp.full((height, width), cp.nan, dtype=cp.float32)
    #     else:
    #         elevation_grid = np.full((height, width), np.nan, dtype=np.float32)
    #         normal_x_grid = np.full((height, width), np.nan, dtype=np.float32)
    #         normal_y_grid = np.full((height, width), np.nan, dtype=np.float32)
    #         normal_z_grid = np.full((height, width), np.nan, dtype=np.float32)

    #     rospy.loginfo(f"Processing {len(grid_dict)} grid cells...")

    #     for (grid_x, grid_y), point_indices in grid_dict.items():
    #         if len(point_indices) < 1:
    #             continue

    #         cell_points = points[point_indices]
    #         cell_normals = normals[point_indices]

    #         # 找到法向量Z分量最小的点
    #         norm_lengths = np.linalg.norm(cell_normals, axis=1)
    #         normalized_z = np.abs(cell_normals[:, 2] / norm_lengths)

    #         min_z_local_idx = np.argmin(normalized_z)
    #         min_z_global_idx = point_indices[min_z_local_idx]

    #         selected_point = points[min_z_global_idx]
    #         selected_normal = normals[min_z_global_idx]
    #         norm_length = np.linalg.norm(selected_normal)
    #         normalized_normal = selected_normal / norm_length

    #         elevation_grid[grid_y, grid_x] = selected_point[2]
    #         normal_x_grid[grid_y, grid_x] = normalized_normal[0]
    #         normal_y_grid[grid_y, grid_x] = normalized_normal[1]
    #         normal_z_grid[grid_y, grid_x] = normalized_normal[2]

    #     # 转换回CPU
    #     if CUPY_AVAILABLE:
    #         elevation_grid = cp.asnumpy(elevation_grid)
    #         normal_x_grid = cp.asnumpy(normal_x_grid)
    #         normal_y_grid = cp.asnumpy(normal_y_grid)
    #         normal_z_grid = cp.asnumpy(normal_z_grid)

    #     rospy.loginfo("Grid cell processing completed")
    #     return elevation_grid, normal_x_grid, normal_y_grid, normal_z_grid

    def process_grid_cells(self, points, normals, grid_dict, map_size):
        """修正为匹配C++逻辑：最高点高程 + 最水平法向量"""
        width, height = map_size
        
        elevation_grid = np.full((height, width), np.nan, dtype=np.float32)
        normal_x_grid = np.full((height, width), np.nan, dtype=np.float32)
        normal_y_grid = np.full((height, width), np.nan, dtype=np.float32)
        normal_z_grid = np.full((height, width), np.nan, dtype=np.float32)

        for (grid_x, grid_y), point_indices in grid_dict.items():
            if len(point_indices) < 1:
                continue

            cell_points = points[point_indices]
            cell_normals = normals[point_indices]

            # 匹配C++逻辑1: 找到最高点的高程
            elevations = cell_points[:, 2]
            highest_idx = np.argmax(elevations)
            highest_elevation = elevations[highest_idx]
            
            # 匹配C++逻辑2: 找到最水平的法向量（Z分量绝对值最小）
            norm_lengths = np.linalg.norm(cell_normals, axis=1)
            # 修正：使用原始Z值，不是绝对值，匹配C++的min_z计算
            normalized_z = np.abs(cell_normals[:, 2] / norm_lengths)
            min_z_idx = np.argmin(normalized_z)
            
            selected_normal = cell_normals[min_z_idx]
            norm_length = np.linalg.norm(selected_normal)
            
            # 防止除零
            if norm_length < 1e-8:
                normalized_normal = np.array([0, 0, 1])
            else:
                normalized_normal = selected_normal / norm_length
            
            # 匹配C++的赋值逻辑
            elevation_grid[grid_y, grid_x] = highest_elevation  # 最高点高程
            normal_x_grid[grid_y, grid_x] = normalized_normal[0]  # 最水平点的法向量
            normal_y_grid[grid_y, grid_x] = normalized_normal[1]
            normal_z_grid[grid_y, grid_x] = normalized_normal[2]

        return elevation_grid, normal_x_grid, normal_y_grid, normal_z_grid

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
        
        # 外部地图参数
        self.use_external_map = rospy.get_param('~use_external_map', False)
        self.external_map_path = rospy.get_param('~external_map_path', '')
        self.external_map_format = rospy.get_param('~external_map_format', 'pcd')  # 支持 'pcd', 'ply', 'txt', 'heightmap'
        
        # 地形生成参数（仅在不使用外部地图时有效）
        self.map_size = rospy.get_param('~map_size', 10.0)
        self.map_resolution = rospy.get_param('~map_resolution', 0.02)
        self.max_height = rospy.get_param('~max_height', 3.0)
        self.min_height = rospy.get_param('~min_height', 0.0)
        
        # 地图转换参数（完全匹配grid_transformer.cpp）
        self.coarse_resolution = rospy.get_param('~coarse_resolution', 0.4)  # 匹配grid_coarse_resolution_
        self.fine_resolution = rospy.get_param('~fine_resolution', 0.2)      # 匹配grid_desired_resolution_
        self.voxel_size = rospy.get_param('~voxel_size', 0.2)                # 匹配voxelSize
        
        # 路径生成参数（动态计算，在地图生成后更新）
        self.effective_map_size = self.map_size  # 有效地图尺寸，外部地图时会更新
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
            rospy.loginfo(f"Using external map from: {self.external_map_path}")
            
            # 验证外部地图文件存在
            if not os.path.exists(self.external_map_path):
                rospy.logerr(f"External map file not found: {self.external_map_path}")
                raise FileNotFoundError(f"External map file not found: {self.external_map_path}")
        
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
    
    def copy_map_to_val_directory(self):
        """将地图文件从训练目录复制到验证目录"""
        env_name = f"env{self.current_env_id:06d}"
        train_env_dir = os.path.join(self.train_dir, env_name)
        val_env_dir = os.path.join(self.val_dir, env_name)
        
        # 确保验证目录存在
        os.makedirs(val_env_dir, exist_ok=True)
        
        # 需要复制的文件列表
        files_to_copy = ['map.p', 'terrain_2d.png', 'terrain_3d.png']
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
        
        # 生成或加载地形
        if self.use_external_map:
            # 使用外部地图
            rospy.loginfo("Loading external map...")
            pcd = self.load_external_map()
            # 对pcd进行180度旋转
            R = np.array([[-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, 1]])
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).dot(R.T))

            # 为外部地图创建基本的terrain_info
            points = np.asarray(pcd.points)
            terrain_info = {
                'map_size': self.map_size,
                'resolution': self.map_resolution,
                'max_height': np.max(points[:, 2]) if len(points) > 0 else self.max_height,
                'min_height': np.min(points[:, 2]) if len(points) > 0 else self.min_height,
                'num_points': len(points),
                'source': 'external_map',
                'external_map_path': self.external_map_path,
                'external_map_format': self.external_map_format
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
            # 对于外部地图，强制使用目标地图尺寸和分辨率
            target_map_size = rospy.get_param('~target_map_size', self.map_size)  # 目标地图尺寸，默认40m
            target_resolution = rospy.get_param('~target_resolution', 0.4)  # 目标分辨率，默认0.4m
            
            # 计算目标栅格尺寸
            target_grid_size = int(target_map_size / target_resolution)
            rospy.loginfo(f"External map: target size={target_map_size}m, resolution={target_resolution}m, grid={target_grid_size}x{target_grid_size}")
            
            # 强制使用目标边界（-20m到+20m）
            map_bounds = (-target_map_size/2, target_map_size/2, -target_map_size/2, target_map_size/2)
            
            # 临时修改grid_transformer的分辨率以匹配目标要求
            original_fine_resolution = self.grid_transformer.fine_resolution
            self.grid_transformer.fine_resolution = target_resolution
            
            grid_map_data = self.grid_transformer.transform_pointcloud_to_grid(pcd, map_bounds)
            
            # 恢复原分辨率
            self.grid_transformer.fine_resolution = original_fine_resolution
            
            # 验证生成的栅格尺寸
            actual_grid_shape = grid_map_data['elevation'].shape
            rospy.loginfo(f"Generated grid shape: {actual_grid_shape}, expected: ({target_grid_size}, {target_grid_size})")
            
            if actual_grid_shape != (target_grid_size, target_grid_size):
                rospy.loginfo(f"Grid size mismatch! Cropping from {actual_grid_shape} to ({target_grid_size}, {target_grid_size})")
                
                # 计算裁剪区域（中心裁剪）
                current_h, current_w = actual_grid_shape
                
                # 计算裁剪的起始和结束索引
                if current_h > target_grid_size:
                    start_h = (current_h - target_grid_size) // 2
                    end_h = start_h + target_grid_size
                else:
                    start_h = 0
                    end_h = current_h
                    
                if current_w > target_grid_size:
                    start_w = (current_w - target_grid_size) // 2
                    end_w = start_w + target_grid_size
                else:
                    start_w = 0
                    end_w = current_w
                
                rospy.loginfo(f"Cropping region: H[{start_h}:{end_h}], W[{start_w}:{end_w}]")
                
                # 执行裁剪
                grid_map_data['elevation'] = grid_map_data['elevation'][start_h:end_h, start_w:end_w]
                grid_map_data['normal_x'] = grid_map_data['normal_x'][start_h:end_h, start_w:end_w]
                grid_map_data['normal_y'] = grid_map_data['normal_y'][start_h:end_h, start_w:end_w]
                grid_map_data['normal_z'] = grid_map_data['normal_z'][start_h:end_h, start_w:end_w]
                
                # 如果裁剪后仍然小于目标尺寸，进行零填充
                cropped_h, cropped_w = grid_map_data['elevation'].shape
                if cropped_h < target_grid_size or cropped_w < target_grid_size:
                    rospy.loginfo(f"Padding cropped grid from {(cropped_h, cropped_w)} to ({target_grid_size}, {target_grid_size})")
                    
                    # 计算需要填充的尺寸
                    pad_h = max(0, target_grid_size - cropped_h)
                    pad_w = max(0, target_grid_size - cropped_w)
                    
                    pad_h_before = pad_h // 2
                    pad_h_after = pad_h - pad_h_before
                    pad_w_before = pad_w // 2
                    pad_w_after = pad_w - pad_w_before
                    
                    # 对每个通道进行填充
                    grid_map_data['elevation'] = np.pad(grid_map_data['elevation'], 
                                                       ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)), 
                                                       mode='edge')
                    grid_map_data['normal_x'] = np.pad(grid_map_data['normal_x'], 
                                                      ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)), 
                                                      mode='edge')
                    grid_map_data['normal_y'] = np.pad(grid_map_data['normal_y'], 
                                                      ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)), 
                                                      mode='edge')
                    grid_map_data['normal_z'] = np.pad(grid_map_data['normal_z'], 
                                                      ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)), 
                                                      mode='edge')
                
                rospy.loginfo(f"Final grid shape after cropping: {grid_map_data['elevation'].shape}")
            
            # 验证处理结果
            self.validate_external_map_processing(grid_map_data, target_map_size, target_resolution)
            
            # 更新位姿生成范围以匹配外部地图尺寸
            self.update_pose_generation_bounds(target_map_size)
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
            target_map_size = rospy.get_param('~target_map_size', self.map_size)
            target_resolution = rospy.get_param('~target_resolution', 0.4)
            target_bounds = (-target_map_size/2, target_map_size/2, -target_map_size/2, target_map_size/2)
            
            map_data = {
                'tensor': grid_map_tensor,
                'bounds': target_bounds,  # 强制使用目标边界
                'resolution': target_resolution,  # 强制使用目标分辨率
                'map_name': env_name,
                'channels': ['elevation', 'normal_x', 'normal_y', 'normal_z'],
                'shape': grid_map_tensor.shape,
                'source': 'external_map',
                'original_map_path': self.external_map_path,
                'target_grid_size': f"{grid_map_tensor.shape[0]}x{grid_map_tensor.shape[1]}"
            }
            rospy.loginfo(f"External map: saved with target bounds {target_bounds}, resolution {target_resolution}, grid shape {grid_map_tensor.shape}")
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
        rospy.loginfo(f"Map bounds: {grid_map_data.get('bounds', 'Unknown')}")
        rospy.loginfo(f"Map resolution: {grid_map_data.get('resolution', 'Unknown')}")
        rospy.loginfo(f"Map shape: {grid_map_tensor.shape}")
        
        # 更新地图生成时间戳，用于轨迹验证
        self.map_generation_timestamp = rospy.Time.now().to_sec()
        
        # 清理任何待处理的轨迹回调状态，防止旧地图的轨迹被保存到新地图
        self.waiting_for_result = False
        self.current_trajectory = None
        
        # 更新期望的环境和路径ID
        self.expected_env_id = self.current_env_id
        self.expected_path_id = 0
        
        # 等待地图更新 - 为外部地图增加更多等待时间
        wait_time = 8.0 if self.use_external_map else 3.0
        rospy.loginfo(f"Waiting {wait_time}s for map system to process new terrain...")
        rospy.sleep(wait_time)
        
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
                # 对于外部地图，创建说明图
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.text(0.5, 0.5, f'External Map Used\n\nSource: {terrain_info.get("external_map_path", "Unknown")}\n'
                                  f'Format: {terrain_info.get("external_map_format", "Unknown")}\n'
                                  f'Points: {terrain_info.get("num_points", "Unknown")}\n'
                                  f'Height Range: {terrain_info.get("min_height", "N/A"):.2f} - {terrain_info.get("max_height", "N/A"):.2f} m',
                        transform=ax.transAxes, fontsize=14, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title('External Map Information')
                ax.axis('off')
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
                # 对于外部地图，尝试从点云创建3D可视化
                # 注意：这可能对大型点云很慢，所以我们进行采样
                try:
                    # 这里需要访问刚加载的外部点云数据
                    # 我们需要重新加载或者传递点云数据
                    pcd = self.load_external_map()
                    points = np.asarray(pcd.points)
                    
                    # 采样以减少绘制时间
                    if len(points) > 10000:
                        indices = np.random.choice(len(points), 10000, replace=False)
                        points = points[indices]
                    
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                       c=points[:, 2], cmap='terrain', s=1, alpha=0.6)
                    
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_zlabel('Height (m)')
                    ax.set_title('3D External Map View')
                    plt.colorbar(scatter, ax=ax, label='Height (m)')
                    
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
        
        # 默认安全位姿对，使用当前地图边界内的相对坐标
        rospy.logwarn("Failed to generate valid pose pair, using default safe poses")
        safe_offset = min(abs(self.map_x_min), abs(self.map_x_max), abs(self.map_y_min), abs(self.map_y_max)) * 0.5
        return (self.map_x_min + safe_offset, self.map_y_min + safe_offset, 0.0), \
               (self.map_x_max - safe_offset, self.map_y_max - safe_offset, 0.0)
    
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
        if traj_points is None or len(traj_points) == 0:
            rospy.logwarn(f"Invalid or empty trajectory for path_{self.current_path_id}, retrying...")
            self.waiting_for_result = False
            rospy.Timer(rospy.Duration(self.publish_delay), self.generate_next_path, oneshot=True)
            return

        # 如果预计算了 yaw_scores，使用其进行精确验证
        invalid_found = False
        unstable_count = 0
        total_points = len(traj_points)
        
        # 强制要求使用预计算的 yaw_scores 进行验证；如果不可用则直接拒绝轨迹，避免回退导致保存不安全轨迹
        if self.current_yaw_scores is None or self.current_map_bounds is None or self.current_resolution is None:
            rospy.logwarn("Rejecting trajectory because precomputed yaw stability map or map bounds/resolution is unavailable")
            invalid_found = True
        else:
            min_x, max_x, min_y, max_y = self.current_map_bounds
            H, W, Y = self.current_yaw_scores.shape[0], self.current_yaw_scores.shape[1], self.current_yaw_scores.shape[2]
            
            rospy.loginfo(f"Validating trajectory with {total_points} points against stability map shape {(H, W, Y)}")
            rospy.loginfo(f"Map bounds: [{min_x:.2f}, {max_x:.2f}] x [{min_y:.2f}, {max_y:.2f}], resolution: {self.current_resolution:.3f}")
            
            for point_idx, (x, y, yaw) in enumerate(traj_points):
                # 转换到栅格索引
                ix = int(math.floor((x - min_x) / self.current_resolution))
                iy = int(math.floor((y - min_y) / self.current_resolution))
                
                # 检查边界
                if ix < 0 or ix >= W or iy < 0 or iy >= H:
                    rospy.logwarn(f"Point {point_idx}: [{x:.3f},{y:.3f}] -> grid[{ix},{iy}] out of bounds [0,{W})x[0,{H})")
                    invalid_found = True
                    break

                # yaw bin 计算
                yaw_norm = yaw
                # normalize to [-pi, pi)
                while yaw_norm < -math.pi:
                    yaw_norm += 2*math.pi
                while yaw_norm >= math.pi:
                    yaw_norm -= 2*math.pi
                
                bin_width = 2*math.pi / float(Y)
                yidx = int(math.floor((yaw_norm + math.pi) / bin_width))
                
                # 确保yaw索引在有效范围内
                yidx = max(0, min(yidx, Y-1))

                # yaw_scores: 1 -> stable, 0 -> tipping
                # 注意：numpy 数组的第一个维度是行（y / height），第二个维度是列（x / width）
                try:
                    val = self.current_yaw_scores[iy, ix, yidx]
                    # val = self.current_yaw_scores[ix, iy, yidx]
                    
                    # val = self.current_yaw_scores[W - 1 - ix, H - 1 - iy, yidx]
                    # val = self.current_yaw_scores[H - 1 - iy, W - 1 - ix, yidx]
                except IndexError as e:
                    rospy.logwarn(f"Point {point_idx}: index error at grid[{iy},{ix},{yidx}] for traj point [{x:.3f},{y:.3f},{yaw:.3f}]: {e}")
                    invalid_found = True
                    break

                # 处理可能的 NaN / 非数值情况并进行阈值判定
                try:
                    # 将 val 转为浮点数后比较阈值
                    val_float = float(val)
                    
                    # 检查是否为NaN或无效值
                    if math.isnan(val_float) or math.isinf(val_float):
                        rospy.logwarn(f"Point {point_idx}: NaN/Inf stability value at [{x:.3f},{y:.3f},{yaw:.3f}] -> grid[{iy},{ix},{yidx}]")
                        invalid_found = True
                        break
                    
                    # 稳定性判断：val > 0.5 表示稳定，<= 0.5 表示不稳定
                    stable = val_float > 0.5
                    
                    if not stable:
                        unstable_count += 1
                        rospy.logwarn(f"Point {point_idx}: UNSTABLE at [{x:.3f},{y:.3f},{yaw:.3f}] -> grid[{iy},{ix},{yidx}], stability={val_float:.3f}")
                        
                        # 严格模式：任何一个点不稳定就拒绝整条轨迹
                        invalid_found = True
                        break
                    else:
                        # 可选：记录稳定点的调试信息
                        if point_idx < 5 or point_idx % 20 == 0:  # 只记录前几个点和每20个点
                            rospy.logdebug(f"Point {point_idx}: stable at [{x:.3f},{y:.3f},{yaw:.3f}] -> stability={val_float:.3f}")
                            
                except (ValueError, TypeError) as e:
                    rospy.logwarn(f"Point {point_idx}: error processing stability value {val} at [{x:.3f},{y:.3f},{yaw:.3f}]: {e}")
                    invalid_found = True
                    break
                    
        # 输出验证结果统计
        if not invalid_found:
            rospy.loginfo(f"Trajectory validation PASSED: {total_points} points, all stable")
        else:
            rospy.logwarn(f"Trajectory validation FAILED: {unstable_count}/{total_points} unstable points detected")
            
        if invalid_found:
            rospy.logwarn(f"Trajectory for path_{self.current_path_id} invalid due to instability, retrying...")
            self.waiting_for_result = False
            # 继续生成新的路径尝试
            rospy.Timer(rospy.Duration(self.publish_delay), self.generate_next_path, oneshot=True)
            return

        # 验证通过，接受轨迹并继续处理
        rospy.loginfo(f"Trajectory validation successful, saving path_{self.current_path_id}")
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
        rospy.loginfo(f"  Pose bounds: X[{self.map_x_min:.1f}, {self.map_x_max:.1f}], Y[{self.map_y_min:.1f}, {self.map_y_max:.1f}]")
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

        # 对于外部地图，使用目标参数；对于生成地图，使用默认参数
        if self.use_external_map:
            # 使用外部地图的目标参数
            map_size_x = rospy.get_param('~target_map_size', self.map_size)
            map_size_y = map_size_x
            xy_res = rospy.get_param('~target_resolution', 0.4)
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
        expected_grid_size = int(target_map_size / target_resolution)
        actual_shape = grid_map_data['elevation'].shape
        
        rospy.loginfo("=== External Map Processing Validation ===")
        rospy.loginfo(f"Target map size: {target_map_size}m x {target_map_size}m")
        rospy.loginfo(f"Target resolution: {target_resolution}m")
        rospy.loginfo(f"Expected grid size: {expected_grid_size} x {expected_grid_size}")
        rospy.loginfo(f"Actual grid shape: {actual_shape}")
        rospy.loginfo(f"Bounds: {grid_map_data.get('bounds', 'Unknown')}")
        rospy.loginfo(f"Resolution: {grid_map_data.get('resolution', 'Unknown')}")
        rospy.loginfo("Processing method: Center cropping (preserves original resolution)")
        
        # 验证栅格尺寸
        if actual_shape == (expected_grid_size, expected_grid_size):
            rospy.loginfo("✓ Grid size matches target after cropping")
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
        
        rospy.loginfo("=" * 45)
        
if __name__ == '__main__':
    try:
        generator = TerrainDatasetGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        pass