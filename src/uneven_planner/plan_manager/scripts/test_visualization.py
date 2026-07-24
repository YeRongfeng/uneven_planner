#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试地图可视化功能的独立脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors
import os

def create_test_data():
    """创建测试数据"""
    # 创建一个简单的测试地图
    height, width = 20, 20
    
    # 生成高程数据（简单的山丘）
    x = np.linspace(-2, 2, width)
    y = np.linspace(-2, 2, height)
    X, Y = np.meshgrid(x, y)
    
    # 高程：高斯山丘
    elevation = 2 * np.exp(-(X**2 + Y**2))
    
    # 计算法向量（基于高程梯度）
    dy, dx = np.gradient(elevation)
    normal_x = -dx
    normal_y = -dy
    normal_z = np.ones_like(elevation)
    
    # 归一化法向量
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm
    
    return elevation, normal_x, normal_y, normal_z, (-2, 2, -2, 2)

def rodrigues_rotation(vector, axis, angle):
    """罗德里格旋转公式"""
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rotated = (vector * cos_angle +
              np.cross(axis, vector) * sin_angle +
              axis * np.dot(axis, vector) * (1 - cos_angle))

    return rotated

def create_tilted_patch(cx, cy, cz, nx, ny, nz, dx, dy):
    """创建倾斜面片"""
    # 归一化法向量
    normal = np.array([nx, ny, nz])
    normal_length = np.linalg.norm(normal)
    if normal_length < 1e-6:
        normal = np.array([0, 0, 1])
    else:
        normal = normal / normal_length

    # 创建局部坐标系的矩形顶点
    local_vertices = np.array([
        [-dx/2, -dy/2, 0],
        [dx/2, -dy/2, 0],
        [dx/2, dy/2, 0],
        [-dx/2, dy/2, 0]
    ])

    # 计算旋转
    z_axis = np.array([0, 0, 1])

    if abs(np.dot(normal, z_axis)) > 0.999:
        if np.dot(normal, z_axis) < 0:
            local_vertices[:, 2] *= -1
        rotated_vertices = local_vertices
    else:
        rotation_axis = np.cross(z_axis, normal)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        cos_angle = np.dot(z_axis, normal)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        rotated_vertices = []
        for vertex in local_vertices:
            rotated_vertex = rodrigues_rotation(vertex, rotation_axis, angle)
            rotated_vertices.append(rotated_vertex)
        rotated_vertices = np.array(rotated_vertices)

    # 平移到实际位置
    world_vertices = rotated_vertices + np.array([cx, cy, cz])

    return world_vertices

def test_3d_visualization():
    """测试3D可视化"""
    elevation, normal_x, normal_y, normal_z, bounds = create_test_data()
    height, width = elevation.shape

    # 创建坐标
    x = np.linspace(bounds[0], bounds[1], width)
    y = np.linspace(bounds[2], bounds[3], height)

    # 栅格大小
    dx = (bounds[1] - bounds[0]) / width
    dy = (bounds[3] - bounds[2]) / height

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 收集面片
    patches = []
    colors_list = []

    for i in range(0, height, 2):  # 减少密度以便测试
        for j in range(0, width, 2):
            cx = x[j]
            cy = y[i]
            cz = elevation[i, j]
            nx = normal_x[i, j]
            ny = normal_y[i, j]
            nz = normal_z[i, j]

            # 创建倾斜面片
            patch_vertices = create_tilted_patch(cx, cy, cz, nx, ny, nz, dx*1.5, dy*1.5)

            patches.append(patch_vertices)
            colors_list.append(cz)
    
    # 创建3D面片集合
    poly_collection = Poly3DCollection(patches, alpha=0.8)
    
    # 设置颜色映射
    norm = colors.Normalize(vmin=min(colors_list), vmax=max(colors_list))
    poly_collection.set_facecolors(plt.cm.terrain(norm(colors_list)))
    
    ax.add_collection3d(poly_collection)
    
    # 设置坐标轴
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('Test 3D Grid Map')
    
    # 设置坐标轴范围
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_zlim(min(colors_list), max(colors_list))
    
    # 添加颜色条
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.terrain)
    mappable.set_array(colors_list)
    plt.colorbar(mappable, ax=ax, shrink=0.8, label='Elevation (m)')
    
    plt.savefig('test_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved test_3d.png")

def test_2d_visualization():
    """测试2D可视化"""
    elevation, normal_x, normal_y, normal_z, bounds = create_test_data()
    height, width = elevation.shape
    
    # 创建坐标
    x = np.linspace(bounds[0], bounds[1], width)
    y = np.linspace(bounds[2], bounds[3], height)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制高程热力图
    im = ax.imshow(elevation, extent=[bounds[0], bounds[1], bounds[2], bounds[3]], 
                  origin='lower', cmap='terrain', aspect='equal')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Elevation (m)', fontsize=12)
    
    # 绘制法向量箭头
    step = 2  # 步长
    arrow_scale = 0.2
    
    for i in range(0, height, step):
        for j in range(0, width, step):
            cx = x[j]
            cy = y[i]
            nx = normal_x[i, j]
            ny = normal_y[i, j]
            
            ax.arrow(cx, cy, nx * arrow_scale, ny * arrow_scale,
                    head_width=arrow_scale*0.3, head_length=arrow_scale*0.2,
                    fc='red', ec='red', alpha=0.7, linewidth=1)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Test 2D Grid Map\n(Elevation Heatmap + Normal Vectors)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.savefig('test_2d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved test_2d.png")

if __name__ == '__main__':
    print("Testing visualization functions...")
    test_3d_visualization()
    test_2d_visualization()
    print("Visualization test completed!")
