#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
测试位姿边界生成的简单脚本
"""

import rospy
import random
import numpy as np

def test_pose_bounds():
    """测试位姿边界"""
    
    # 模拟40m地图的参数
    target_map_size = 40.0
    safety_margin = 5.0
    
    map_x_min = -target_map_size / 2 + safety_margin
    map_x_max = target_map_size / 2 - safety_margin
    map_y_min = -target_map_size / 2 + safety_margin
    map_y_max = target_map_size / 2 - safety_margin
    
    print(f"Map size: {target_map_size}m")
    print(f"Safety margin: {safety_margin}m")
    print(f"Pose bounds: X[{map_x_min:.1f}, {map_x_max:.1f}], Y[{map_y_min:.1f}, {map_y_max:.1f}]")
    
    # 生成一些测试位姿
    print("\nTest poses:")
    for i in range(10):
        x = random.uniform(map_x_min, map_x_max)
        y = random.uniform(map_y_min, map_y_max)
        yaw = random.uniform(-np.pi, np.pi)
        print(f"  Pose {i}: [{x:.3f}, {y:.3f}, {yaw:.3f}]")

if __name__ == '__main__':
    test_pose_bounds()
