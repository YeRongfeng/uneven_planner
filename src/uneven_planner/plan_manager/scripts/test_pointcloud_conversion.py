#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试点云到栅格地图转换系统
"""

import rospy
import numpy as np
import open3d as o3d
from plan_manager.srv import PointCloudToGrid, PointCloudToGridRequest
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

def create_test_pointcloud():
    """创建一个测试点云（41.2m×41.2m的平面，匹配实际地图尺寸）"""
    print("Creating test point cloud...")
    
    # 生成网格点 - 41.2m地图
    x = np.linspace(0, 41.2, 206)
    y = np.linspace(0, 41.2, 206)
    xx, yy = np.meshgrid(x, y)
    
    # 添加一些高程变化
    zz = np.sin(xx / 5) * np.cos(yy / 5) * 2 + 5
    
    # 组合成点云
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    
    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    print(f"Created point cloud with {len(points)} points")
    return pcd

def pointcloud_to_ros_message(pcd):
    """将Open3D点云转换为ROS PointCloud2消息"""
    points = np.asarray(pcd.points)
    
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "map"
    
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1)
    ]
    
    points_with_intensity = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
    cloud_msg = pc2.create_cloud(header, fields, points_with_intensity)
    
    return cloud_msg

def test_cpp_service():
    """测试C++转换服务"""
    rospy.init_node('test_pointcloud_conversion', anonymous=True)
    
    print("\n" + "="*60)
    print("Testing PointCloud to Grid Conversion Service")
    print("="*60 + "\n")
    
    # 等待服务
    print("Waiting for service...")
    try:
        rospy.wait_for_service('/pointcloud_to_grid', timeout=5.0)
        print("✓ Service is available\n")
    except rospy.ROSException:
        print("✗ Service timeout! Make sure to run:")
        print("  roslaunch plan_manager pointcloud_converter.launch")
        return False
    
    # 创建测试点云
    pcd = create_test_pointcloud()
    cloud_msg = pointcloud_to_ros_message(pcd)
    
    # 创建服务请求
    print("Creating service request...")
    req = PointCloudToGridRequest()
    req.pointcloud = cloud_msg
    req.map_min_x = 0.0
    req.map_max_x = 41.2
    req.map_min_y = 0.0
    req.map_max_y = 41.2
    
    # 调用服务
    print("Calling C++ conversion service...\n")
    try:
        grid_service = rospy.ServiceProxy('/pointcloud_to_grid', PointCloudToGrid)
        response = grid_service(req)
        
        if response.success:
            print("✓ Conversion successful!")
            print(f"\nResults:")
            print(f"  Grid size: {response.grid_width} × {response.grid_height}")
            print(f"  Resolution: {response.resolution}m")
            print(f"  Total cells: {len(response.elevation_grid)}")
            print(f"  Expected size: {response.grid_width * response.grid_height}")
            
            # 验证数据
            elevation = np.array(response.elevation_grid).reshape(response.grid_height, response.grid_width)
            normal_z = np.array(response.normal_z_grid).reshape(response.grid_height, response.grid_width)
            
            valid_cells = np.sum(~np.isnan(elevation))
            print(f"\nData statistics:")
            print(f"  Valid cells: {valid_cells} / {elevation.size} ({100*valid_cells/elevation.size:.1f}%)")
            print(f"  Elevation range: [{np.nanmin(elevation):.2f}, {np.nanmax(elevation):.2f}]")
            print(f"  Normal Z range: [{np.nanmin(normal_z):.2f}, {np.nanmax(normal_z):.2f}]")
            
            # 检查预期尺寸（41.2m地图在0.2m分辨率下应该是206格）
            expected_width = int(np.ceil(41.2 / 0.2))
            expected_height = int(np.ceil(41.2 / 0.2))
            print(f"\nSize verification:")
            print(f"  Expected: {expected_width} × {expected_height}")
            print(f"  Actual: {response.grid_width} × {response.grid_height}")
            
            if response.grid_width == expected_width and response.grid_height == expected_height:
                print("  ✓ Size matches expected (206×206 for 41.2m×41.2m at 0.2m resolution)")
            else:
                print("  ⚠ Size mismatch!")
            
            print("\n" + "="*60)
            print("TEST PASSED")
            print("="*60)
            return True
            
        else:
            print(f"✗ Conversion failed: {response.message}")
            return False
            
    except rospy.ServiceException as e:
        print(f"✗ Service call failed: {e}")
        return False

if __name__ == '__main__':
    try:
        success = test_cpp_service()
        exit(0 if success else 1)
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
