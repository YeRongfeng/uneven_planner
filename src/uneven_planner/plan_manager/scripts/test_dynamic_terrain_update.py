#!/usr/bin/env python3
"""Deterministic regression for atomic terrain/occupancy replacement."""

import math

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, Header, MultiArrayDimension
import sensor_msgs.point_cloud2 as point_cloud2

from plan_manager.srv import (
    QueryTerrainMap,
    QueryTerrainMapRequest,
    UpdateTerrainMap,
    UpdateTerrainMapRequest,
)


HEIGHT = 20
WIDTH = 70
SOURCE_YAW_BINS = 63
INTERNAL_YAW_BINS = 64
RESOLUTION = 0.2
MIN_X = -7.0
MAX_X = 7.0
MIN_Y = -2.0
MAX_Y = 2.0


def make_plane(z_value):
    header = Header(stamp=rospy.Time.now(), frame_id="map")
    points = []
    for row in range(HEIGHT):
        y = MIN_Y + (row + 0.5) * RESOLUTION
        for column in range(WIDTH):
            x = MIN_X + (column + 0.5) * RESOLUTION
            points.append((x, y, z_value))
    return point_cloud2.create_cloud_xyz32(header, points)


def make_occupancy(occupied):
    """occupied is an iterable of (row, column, source_yaw), or yaw=None."""
    msg = Float32MultiArray()
    data = [0.0] * (HEIGHT * WIDTH * SOURCE_YAW_BINS)
    for row, column, yaw_index in occupied:
        yaw_indices = (
            range(SOURCE_YAW_BINS)
            if yaw_index is None else (yaw_index,))
        for source_yaw in yaw_indices:
            address = (
                (row * WIDTH + column) * SOURCE_YAW_BINS + source_yaw)
            data[address] = 1.0
    msg.data = data
    dim_h = MultiArrayDimension(
        label="height", size=HEIGHT,
        stride=WIDTH * SOURCE_YAW_BINS)
    dim_w = MultiArrayDimension(
        label="width", size=WIDTH, stride=SOURCE_YAW_BINS)
    dim_yaw = MultiArrayDimension(
        label="yaw", size=SOURCE_YAW_BINS, stride=1)
    msg.layout.dim = [dim_h, dim_w, dim_yaw]
    msg.layout.data_offset = 0
    return msg


def source_bin_for_internal(internal_index):
    map_origin_yaw = -(2.0 * math.pi + 0.05) / 2.0
    theta = map_origin_yaw + (internal_index + 0.5) * 0.1
    phase = (theta + math.pi) % (2.0 * math.pi)
    return int(math.floor(
        phase * SOURCE_YAW_BINS / (2.0 * math.pi)))


def internal_center_for_source(source_index):
    for internal_index in range(INTERNAL_YAW_BINS):
        if source_bin_for_internal(internal_index) == source_index:
            map_origin_yaw = -(2.0 * math.pi + 0.05) / 2.0
            return map_origin_yaw + (internal_index + 0.5) * 0.1
    raise AssertionError(f"source yaw {source_index} is not represented")


class Worker:
    def __init__(self, index):
        namespace = f"/dataset_worker_{index}"
        update_name = namespace + "/update_terrain_map"
        query_name = namespace + "/query_terrain_map"
        rospy.wait_for_service(update_name, timeout=90.0)
        rospy.wait_for_service(query_name, timeout=90.0)
        self.update = rospy.ServiceProxy(update_name, UpdateTerrainMap)
        self.query = rospy.ServiceProxy(query_name, QueryTerrainMap)
        self.request_id = 0

    def apply(self, environment_id, z_value, occupancy):
        self.request_id += 1
        req = UpdateTerrainMapRequest()
        req.request_id = self.request_id
        req.environment_id = environment_id
        req.pointcloud = make_plane(z_value)
        req.occupancy_hwy = occupancy
        req.min_x = MIN_X
        req.min_y = MIN_Y
        req.max_x = MAX_X
        req.max_y = MAX_Y
        req.resolution = RESOLUTION
        response = self.update(req)
        assert response.success, response.message
        assert response.request_id == self.request_id
        assert response.environment_id == environment_id
        assert response.source_height == HEIGHT
        assert response.source_width == WIDTH
        assert response.source_yaw_bins == SOURCE_YAW_BINS
        assert response.internal_yaw_bins == INTERNAL_YAW_BINS
        return response

    def probe(self, x, y, yaw):
        req = QueryTerrainMapRequest(x=x, y=y, yaw=yaw)
        response = self.query(req)
        assert response.success, response.message
        return response


def close(actual, expected, tolerance=1e-3):
    assert abs(actual - expected) <= tolerance, (
        f"{actual} != {expected} within {tolerance}")


def main():
    rospy.init_node("test_dynamic_terrain_update", anonymous=True)
    worker0 = Worker(0)
    worker1 = Worker(1)
    free = make_occupancy([])

    baseline0 = worker0.apply(100, 0.2, free)
    baseline1 = worker1.apply(200, 0.8, free)
    assert baseline0.occupied_voxel_count == 0
    assert baseline1.occupied_voxel_count == 0
    close(worker0.probe(0.1, 0.1, 0.0).z, 0.2)
    close(worker1.probe(0.1, 0.1, 0.0).z, 0.8)

    row, column, source_yaw = 17, 63, 11
    x = MIN_X + (column + 0.5) * RESOLUTION
    y = MIN_Y + (row + 0.5) * RESOLUTION
    yaw = internal_center_for_source(source_yaw)
    sentinel = make_occupancy([(row, column, source_yaw)])
    sentinel_response = worker0.apply(100, 0.4, sentinel)
    assert sentinel_response.occupied_voxel_count >= 1
    correct = worker0.probe(x, y, yaw)
    assert correct.occupancy_se2 == 1
    assert worker0.probe(x - RESOLUTION, y, yaw).occupancy_se2 == 0
    # The other worker must retain both its terrain and occupancy.
    isolated = worker1.probe(x, y, yaw)
    assert isolated.occupancy_se2 == 0
    close(isolated.z, 0.8)

    all_yaw = make_occupancy([(row, column, None)])
    all_yaw_response = worker0.apply(100, 0.6, all_yaw)
    assert all_yaw_response.occupied_xy_count == 1
    assert all_yaw_response.occupied_voxel_count == INTERNAL_YAW_BINS
    assert worker0.probe(x, y, -math.pi).occupancy_se2 == 1
    assert worker0.probe(x, y, math.pi).occupancy_se2 == 1

    restored = worker0.apply(100, 1.0, free)
    assert restored.occupied_xy_count == 0
    assert restored.occupied_voxel_count == 0
    restored_probe = worker0.probe(x, y, yaw)
    assert restored_probe.occupancy_se2 == 0
    close(restored_probe.z, 1.0)
    close(worker1.probe(x, y, yaw).z, 0.8)
    print("PASS: atomic terrain update, H/W, 63->64 yaw, replacement, isolation")


if __name__ == "__main__":
    main()
