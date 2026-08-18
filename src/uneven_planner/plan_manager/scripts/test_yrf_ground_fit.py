#!/usr/bin/env python3
"""YRF ground fit at our 20 m / 0.2 m cell size."""

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_laz_terrain_map import (
    fit_yrf_ground_grid,
    yrf_horizontal_params,
)


class YrfHorizontalParamsTest(unittest.TestCase):
    def test_internals_stay_at_yrf_metres(self):
        self.assertEqual(yrf_horizontal_params(0.2), (0.2, 0.4))
        self.assertEqual(yrf_horizontal_params(0.05), (0.2, 0.4))


class YrfGroundFitTest(unittest.TestCase):
    def test_flat_ground_stays_flat(self):
        axis = np.arange(-9.85, 9.9, 0.1)
        xx, yy = np.meshgrid(axis, axis)
        xyz = np.column_stack((
            xx.ravel(), yy.ravel(), np.full(xx.size, 10.0)))
        _, _, elevation, _, valid, _, _, diagnostics = fit_yrf_ground_grid(
            xyz, 20.0, 0.2)
        self.assertTrue(valid.all())
        self.assertEqual(elevation.shape, (100, 100))
        self.assertAlmostEqual(diagnostics["voxel_size_m"], 0.2)
        self.assertAlmostEqual(diagnostics["coarse_resolution_m"], 0.4)
        self.assertLess(float(np.ptp(elevation)), 1e-6)
        self.assertAlmostEqual(float(np.mean(elevation)), 10.0, places=5)
        self.assertEqual(diagnostics["obstacle_cell_count"], 0)

    def test_canopy_stays_out_of_elevation_and_in_obstacle(self):
        axis = np.arange(-9.85, 9.9, 0.1)
        xx, yy = np.meshgrid(axis, axis)
        ground = np.column_stack((
            xx.ravel(), yy.ravel(), np.full(xx.size, 10.0)))
        tree_x, tree_y = np.meshgrid(
            np.linspace(0.9, 1.1, 6), np.linspace(0.9, 1.1, 6))
        tree = np.column_stack((
            tree_x.ravel(), tree_y.ravel(), np.full(tree_x.size, 11.5)))
        xyz = np.vstack((ground, tree))
        local_x, local_y, elevation, _, _, _, _, diagnostics = (
            fit_yrf_ground_grid(xyz, 20.0, 0.2))
        tree_cells = (
            (np.abs(local_x - 1.0) <= 0.3)
            & (np.abs(local_y - 1.0) <= 0.3))
        self.assertGreater(int(np.count_nonzero(tree_cells)), 0)
        self.assertLess(float(np.max(np.abs(elevation - 10.0))), 0.05)
        obstacle_height = diagnostics["obstacle_height"]
        self.assertGreater(float(np.max(obstacle_height[tree_cells])), 1.2)
        self.assertGreater(diagnostics["obstacle_cell_count"], 0)

    def test_ultra_low_flyers_do_not_set_the_ground(self):
        axis = np.arange(-9.85, 9.9, 0.1)
        xx, yy = np.meshgrid(axis, axis)
        ground = np.column_stack((
            xx.ravel(), yy.ravel(), np.full(xx.size, 10.0)))
        flyers = np.array([
            [-4.0, 1.0, -300.0],
            [3.0, -2.0, -250.0],
            [8.0, 8.0, -180.0],
        ])
        xyz = np.vstack((ground, flyers))
        _, _, elevation, _, _, _, _, diagnostics = fit_yrf_ground_grid(
            xyz, 20.0, 0.2)
        self.assertEqual(diagnostics["low_flyer_removed"], 3)
        self.assertLess(float(np.max(np.abs(elevation - 10.0))), 0.05)


if __name__ == "__main__":
    unittest.main()
