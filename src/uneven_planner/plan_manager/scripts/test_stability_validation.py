#!/usr/bin/env python3

import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np

from stability_validation import (
    build_periodic_signed_stability_esdf,
    sample_periodic_stability_esdf,
    validate_trajectory_stability,
)


class StabilityValidationTest(unittest.TestCase):
    def test_numerical_parity_with_mpt_production_functions(self):
        mpt_root = Path(os.environ.get("MPT_ROOT", "/home/yrf/MPT"))
        if not (mpt_root / "dataLoader_dit.py").is_file():
            self.skipTest("MPT_ROOT does not point to an MPT checkout")
        sys.path.insert(0, str(mpt_root))
        try:
            import torch
            from dataLoader_dit import generate_sdf_from_yaw_stability
            from grad_optimizer import _sample_cost_map_at_xy_yaw

            rng = np.random.default_rng(17)
            binary = (rng.random((3, 5, 6)) > 0.32).astype(np.float32)
            binary[0, 0, 0] = 0.0
            binary[2, 4, 5] = 1.0
            resolution = 0.2
            yaw_weight = 1.4
            ours_esdf = build_periodic_signed_stability_esdf(
                binary, resolution, yaw_weight)
            mpt_esdf = generate_sdf_from_yaw_stability(
                binary,
                voxel_size_xy=resolution,
                yaw_weight=yaw_weight,
            )
            np.testing.assert_allclose(
                ours_esdf, mpt_esdf, rtol=0.0, atol=1e-6)

            bounds = (-1.0, 0.0, 2.0, 2.6)
            epsilon = 1e-4
            probes = np.array([
                [-0.7, 2.5, 0.0],       # exact non-square cell center
                [-0.6, 2.3, 0.37],      # half-cell XY interpolation
                [-0.9, 2.1, math.pi - epsilon],
                [-0.9, 2.1, -math.pi + epsilon],
            ], dtype=np.float32)
            ours = sample_periodic_stability_esdf(
                probes, ours_esdf, bounds, resolution)
            mpt = _sample_cost_map_at_xy_yaw(
                torch.from_numpy(probes[None, :, :2]),
                torch.from_numpy(probes[None, :, 2]),
                torch.from_numpy(mpt_esdf),
                {
                    "origin": (bounds[0], bounds[2], -math.pi),
                    "resolution": resolution,
                    "size": (5, 3, 6),
                },
                "cpu",
            )[0].detach().cpu().numpy()
            np.testing.assert_allclose(ours, mpt, rtol=0.0, atol=2e-6)
        finally:
            if sys.path[0] == str(mpt_root):
                sys.path.pop(0)

    def test_non_square_axes_cell_centers_and_half_cell_interpolation(self):
        field = np.empty((2, 3, 4), dtype=np.float32)
        for row in range(2):
            for column in range(3):
                for yaw_bin in range(4):
                    field[row, column, yaw_bin] = (
                        100 * row + 10 * column + yaw_bin)
        bounds = (0.0, 3.0, 0.0, 2.0)
        values = sample_periodic_stability_esdf(
            np.array([
                [1.5, 0.5, 0.0],  # row=0, col=1, yaw-bin=2
                [2.5, 1.5, 0.0],  # row=1, col=2, yaw-bin=2
                [1.0, 0.5, 0.0],  # halfway between col=0 and col=1
            ]),
            field,
            bounds,
            resolution=1.0,
        )
        np.testing.assert_allclose(values, [12.0, 122.0, 7.0])

    def test_yaw_interpolation_wraps_periodically(self):
        field = np.zeros((2, 3, 4), dtype=np.float32)
        field[:, :, 3] = 1.0
        epsilon = 1e-4
        values = sample_periodic_stability_esdf(
            np.array([
                [0.5, 0.5, math.pi - epsilon],
                [0.5, 0.5, -math.pi + epsilon],
            ]),
            field,
            (0.0, 3.0, 0.0, 2.0),
            resolution=1.0,
        )
        self.assertLess(abs(float(values[0] - values[1])), 1e-3)

    def test_periodic_signed_esdf_and_hard_margin_gate(self):
        binary = np.ones((2, 3, 4), dtype=np.float32)
        binary[0, 0, 0] = 0.0
        esdf = build_periodic_signed_stability_esdf(
            binary, voxel_size_xy=1.0, yaw_weight=1.4)
        self.assertEqual(esdf.shape, binary.shape)
        self.assertLess(float(esdf[0, 0, 0]), 0.0)
        self.assertAlmostEqual(
            float(esdf[0, 0, 1]),
            float(esdf[0, 0, 3]),
            places=5,
        )

        field = np.full((2, 3, 4), 0.2, dtype=np.float32)
        accepted = validate_trajectory_stability(
            np.array([[0.5, 0.5, 0.0], [2.5, 1.5, 0.0]]),
            field,
            (0.0, 3.0, 0.0, 2.0),
            resolution=1.0,
            d_safe=0.15,
        )
        self.assertTrue(accepted["valid"])
        field[0, 0, :] = 0.1
        rejected = validate_trajectory_stability(
            np.array([[0.5, 0.5, 0.0]]),
            field,
            (0.0, 3.0, 0.0, 2.0),
            resolution=1.0,
            d_safe=0.15,
        )
        self.assertFalse(rejected["valid"])
        self.assertEqual(rejected["first_invalid_reason"], "below_d_safe")


if __name__ == "__main__":
    unittest.main()
