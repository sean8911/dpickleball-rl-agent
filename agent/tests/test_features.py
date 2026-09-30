"""Tests for the frozen 16-dim feature packing."""

import math
import unittest

import numpy as np

from agent import config
from agent.features import FEATURE_NAMES, pack_features


def _pack_center(**overrides):
    kwargs = dict(
        ball_xy=(config.IMG_W / 2.0, config.IMG_H / 2.0),
        ball_vxy=(0.0, 0.0),
        paddle_xy=(config.IMG_W / 2.0, config.IMG_H / 2.0),
        paddle_angle_rad=math.pi / 2.0,
        side_sign=1.0,
    )
    kwargs.update(overrides)
    return pack_features(**kwargs)


class TestPackFeatures(unittest.TestCase):
    def test_dim_and_dtype(self):
        f = _pack_center()
        self.assertEqual(f.shape, (config.FEATURE_DIM,))
        self.assertEqual(f.dtype, np.float32)
        self.assertEqual(len(FEATURE_NAMES), config.FEATURE_DIM)

    def test_center_positions_near_zero(self):
        f = _pack_center()
        for idx in (0, 1, 4, 5):
            self.assertAlmostEqual(f[idx], 0.0, delta=0.02)

    def test_angle_encoding(self):
        f = _pack_center()
        self.assertAlmostEqual(f[6], 1.0, places=5)  # sin(pi/2)
        self.assertAlmostEqual(f[7], 0.0, places=5)  # cos(pi/2)

    def test_velocity_clipping(self):
        f = _pack_center(ball_vxy=(100.0, -100.0))
        self.assertEqual(f[2], 1.0)
        self.assertEqual(f[3], -1.0)

    def test_side_sign_encoded(self):
        self.assertEqual(_pack_center(side_sign=1.0)[14], 1.0)
        self.assertEqual(_pack_center(side_sign=-1.0)[14], -1.0)

    def test_invalid_side_sign_raises(self):
        with self.assertRaises(ValueError):
            _pack_center(side_sign=0.5)

    def test_all_values_bounded(self):
        f = _pack_center(
            ball_xy=(0.0, 0.0),
            paddle_xy=(config.IMG_W - 1.0, config.IMG_H - 1.0),
            ball_vxy=(-50.0, 50.0),
            opponent_y=999.0,
        )
        self.assertTrue(np.all(f >= -1.0) and np.all(f <= 1.0))

    def test_opponent_default_is_mid_court(self):
        f = _pack_center(opponent_y=None)
        self.assertAlmostEqual(f[15], 0.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
