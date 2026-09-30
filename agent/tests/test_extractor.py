"""Extractor tests against recorded Unity frames.

Skips when the smoke-test recording is absent so the suite stays
runnable on machines without the dataset.
"""

import unittest
from pathlib import Path

import numpy as np

from agent import config
from agent.extractor import OpenCVStateExtractor

SMOKE_CHUNK = (
    config.RECORDINGS_DIR / "smoke_test" / "chunk_0000.npz"
)


@unittest.skipUnless(
    SMOKE_CHUNK.is_file(), "smoke recording not available"
)
class TestExtractorOnUnityFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = np.load(SMOKE_CHUNK)["frames"][:200]

    def test_paddles_detected_on_most_frames(self):
        ext = OpenCVStateExtractor(side="right")
        paddle_hits = opponent_hits = 0
        for frame in self.frames:
            det = ext.detect(frame)
            paddle_hits += det.paddle_xy is not None
            opponent_hits += det.opponent_xy is not None
        n = len(self.frames)
        self.assertGreater(paddle_hits / n, 0.95)
        self.assertGreater(opponent_hits / n, 0.95)

    def test_paddle_sides_are_split_correctly(self):
        right = OpenCVStateExtractor(side="right")
        left = OpenCVStateExtractor(side="left")
        half = config.IMG_W / 2.0
        for frame in self.frames[:50]:
            det_r = right.detect(frame)
            det_l = left.detect(frame)
            if det_r.paddle_xy is not None:
                self.assertGreaterEqual(det_r.paddle_xy[0], half)
            if det_l.paddle_xy is not None:
                self.assertLess(det_l.paddle_xy[0], half)

    def test_features_shape_and_bounds(self):
        ext = OpenCVStateExtractor(side="left")
        for frame in self.frames[:20]:
            f = ext.features_from_image(frame)
            self.assertEqual(f.shape, (config.FEATURE_DIM,))
            self.assertTrue(np.all(f >= -1.0) and np.all(f <= 1.0))

    def test_accepts_chw_float_layout(self):
        ext = OpenCVStateExtractor(side="right")
        hwc_uint8 = self.frames[0]
        chw_float = np.transpose(
            hwc_uint8.astype(np.float32) / 255.0, (2, 0, 1)
        )
        det_a = ext.detect(hwc_uint8)
        det_b = ext.detect(chw_float)
        self.assertEqual(
            det_a.ball_xy is None, det_b.ball_xy is None
        )
        if det_a.ball_xy is not None:
            self.assertAlmostEqual(
                det_a.ball_xy[0], det_b.ball_xy[0], delta=0.5
            )

    def test_invalid_input_raises(self):
        ext = OpenCVStateExtractor(side="right")
        with self.assertRaises(ValueError):
            ext.detect(np.zeros((84, 168), dtype=np.uint8))


class TestExtractorValidation(unittest.TestCase):
    def test_bad_side_raises(self):
        with self.assertRaises(ValueError):
            OpenCVStateExtractor(side="up")


if __name__ == "__main__":
    unittest.main()
