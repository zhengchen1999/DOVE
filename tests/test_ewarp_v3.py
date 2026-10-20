"""Lock the normalization, padding, RGB sum, and occlusion conventions."""
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
    from metrics.raft import ewarp_v3
except ImportError:
    ewarp_v3 = None


@unittest.skipIf(ewarp_v3 is None, "eWarp dependencies/extension not installed")
class EwarpProtocolTests(unittest.TestCase):
    def run_pair(self, occluded):
        class Capture:
            def __init__(self, *args):
                self.index = 0

            def isOpened(self):
                return True

            def read(self):
                self.index += 1
                if self.index > 2:
                    return False, None
                return True, np.full((256, 260, 3), 51 * self.index, dtype=np.uint8)

        calls = []

        def model(first, second, **kwargs):
            calls.append((first.clone(), second.clone(), kwargs))
            flow = torch.zeros(1, 2, *first.shape[-2:])
            return flow, flow

        def occlusion(backward, forward, image):
            return np.full(image.shape[-2:], occluded, dtype=float), image

        with patch.object(ewarp_v3.cv2, "VideoCapture", Capture), \
             patch.object(ewarp_v3.warp_utils, "detect_occlusion", occlusion):
            result = ewarp_v3.compute_video_warping_error("unused.mp4", model, "cpu")
        return result.item() * 1000, calls

    def test_rgb255_raft_and_rgb_squared_error_scale(self):
        score, calls = self.run_pair(0)
        self.assertAlmostEqual(score, 120.0, places=4)
        self.assertEqual(len(calls), 2)
        self.assertEqual(tuple(calls[0][0].shape), (1, 3, 128, 136))
        self.assertAlmostEqual(calls[0][0].mean().item(), 51.0, places=4)
        self.assertAlmostEqual(calls[0][1].mean().item(), 102.0, places=4)
        self.assertEqual(calls[0][2], {"iters": 20, "test_mode": True})

    def test_fully_occluded_pair_remains_zero(self):
        score, _ = self.run_pair(1)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
