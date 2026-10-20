"""Regression checks for metric selection, pairing, preprocessing, and scales."""
import argparse
import math
from pathlib import Path
import tempfile
import unittest

from metrics.cli import requested_metrics, select_inputs, validate_results
from metrics.io import discover
from metrics.video import dover_fusion, vbench_fusion, VBENCH_RANGES


class InputTests(unittest.TestCase):
    def test_mixed_sequences_and_duplicate_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "seq10.mp4").touch()
            (root / "seq2").mkdir()
            (root / "seq2" / "0.png").touch()
            (root / "metrics.json").write_text("{}")
            self.assertEqual(list(discover(root)), ["seq2", "seq10"])
            (root / "seq10.mov").touch()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                discover(root)

    def test_empty_input_and_missing_reference_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "No videos"):
                discover(root)
            (root / "prediction.mp4").touch()
            (root / "gt").mkdir()
            (root / "gt" / "different.mp4").touch()
            args = argparse.Namespace(pred=str(root / "prediction.mp4"),
                                      gt=str(root / "gt"), sequences="", limit=None)
            with self.assertRaisesRegex(ValueError, "Missing GT"):
                select_inputs(args, ["psnr"])

    def test_metric_selection_normalizes_and_deduplicates_names(self):
        self.assertEqual(requested_metrics(" ewarp ,EWarp,ssim"), ["ewarp", "ssim"])
        with self.assertRaises(ValueError):
            requested_metrics("legacy_ewarp")

    def test_missing_or_invalid_scores_fail(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_results({"clip": {"psnr": 30}}, ["psnr", "ssim"])
        with self.assertRaisesRegex(ValueError, "Invalid"):
            validate_results({"clip": {"ewarp": math.nan}}, ["ewarp"])
        validate_results({"clip": {"psnr": math.inf}}, ["psnr"])


class FusionTests(unittest.TestCase):
    def test_dover_named_branch_order(self):
        self.assertAlmostEqual(dover_fusion(0.1107, -0.08285), 0.5)
        self.assertGreater(dover_fusion(0.18425, -0.08285), 0.5)

    def test_vbench_is_supplied_weighted_sum(self):
        self.assertAlmostEqual(vbench_fusion({k: v[1] for k, v in VBENCH_RANGES.items()}), 5.5)
        with self.assertRaises(ValueError):
            vbench_fusion({"dynamic_degree": 0.5})


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class PreprocessingTests(unittest.TestCase):
    def test_frame_tensor_matches_reference_layout(self):
        import numpy as np
        from unittest.mock import patch
        from metrics.io import load_sequence
        frames = [np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)] * 2
        expected = torch.stack([torch.from_numpy(f.transpose(2, 0, 1).copy()).float() / 255
                                for f in frames])
        with patch("metrics.io.read_frames", return_value=iter(frames)):
            actual = load_sequence("unused.mp4")
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(actual, expected))

    def test_y_channel_preserves_lpips_and_dists_rgb(self):
        from metrics.iqa import evaluation_inputs
        pred = torch.ones(2, 3, 16, 16)
        for name in ["lpips", "dists"]:
            p, g = evaluation_inputs(pred, pred, name, 2, True)
            self.assertEqual(tuple(p.shape), (2, 3, 12, 12))
        p, _ = evaluation_inputs(pred, pred, "psnr", 0, True)
        self.assertEqual(tuple(p.shape), (2, 1, 16, 16))
        self.assertAlmostEqual(p.mean().item(), 0.9215, places=5)

    def test_alignment_requires_explicit_opt_in(self):
        from metrics.iqa import align
        pred = torch.zeros(2, 3, 16, 16)
        gt = torch.zeros(3, 3, 18, 20)
        with self.assertRaisesRegex(ValueError, "prediction"):
            align(pred, gt, "strict", "clip")
        p, g = align(pred, gt, "top-left", "clip")
        self.assertEqual(p.shape, g.shape)


if __name__ == "__main__":
    unittest.main()
