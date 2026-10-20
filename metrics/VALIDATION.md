# Validation on existing videos — 2026-10-20

Host: **leo**, NVIDIA RTX A6000. Two already generated DiffST UDM10 clips,
`000.mp4` and `001.mp4`, were used solely as metric regression inputs. Each has
33 frames at 1272 × 720. No restoration model was run and no videos were regenerated.
These numbers are smoke-test results, not a reproduction of DOVE paper scores.

Both input SHA-256 values match the server's pre-existing video validation record:

| Clip | SHA-256 |
| --- | --- |
| 000 | `0230940b0f6fc36091325721adbe63c6042f9a69e20b2a848a42a7ac56ba2e08` |
| 001 | `1d68b22a778cf6cbab0121ad26a8aa42a40e928bb9afe944f147f2e6d9a4d7d2` |

## eWarp v3

The new CLI and the unmodified `eval_ewarp_v3.py` from the supplied archive ran on
the same two complete videos with the same checkpoint and environment.

| Clip | New CLI | Supplied v3 | Absolute difference |
| --- | ---: | ---: | ---: |
| 000 | 1.105746952816844 | 1.105746952816844 | 0 |
| 001 | 0.256330706179142 | 0.256330706179142 | 0 |
| Mean | 0.681038829497993 | 0.681038829497993 | 0 |

Environment: PyTorch 2.3.1+cu121. RAFT checkpoint SHA-256:
`fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`.
Both supplied archives contain exactly this checkpoint.

The resample2d extension was also built from this repository with CUDA 12.4 for
SM 8.6, in an isolated build directory. The build emitted a CUDA minor-version
warning and upstream deprecation warnings but succeeded. Running clip 000 with
that newly built extension again produced **1.105746952816844**.

All vendored Python computation ASTs match the supplied v3 after excluding
package imports/path setup. The calculation and CUDA kernel were not replaced.

## Image metrics and DOVER

Eight image metrics were evaluated on all 66 prediction frames against the
existing corresponding GT videos, with strict shape matching, batch size 1,
no border crop, and the supplied PyIQA defaults. SSIM uses its default Y channel;
PSNR uses RGB. Image tensors retain the original contiguous NCHW layout.

| Metric | Mean across the two videos |
| --- | ---: |
| PSNR | 27.3955584150 |
| SSIM | 0.7677342109 |
| LPIPS | 0.2200230594 |
| DISTS | 0.1384453015 |
| CLIP-IQA | 0.3243538613 |
| MUSIQ | 56.8777612050 |
| NIQE | 4.8352482836 |
| BRISQUE | 26.8353049538 |
| DOVER | 0.7585717691 |

Image-metric environment: PyIQA 0.1.13, PyTorch 2.7.1+cu126. The new output retains
full precision, whereas the supplied reference function rounds each video's
score to four decimal places. All 16 per-video metric values match the reference
at that precision; the maximum absolute difference before rounding is
**0.0000485315352**, within the expected four-decimal rounding tolerance.

DOVER environment: PyTorch 1.13.1+cu117. The supplied `dover.yml` and
`evaluate_one_video.py` were used for comparison, with seed 0 applied immediately
before each video's sampling. Both clips matched exactly:
`0.83222687857049` and `0.6849166596505307` (maximum difference **0**).
DOVER checkpoint SHA-256:
`f4a42c0bbc94c94dd7409e7f40887d44c5c30314d1d09e7edf03cc35813b4838`.

## Additional checks and scope

- All **11 regression tests passed on leo**, including two eWarp protocol tests.
- Empty input, duplicate sequence names, missing GT, incomplete/non-finite
  results, and unintended shape mismatches are rejected.
- The Y-channel option retains RGB inputs for LPIPS/DISTS.
- The archive setup helper was exercised with both supplied archives.
- VBench's six-dimension fusion was checked against the supplied ranges and
  weights. Its full pretrained-model evaluation was **not run** in this small
  validation; no measured VBench result is claimed.
- No inference or training code was changed.

The server retains complete JSON reports, input hashes, source archives,
reference comparison code and the isolated extension build in the task's
`DOVE_metric_validation_20261020` directory.
