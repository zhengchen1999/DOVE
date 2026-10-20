# Evaluating existing video results

`python eval_metrics.py` is the common entry point. It evaluates saved videos or
frame folders; it does not load DOVE or generate new restoration results.
Python 3.10 or later is required.

## Sources and metric definitions

This integration uses the supplied
[eval_metric.zip](https://drive.google.com/file/d/1mzinw3d-QAp0wn5djk1KaGTR3U-aTF9e/view)
for IQA, DOVER and the six-dimension VBench protocol, and
the supplied eWarp reference implementation. Archive and original source hashes are recorded
in [sources.json](sources.json).

| CLI name | Definition / output |
| --- | --- |
| `psnr`, `ssim`, `lpips`, `dists` | PyIQA full-reference frame metrics |
| `clipiqa`, `musiq`, `niqe`, `brisque` | PyIQA no-reference frame metrics |
| `dover` | Supplied `dover.yml`, `val-l1080p` sampling, fused score in [0, 1] |
| `ewarp` | RAFT-based warping error, lower is better, ×1000 |
| `vbench` | Supplied six-dimension weighted sum, with the raw dimensions retained |

eWarp uses prediction-to-prediction forward/backward RAFT flow, half-resolution
bilinear images, 20 iterations and the supplied `raft-things.pth`. The RAFT inputs
are **RGB in [0, 255]**; the photometric inputs are **RGB in [0, 1]**.
The score is the RGB-summed squared error divided by visible pixels, averaged
over all adjacent pairs and then equally across videos, and multiplied by 1000.
The supplied padding, boundary-clamped sampling, forward/backward and motion
boundary occlusion masks, and zero-error fully occluded pairs are preserved.
The computation in [raft/ewarp.py](raft/ewarp.py) follows the supplied reference.
RAFT and its CUDA resampling kernel are included; their BSD license is in
[raft/LICENSE](raft/LICENSE). eWarp requires CUDA.

> eWarp uses corrected RAFT input normalization, so its scores may differ from previously reported values.

## Install

Install an appropriate PyTorch/torchvision pair first, then:

```bash
python -m pip install -r metrics/requirements.txt
```

For eWarp, use an environment with CUDA PyTorch and a compatible CUDA toolkit:

```bash
python -m pip install -r metrics/requirements-ewarp.txt
python -m pip install --no-build-isolation ./metrics/raft/networks/resample2d_package
python metrics/prepare_sources.py --metrics-zip /path/to/eval_metric.zip
```

The helper verifies the supplied archive hash and installs the RAFT checkpoint
under `pretrained_models/metrics/`. Both supplied archives contain the identical
RAFT checkpoint, so the linked `eval_metric.zip` is sufficient for setup.
The eWarp computation is already included in this repository. If you have the
separate eWarp archive, pass its path with `--ewarp-zip /path/to/ewarp.zip`;
the archive filename does not affect verification.
Alternatively, extract the archive yourself and pass its `raft-things.pth` with
`--raft-model`. Weights are not stored in Git.
`TORCH_CUDA_ARCH_LIST` can be set when building the extension for another GPU.

DOVER and VBench are optional. Extract their supplied source using:

```bash
python metrics/prepare_sources.py --metrics-zip /path/to/eval_metric.zip
```

This places the optional source in the ignored `third_party/eval_metric-master/`
directory and the verified RAFT weight under `pretrained_models/metrics/`.
Existing extracted source can instead be selected with
`--metric-code-root /path/to/eval_metric-master`.
Install optional dependencies in the environment used for that metric:

```bash
# DOVER: the supplied package targets torch 1.13; a separate environment is fine.
python -m pip install -e third_party/eval_metric-master/DOVER
# Download DOVER.pth from the upstream release, or reuse an existing checkpoint:
# https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth

# VBench: install dependencies; the runner imports the supplied customized source.
python -m pip install 'vbench==0.1.5'
```

Pass `--dover-model /path/to/DOVER.pth` or place it in the supplied DOVER
directory's `pretrained_weights/`. DOVER's backbone initializer and PyIQA/VBench
may download their pretrained assets on first use. Separate metric environments
can run the same CLI with different `--metrics` selections.

## Commands

```bash
# Image quality, with exact matching frame counts and sizes:
python eval_metrics.py \
  --pred results/DOVE/UDM10 \
  --gt datasets/test/UDM10/GT \
  --metrics psnr,ssim,lpips,dists,clipiqa,musiq,niqe,brisque \
  --batch-size 1 --out results/metrics/UDM10

# eWarp:
python eval_metrics.py \
  --pred results/DOVE/UDM10 --metrics ewarp \
  --raft-model pretrained_models/metrics/raft-things.pth \
  --out results/metrics/UDM10

# DOVER:
python eval_metrics.py \
  --pred results/DOVE/UDM10 --metrics dover \
  --dover-model /path/to/DOVER.pth --out results/metrics/UDM10

# VBench, using the supplied six-dimension protocol:
python eval_metrics.py \
  --pred results/DOVE/UDM10 --metrics vbench --out results/metrics/UDM10

# Small check of already generated results, without frame truncation:
python eval_metrics.py \
  --pred results/DOVE/UDM10 --metrics ewarp --limit 2 \
  --out results/metrics/smoke
```

Use `--device cuda:1` to choose a GPU, or set `CUDA_VISIBLE_DEVICES`.
`--pred` / `--gt` also accept a single video or a single folder of frames.
Dataset roots may mix videos and sequence folders. Frames use natural filename
order (`1.png`, `2.png`, `10.png`); duplicate sequence names are rejected.
`--sequences 000,001` selects named videos. Non-video result JSON files are ignored.
Video metrics use existing videos directly. Frame folders are converted in a
private temporary directory to lossless RGB video at `--fps 8` by default.

Frame counts and sizes must match by default. To reproduce the source script's
explicit crop/truncation behavior, select `--alignment top-left` or `center`;
the evaluated shape is written to JSON and a warning is printed.
This alignment applies to the frame-metric inputs when GT is required. Video
metrics always evaluate the full prediction video independently of GT.

`--crop N` removes an N-pixel border for full-reference metrics.
By default, the source script's PyIQA model options are preserved (in particular,
`ssim` uses PyIQA's Y-channel default, while `psnr` uses RGB); effective options
and the PyIQA version are recorded in the report. `--test-y-channel` uses the source's
`0.257 R + 0.504 G + 0.098 B + 0.0625` conversion for PSNR/SSIM only.
LPIPS/DISTS retain RGB. No-reference metrics do not receive this border crop or
Y conversion. Per-frame and dataset averages retain full precision.

DOVER sampling uses `--seed 0` for each video. VBench preserves
the supplied sum of normalized dimensions (weights 1, 1, 1, 0.5, 1, 1), without
dividing by the sum of weights or converting it to a percentage. VBench runs at
the original input resolution.

## Output and validation

Reports are named `metrics_<selected_metrics>.json`, such as `metrics_ewarp.json`
and `metrics_dover.json`. Each report contains `per_sample`,
`average`, `count`, and `metadata` with paths, metric options, library versions,
and relevant weight hashes. Infinite PSNR is represented by the JSON string
`"Infinity"`. Missing GT, empty input, duplicate names, missing dependencies,
non-finite scores, or failed metrics stop the command with a nonzero exit.
A report is published only after all selected videos and metrics succeed.

The `finetune/scripts/eval_*.py` commands delegate to this same implementation.
Old CLI spellings `--out_path`, `--batch_mode`, `--test_y_channel`, `--is_center`,
and eWarp's `--model` are accepted. The unified JSON structure replaces
the older single-video-metric scalar structure.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Tests cover pairing and failures, explicit alignment, RGB/Y separation,
DOVER branch order, VBench aggregation, and eWarp normalization/padding/scaling.
The eWarp protocol tests require its compiled extension. See
[VALIDATION.md](VALIDATION.md) for the measured checks on existing videos.
