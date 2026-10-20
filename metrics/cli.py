"""Unified evaluation of existing restoration results; never runs inference."""
import argparse
import json
import math
from pathlib import Path
import tempfile

from .io import discover
from .iqa import FULL_REFERENCE, SUPPORTED

ROOT = Path(__file__).resolve().parents[1]


def build_parser(default_metrics="psnr,ssim,clipiqa"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="Video, frame folder, or directory of sequences")
    parser.add_argument("--gt", default="", help="Matching reference video or sequence directory")
    parser.add_argument("--out", "--out_path", default="", help="Output directory (default: prediction directory)")
    parser.add_argument("--metrics", default=default_metrics,
                        help="Comma-separated psnr,ssim,lpips,dists,clipiqa,musiq,niqe,brisque,dover,ewarp,vbench")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=None, help="IQA frames per batch (default: 1)")
    parser.add_argument("--batch_mode", "--batch-mode", action="store_true",
                        help="Use batches of 8 IQA frames unless --batch-size is set")
    parser.add_argument("--crop", type=int, default=0, help="Border crop for full-reference image metrics")
    parser.add_argument("--test_y_channel", "--test-y-channel", action="store_true",
                        help="Use the supplied Y conversion for PSNR/SSIM; LPIPS/DISTS remain RGB")
    parser.add_argument("--alignment", choices=["strict", "top-left", "center"], default="strict",
                        help="Strict shape matching by default; crop/truncate only when explicitly selected")
    parser.add_argument("--is_center", action="store_true", help="Alias for --alignment center")
    parser.add_argument("--sequences", default="", help="Comma-separated sequence names to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N sequences")
    parser.add_argument("--raft-model", "--model", default=str(ROOT / "pretrained_models/metrics/raft-things.pth"))
    parser.add_argument("--metric", choices=["warping_error"], default="warping_error",
                        help=argparse.SUPPRESS)
    parser.add_argument("--metric-code-root", default=str(ROOT / "third_party/eval_metric-master"),
                        help="Extracted eval_metric-master directory for DOVER/VBench")
    parser.add_argument("--dover-model", default="", help="Optional DOVER.pth path")
    parser.add_argument("--seed", type=int, default=0, help="DOVER/VBench sampling seed")
    parser.add_argument("--fps", type=float, default=8, help="FPS for lossless frame-folder conversion only")
    parser.add_argument("--vbench-local", action="store_true", help="Use VBench's local checkpoint paths")
    return parser


def requested_metrics(text):
    aliases = {"ewarp": "ewarp_v3"}
    names = list(dict.fromkeys(aliases.get(m.strip().lower(), m.strip().lower())
                              for m in text.split(",") if m.strip()))
    unknown = set(names) - (SUPPORTED | {"ewarp_v3", "dover", "vbench"})
    if not names or unknown:
        raise ValueError(f"Unsupported or empty metric selection: {sorted(unknown)}")
    return names


def select_inputs(args, metrics):
    sequences = discover(args.pred)
    if args.sequences:
        names = [name.strip() for name in args.sequences.split(",") if name.strip()]
        missing = set(names) - set(sequences)
        if missing:
            raise ValueError(f"Prediction sequences not found: {sorted(missing)}")
        sequences = {name: path for name, path in sequences.items() if name in names}
        if not sequences:
            raise ValueError("--sequences must select at least one video")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        sequences = dict(list(sequences.items())[:args.limit])
    references = {}
    if set(metrics) & FULL_REFERENCE:
        if not args.gt:
            raise ValueError("Full-reference metrics require --gt")
        references = discover(args.gt)
        missing = set(sequences) - set(references)
        if missing:
            raise ValueError(f"Missing GT sequences: {sorted(missing)}")
        references = {name: references[name] for name in sequences}
    return sequences, references


def json_values(value):
    # Standard JSON has no Infinity token. Identical-image PSNR may be infinite.
    if isinstance(value, dict):
        return {k: json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    if isinstance(value, float) and math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def validate_results(results, names):
    for sequence, scores in results.items():
        if set(scores) != set(names):
            raise ValueError(f"Incomplete metrics for {sequence}: {list(scores)}")
        for metric, score in scores.items():
            if not math.isfinite(score) and not (metric == "psnr" and score == math.inf):
                raise ValueError(f"Invalid {metric} score for {sequence}: {score}")


def main(argv=None, default_metrics="psnr,ssim,clipiqa"):
    parser = build_parser(default_metrics)
    args = parser.parse_args(argv)
    names = requested_metrics(args.metrics)
    if args.crop < 0 or args.fps <= 0 or not math.isfinite(args.fps):
        parser.error("--crop must be nonnegative and --fps must be finite and positive")
    args.batch_size = args.batch_size if args.batch_size is not None else (8 if args.batch_mode else 1)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.is_center:
        args.alignment = "center"
    sequences, references = select_inputs(args, names)
    pred_root = Path(args.pred).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve() if args.out else (
        pred_root if pred_root.is_dir() else pred_root.parent)
    args.metric_code_root = str(Path(args.metric_code_root).expanduser().resolve())
    if args.dover_model:
        args.dover_model = str(Path(args.dover_model).expanduser().resolve())

    import torch
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        # The original resample2d wrapper uses .cuda(), so select the device too.
        torch.cuda.set_device(device)
    print(f"Device: {device}; metrics: {', '.join(names)}; sequences: {len(sequences)}", flush=True)
    results = {name: {} for name in sequences}
    metadata = {
        "torch_version": torch.__version__, "device": str(device),
        "requested_metrics": names, "aggregation": "unweighted_mean_of_video_scores",
        "prediction_paths": {k: str(v) for k, v in sequences.items()},
        "reference_paths": {k: str(v) for k, v in references.items()},
        "infinite_psnr_representation": "Infinity",
    }

    def merge(scores):
        if set(scores) != set(results):
            raise ValueError("Backend did not return every selected sequence")
        for name, values in scores.items():
            results[name].update(values)

    image_metrics = [name for name in names if name in SUPPORTED]
    if image_metrics:
        from .iqa import evaluate
        scores, metadata["iqa"] = evaluate(sequences, references, image_metrics, args, device)
        merge(scores)
    video_metrics = set(names) & {"ewarp_v3", "dover", "vbench"}
    if video_metrics:
        from .io import prepared_videos
        from .video import evaluate_dover, evaluate_vbench
        with prepared_videos(sequences, args.fps) as videos:
            if "ewarp_v3" in video_metrics:
                from .ewarp import evaluate
                scores, metadata["ewarp_v3"] = evaluate(videos, args, device)
                merge(scores)
            if "dover" in video_metrics:
                scores, metadata["dover"] = evaluate_dover(videos, args, device)
                merge(scores)
            if "vbench" in video_metrics:
                scores, metadata["vbench"] = evaluate_vbench(videos, args, device, output_dir)
                merge(scores)
        metadata["frame_folder_video_fps"] = args.fps
    validate_results(results, names)
    average = {name: sum(row[name] for row in results.values()) / len(results) for name in names}
    output = json_values({"per_sample": results, "average": average,
                          "count": len(results), "metadata": metadata})
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / ("metrics_" + "_".join(names) + ".json")
    # Publish only a complete, valid report.
    with tempfile.NamedTemporaryFile(mode="w", dir=output_dir, suffix=".json", delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(output, f, indent=2, allow_nan=False)
            f.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(target)
    print(f"Average: {output['average']}\nResults saved to: {target}", flush=True)
    return output
