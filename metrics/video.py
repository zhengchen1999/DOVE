"""DOVER and the six-dimension VBench protocol from the supplied code bundle."""
from contextlib import contextmanager
from pathlib import Path
import math
import os
import random
import sys
import tempfile

from .io import sha256

VBENCH_RANGES = {
    "subject_consistency": (0.1462, 1.0, 1.0),
    "background_consistency": (0.2615, 1.0, 1.0),
    "motion_smoothness": (0.706, 0.9975, 1.0),
    "dynamic_degree": (0.0, 1.0, 0.5),
    "aesthetic_quality": (0.0, 1.0, 1.0),
    "imaging_quality": (0.0, 100.0, 1.0),
}


@contextmanager
def code_directory(root, name):
    path = Path(root).expanduser().resolve() / name
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {path}; extract eval_metric.zip and set --metric-code-root")
    old_cwd, old_path = Path.cwd(), sys.path[:]
    try:
        os.chdir(path)
        sys.path.insert(0, str(path))
        yield path
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dover_fusion(technical, aesthetic):
    t = (technical - 0.1107) / 0.07355
    a = (aesthetic + 0.08285) / 0.03774
    value = t * 0.6104 + a * 0.3896
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def evaluate_dover(sequences, args, device):
    import torch
    import yaml
    with code_directory(args.metric_code_root, "DOVER") as root:
        import dover
        from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
        from dover.models import DOVER
        if not Path(dover.__file__).resolve().is_relative_to(root):
            raise RuntimeError("DOVER was imported from a different installation")
        config = root / "dover.yml"
        opt = yaml.safe_load(config.read_text())
        weight = Path(args.dover_model) if args.dover_model else root / opt["test_load_path"]
        if not weight.is_file():
            raise FileNotFoundError(f"DOVER weight missing: {weight}; set --dover-model")
        evaluator = DOVER(**opt["model"]["args"]).to(device)
        evaluator.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
        sample_types = opt["data"]["val-l1080p"]["args"]["sample_types"]
        samplers = {}
        for key, value in sample_types.items():
            if "t_frag" in value:
                samplers[key] = UnifiedFrameSampler(
                    value["clip_len"] // value["t_frag"], value["t_frag"],
                    value["frame_interval"], value["num_clips"])
            else:
                samplers[key] = UnifiedFrameSampler(
                    value["clip_len"], value["num_clips"], value["frame_interval"])
        mean = torch.FloatTensor([123.675, 116.28, 103.53])
        std = torch.FloatTensor([58.395, 57.12, 57.375])
        results, raw = {}, {}
        for name, path in sequences.items():
            seed_all(args.seed)
            views, _ = spatial_temporal_view_decomposition(str(path), sample_types, samplers)
            for key, value in views.items():
                clips = sample_types[key].get("num_clips", 1)
                views[key] = (((value.permute(1, 2, 3, 0) - mean) / std)
                              .permute(3, 0, 1, 2)
                              .reshape(value.shape[0], clips, -1, *value.shape[2:])
                              .transpose(0, 1).to(device))
            with torch.no_grad():
                scores = evaluator(views, reduce_scores=False)
            raw[name] = {key: value.mean().item() for key, value in zip(views, scores)}
            results[name] = {"dover": dover_fusion(raw[name]["technical"], raw[name]["aesthetic"])}
            print(f"{name}: {results[name]}", flush=True)
        return results, {
            "protocol": "supplied_DOVER_dover.yml_val-l1080p", "score_range": [0, 1],
            "seed_per_video": args.seed, "raw_scores": raw,
            "checkpoint_sha256": sha256(weight), "config_sha256": sha256(config),
        }


def vbench_fusion(dimensions):
    if set(dimensions) != set(VBENCH_RANGES):
        raise ValueError(f"VBench requires all six dimensions; received {list(dimensions)}")
    if not all(math.isfinite(float(x)) for x in dimensions.values()):
        raise ValueError("VBench returned non-finite dimension scores")
    return sum((float(dimensions[name]) - low) / (high - low) * weight
               for name, (low, high, weight) in VBENCH_RANGES.items())


def evaluate_vbench(sequences, args, device, output_dir):
    if device.type != "cuda":
        raise ValueError("The supplied VBench implementation requires CUDA")
    with code_directory(args.metric_code_root, "VBench") as root:
        import vbench
        from vbench import VBench
        if not Path(vbench.__file__).resolve().is_relative_to(root):
            raise RuntimeError("VBench was imported from a different installation")
        seed_all(args.seed)
        evaluator = VBench(device, str(root / "vbench/VBench_full_info.json"),
                           str(output_dir / "vbench_details"))
        with tempfile.TemporaryDirectory(prefix="dove-vbench-") as tmp:
            # The supplied VBench discovers MP4 names only. Symlinks preserve bytes.
            for name, path in sequences.items():
                (Path(tmp) / f"{name}.mp4").symlink_to(path)
            dimensions = evaluator.evaluate(
                videos_path=tmp, name="dove", dimension_list=list(VBENCH_RANGES),
                local=args.vbench_local, read_frame=False, mode="custom_input",
                imaging_quality_preprocessing_mode="longer",
            )
        if set(dimensions) != set(sequences):
            raise ValueError("VBench result names do not match the requested videos")
        results = {name: {"vbench": vbench_fusion(values)} for name, values in dimensions.items()}
        return results, {
            "protocol": "supplied_six_dimension_weighted_sum",
            "dimensions": dimensions, "ranges_and_weights": VBENCH_RANGES,
            "imaging_quality_preprocessing": "longer", "seed": args.seed,
            "spatial_downsampling": False,
        }
