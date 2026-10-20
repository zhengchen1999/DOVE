"""Adapter to the supplied normalization-only eWarp v3 implementation."""
from argparse import Namespace
from pathlib import Path
import math

from .io import read_frames, sha256


def evaluate(sequences, args, device):
    import torch
    if device.type != "cuda":
        raise ValueError("eWarp v3 requires CUDA and the supplied resample2d extension")
    checkpoint = Path(args.raft_model).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"RAFT checkpoint missing: {checkpoint}. Extract raft-things.pth "
            "from ewarp_v3_code(2).zip and pass --raft-model."
        )
    try:
        from .raft.ewarp_v3 import compute_video_warping_error, metric_metadata, RAFT
    except ImportError as error:
        raise RuntimeError(
            "Install the eWarp dependencies and build "
            "metrics/raft/networks/resample2d_package; see metrics/README.md."
        ) from error
    raft_args = Namespace(small=False, mixed_precision=False, alternate_corr=False,
                          model=str(checkpoint))
    model = torch.nn.DataParallel(RAFT(raft_args))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model = model.module.to(device).eval()
    results = {}
    for name, path in sequences.items():
        frames = read_frames(path)
        try:
            first, second = next(frames, None), next(frames, None)
            if first is None or second is None:
                raise ValueError(f"At least two readable frames are required: {path}")
            if first.shape != second.shape or min(first.shape[:2]) < 256:
                raise ValueError(f"eWarp needs equal-sized frames at least 256x256: {path}")
        finally:
            frames.close()
        score = compute_video_warping_error(str(path), model, device).item() * 1000
        if not math.isfinite(score):
            raise ValueError(f"Non-finite eWarp v3 score: {path}")
        results[name] = {"ewarp_v3": score}
        print(f"{name}: ewarp_v3={score:.8f}", flush=True)
    metadata = metric_metadata(raft_args)
    metadata["flow_model_sha256"] = sha256(checkpoint)
    return results, metadata
