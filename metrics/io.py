"""Input discovery and lossless preparation shared by the metric backends."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import re
import tempfile

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", str(path))]


def image_paths(path):
    return sorted((p for p in path.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                  key=natural_key)


def discover(path):
    """Accept one file, a frame folder, or a root of videos/frame folders."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported input: {path}")
        return {path.stem: path}
    children = []
    for p in sorted(path.iterdir(), key=natural_key):
        if p.name.startswith("."):
            continue
        if ((p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
                or (p.is_dir() and image_paths(p))):
            children.append(p)
    if image_paths(path):
        if children:
            raise ValueError(f"Ambiguous root contains both frames and sequences: {path}")
        return {path.name: path}
    result = {}
    for p in children:
        name = p.name if p.is_dir() else p.stem
        if name in result:
            raise ValueError(f"Duplicate sequence name {name!r}: {result[name]}, {p}")
        result[name] = p
    if not result:
        raise ValueError(f"No videos or frame sequences found: {path}")
    return result


def read_frames(path):
    """Yield decoded RGB uint8 frames without changing their size."""
    import cv2
    import numpy as np
    from PIL import Image

    path = Path(path)
    if path.is_dir() or path.suffix.lower() in IMAGE_EXTENSIONS:
        files = image_paths(path) if path.is_dir() else [path]
        for image in files:
            with Image.open(image) as frame:
                yield np.array(frame.convert("RGB"))
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()


def load_sequence(path):
    import numpy as np
    import torch

    frames = list(read_frames(path))
    if not frames:
        raise ValueError(f"No readable frames: {path}")
    if len({f.shape for f in frames}) != 1:
        raise ValueError(f"Frames have different sizes: {path}")
    # Match torchvision.ToTensor + torch.stack in the supplied evaluator.
    # Channels-last strides select different CUDA convolution kernels and can
    # noticeably change LPIPS/DISTS even with identical pixel values.
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous().float() / 255.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def prepared_videos(sequences, fps=8):
    """Use existing videos directly; encode frame folders as lossless RGB."""
    with tempfile.TemporaryDirectory(prefix="dove-metrics-") as tmp:
        result = {}
        for name, path in sequences.items():
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                result[name] = path
                continue
            import imageio.v2 as imageio
            target = Path(tmp) / f"{name}.mp4"
            count = 0
            with imageio.get_writer(
                str(target), fps=fps, codec="libx264rgb", pixelformat="rgb24",
                macro_block_size=None, ffmpeg_params=["-crf", "0"],
            ) as writer:
                for frame in read_frames(path):
                    writer.append_data(frame)
                    count += 1
            if count < 2:
                raise ValueError(f"Video metrics need at least two frames: {path}")
            result[name] = target
        yield result
