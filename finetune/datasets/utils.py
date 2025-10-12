import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torchvision.transforms.functional import resize
from torchvision.io import write_video, read_video

import random
import math
import os
import numpy as np
from PIL import Image


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")


##########  loaders  ##########


def load_prompts(prompt_path: Path) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_videos(video_path: Path) -> List[Path]:
    with open(video_path, "r", encoding="utf-8") as file:
        return [
            video_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]

def load_videos_with_root(video_path: Path, root_path: Path) -> List[Path]:
    with open(video_path, "r", encoding="utf-8") as file:
        return [
            root_path / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]

def load_images(image_path: Path) -> List[Path]:
    with open(image_path, "r", encoding="utf-8") as file:
        return [
            image_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]


def load_images_from_videos(videos_path: List[Path]) -> List[Path]:
    first_frames_dir = videos_path[0].parent.parent / "first_frames"
    first_frames_dir.mkdir(exist_ok=True)

    first_frame_paths = []
    for video_path in videos_path:
        frame_path = first_frames_dir / f"{video_path.stem}.png"
        if frame_path.exists():
            first_frame_paths.append(frame_path)
            continue

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read video: {video_path}")

        # Save frame as PNG with same name as video
        cv2.imwrite(str(frame_path), frame)
        logging.info(f"Saved first frame to {frame_path}")

        # Release video capture
        cap.release()

        first_frame_paths.append(frame_path)

    return first_frame_paths


##########  preprocessors  ##########


def preprocess_image_with_resize(
    image_path: Path | str,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single image.

    Args:
        image_path: Path to the image file.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        torch.Tensor: Image tensor with shape [C, H, W] where:
            C = number of channels (3 for RGB)
            H = height
            W = width
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)
    image = cv2.imread(image_path.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1).contiguous()
    return image


def preprocess_video_with_resize(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count > max_num_frames, downsample frames evenly
      2. If video dimensions don't match (height, width), resize frames

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix(), width=width, height=height)
    video_num_frames = len(video_reader)
    if video_num_frames < max_num_frames:
        # Get all frames first
        frames = video_reader.get_batch(list(range(video_num_frames)))
        # Repeat the last frame until we reach max_num_frames
        last_frame = frames[-1:]
        num_repeats = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)
        return frames.float().permute(0, 3, 1, 2).contiguous()
    else:
        indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
        frames = video_reader.get_batch(indices)
        frames = frames[:max_num_frames].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        return frames


def preprocess_video_with_buckets(
    video_path: Path,
    resolution_buckets: List[Tuple[int, int, int]],
) -> torch.Tensor:
    """
    Args:
        video_path: Path to the video file.
        resolution_buckets: List of tuples (num_frames, height, width) representing
            available resolution buckets.

    Returns:
        torch.Tensor: Video tensor with shape [F, C, H, W] where:
            F = number of frames
            C = number of channels (3 for RGB)
            H = height
            W = width

    The function processes the video through these steps:
        1. Finds nearest frame bucket <= video frame count
        2. Downsamples frames evenly to match bucket size
        3. Finds nearest resolution bucket based on dimensions
        4. Resizes frames to match bucket resolution
    """
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    video_num_frames = len(video_reader)
    resolution_buckets = [bucket for bucket in resolution_buckets if bucket[0] <= video_num_frames]
    if len(resolution_buckets) == 0:
        raise ValueError(
            f"video frame count in {video_path} is less than all frame buckets {resolution_buckets}"
        )

    nearest_frame_bucket = min(
        resolution_buckets,
        key=lambda bucket: video_num_frames - bucket[0],
        default=1,
    )[0]
    frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))
    frames = video_reader.get_batch(frame_indices)
    frames = frames[:nearest_frame_bucket].float()
    frames = frames.permute(0, 3, 1, 2).contiguous() # [F, C, H, W]

    nearest_res = min(
        resolution_buckets, key=lambda x: abs(x[1] - frames.shape[2]) + abs(x[2] - frames.shape[3])
    )
    nearest_res = (nearest_res[1], nearest_res[2])
    frames = torch.stack([resize(f, nearest_res) for f in frames], dim=0)

    return frames

def preprocess_video_match(
    video_path: Path | str,
    is_match: bool = False,
) -> torch.Tensor:
    """
    Loads a single video.

    Args:
        video_path: Path to the video file.
    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    video_num_frames = len(video_reader)
    frames = video_reader.get_batch(list(range(video_num_frames)))
    F, H, W, C = frames.shape
    if is_match:
        remainder = (F - 1) % 8
        if remainder != 0:
            last_frame = frames[-1:]
            num_repeats = 8 - remainder
            repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
            frames = torch.cat([frames, repeated_frames], dim=0)
        
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        if pad_h > 0 or pad_w > 0:
            # pad = (w_left, w_right, h_top, h_bottom)
            frames = torch.nn.functional.pad(frames, pad=(0, 0, 0, pad_w, 0, pad_h))  # pad right and bottom

    # to F, C, H, W // 0 ~ 255
    return frames.float().permute(0, 3, 1, 2).contiguous()

def random_crop(video_path, frame_size=25, height=256, width=256):
    """
    Returns:
        torch.Tensor: Video tensor with shape [F, C, H, W] where:
            F = number of frames
            C = number of channels (3 for RGB)
            H = height
            W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    # [F, H, W, C]
    frames = video_reader.get_batch(list(range(len(video_reader))))
    F, H, W, C = frames.shape
    if F < frame_size:
        logging.warning(
            f"Video {video_path} has only {F} frames, less than the required {frame_size} frames."
        )
        last_frame = frames[-1:]  # shape: [1, H, W, C]
        num_repeats = frame_size - F
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)
        F = frame_size  # update F
    beg_frame = random.randint(0, F - frame_size)
    top = random.randint(0, H - height)
    left = random.randint(0, W - width)
    # crop size: [frame_size, height, width, 3]
    cropped = frames[beg_frame:beg_frame + frame_size, top:top + height, left:left + width, :]
    return cropped.float().permute(0, 3, 1, 2).contiguous() # [F, C, H, W]

def resize_random_crop(video_path, frame_size=25, height=256, width=256):
    """
    Returns:
        torch.Tensor: Video tensor with shape [F, C, H, W] where:
            F = number of frames
            C = number of channels (3 for RGB)
            H = height
            W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    frames = video_reader.get_batch(list(range(len(video_reader))))

    F_total, H_orig, W_orig, C = frames.shape
    if F_total < frame_size:
        logging.warning(
            f"Video {video_path} has only {F_total} frames, less than the required {frame_size} frames."
        )
        last_frame = frames[-1:]  # shape: [1, H, W, C]
        num_repeats = frame_size - F_total
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)
        F_total = frame_size  # update F

    # Compute scale to maintain aspect ratio while covering target crop size
    scale = max(height / H_orig, width / W_orig)
    new_H = math.ceil(H_orig * scale)
    new_W = math.ceil(W_orig * scale)

    # Convert to float and permute to [F, C, H, W]
    frames = frames.permute(0, 3, 1, 2).float()  # [F, C, H, W]

    # Resize using torch's interpolate
    frames_resized = torch.nn.functional.interpolate(frames, size=(new_H, new_W), mode='bilinear', align_corners=False)

    # Random temporal and spatial crop
    beg_frame = random.randint(0, F_total - frame_size)
    top = random.randint(0, new_H - height)
    left = random.randint(0, new_W - width)
    cropped = frames_resized[beg_frame:beg_frame + frame_size, :, top:top + height, left:left + width]  # [frame_size, C, H, W]
    return cropped.contiguous()

def crop_padded_video(artifact_value, ref_video):
    F, _, target_h, target_w = ref_video.shape

    artifact_value = artifact_value[:F]

    cropped_imgs = []
    for img in artifact_value:
        cropped = img.crop((0, 0, target_w, target_h))  # (left, upper, right, lower)
        cropped_imgs.append(cropped)

    return cropped_imgs

def save_and_reload_video(video_tensor, temp_path, fps=8):
    # F, C, H, W // 0, 1
    video_tensor = video_tensor.permute(0, 2, 3, 1)
    video_tensor = (video_tensor * 255).clamp(0, 255).to(torch.uint8).contiguous()

    write_video(temp_path, video_tensor, fps=fps, options={'crf': '5'})

    video_out, _, _ = read_video(temp_path, pts_unit='sec')
    # [F, H, W, C]
    reloaded_tensor = video_out.to(torch.float32) / 255.0

    # [F, C, H, W] // 0, 1
    reloaded_tensor = reloaded_tensor.permute(0, 3, 1, 2).contiguous()

    os.remove(temp_path)

    return reloaded_tensor

def read_video_frames(video_path, frame_size):
    """
    Read video and pad frames if necessary.
    Returns:
        list of np.ndarray: Each frame is [H, W, C], dtype=uint8
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(video_path.as_posix())
    
    # Get all frames as numpy array [F, H, W, C]
    frames = video_reader.get_batch(list(range(len(video_reader)))).numpy()
    F, H, W, C = frames.shape

    if F < frame_size:
        logging.warning(
            f"Video {video_path} has only {F} frames, less than required {frame_size}. Padding with last frame."
        )
        last_frame = frames[-1:]  # shape: [1, H, W, C]
        repeated_frames = np.repeat(last_frame, frame_size - F, axis=0)
        frames = np.concatenate([frames, repeated_frames], axis=0)

    return [frame for frame in frames]  # list of [H, W, C]

def read_video_or_image(input_path, frame_size):
    """
    Read frames from a video file or a single image.

    Args:
        input_path (str or Path): Path to a video or image file.
        frame_size (int, optional): Desired number of frames. If fewer, pad with the last frame.

    Returns:
        list of np.ndarray: Each frame is [H, W, C], dtype=uint8
    """
    if isinstance(input_path, str):
        input_path = Path(input_path)
    
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Read from video
        video_reader = decord.VideoReader(str(input_path))
        frames = video_reader.get_batch(list(range(len(video_reader)))).numpy()
        frame_list = [frame for frame in frames]
    elif input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        # Read from single image
        img = Image.open(input_path).convert("RGB")
        frame_np = np.array(img, dtype=np.uint8)
        frame_list = [frame_np]
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    return frame_list  # [F, H, W, C] uint8

def random_crop_frames(frames, frame_size=25, height=256, width=256):
    """
    Randomly crop video frames.

    Args:
        frames (list of np.ndarray): List of frames, each of shape [H, W, C], dtype=uint8
        frame_size (int): Number of frames to crop in temporal dimension
        height (int): Crop height
        width (int): Crop width

    Returns:
        list of np.ndarray: Cropped frames of shape [H, W, C], dtype=uint8
    """
    F = len(frames)
    H, W, C = frames[0].shape
    beg_frame = random.randint(0, F - frame_size)
    top = random.randint(0, H - height) if H > height else 0
    left = random.randint(0, W - width) if W > width else 0
    _h = min(height, H)
    _w = min(width, W)

    if _h % 4 != 0:
        _h = _h - (_h % 4)
    if _w % 4 != 0:
        _w = _w - (_w % 4)

    cropped = [
        frame[top:top + _h, left:left + _w, :]
        for frame in frames[beg_frame:beg_frame + frame_size]
    ]
    return cropped  # list of [H, W, C] uint8

def paired_random_crop_video(
    hq_frames: List[np.ndarray],
    lq_frames: List[np.ndarray],
    num_frames: int,
    lq_crop_h: int,
    lq_crop_w: int,
    scale: int,
    file_path: str = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Paired random crop for video frames. Ensures alignment between HQ and LQ frames.

    Args:
        hq_frames (list[np.ndarray]): HQ frames, each of shape [H_gt, W_gt, C], value in [0, 255]
        lq_frames (list[np.ndarray]): LQ frames, each of shape [H_lq, W_lq, C]
        num_frames (int): Number of frames to crop in the temporal dimension.
        lq_crop_h (int): Spatial crop height on LQ.
        lq_crop_w (int): Spatial crop width on LQ.
        scale (int): Scale factor between LQ and HQ

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Cropped HQ and LQ frames (aligned), each a list of [H, W, C]
    """
    assert len(hq_frames) == len(lq_frames), "HQ and LQ must have same number of frames"
    assert len(hq_frames) >= num_frames, "Not enough frames for temporal crop"

    h_lq, w_lq, _ = lq_frames[0].shape
    h_hq, w_hq, _ = hq_frames[0].shape

    # Check spatial scale match
    assert h_hq == h_lq * scale and w_hq == w_lq * scale, \
        f"File [{file_path}]: Spatial size mismatch: HQ ({h_hq}, {w_hq}) vs LQ ({h_lq}, {w_lq}) with scale {scale}"

    # Check spatial crop size
    assert h_lq >= lq_crop_h and w_lq >= lq_crop_w, f"File [{file_path}]: LQ crop size too large"

    # Randomly sample spatial crop location on LQ
    top = random.randint(0, h_lq - lq_crop_h)
    left = random.randint(0, w_lq - lq_crop_w)
    top_hq, left_hq = top * scale, left * scale
    hq_crop_h, hq_crop_w = lq_crop_h * scale, lq_crop_w * scale

    # Randomly sample temporal index
    start_idx = random.randint(0, len(hq_frames) - num_frames)

    # Crop frames
    cropped_hq = [
        f[top_hq:top_hq + hq_crop_h, left_hq:left_hq + hq_crop_w, :]
        for f in hq_frames[start_idx:start_idx + num_frames]
    ]
    cropped_lq = [
        f[top:top + lq_crop_h, left:left + lq_crop_w, :]
        for f in lq_frames[start_idx:start_idx + num_frames]
    ]

    return cropped_hq, cropped_lq