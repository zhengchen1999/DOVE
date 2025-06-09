from pathlib import Path
import argparse
import logging

import torch
from torchvision import transforms
from torchvision.io import write_video
from tqdm import tqdm

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
)

from transformers import set_seed

import json
import os
import cv2
from PIL import Image

from pathlib import Path
import pyiqa
import imageio.v3 as iio
import glob

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logging.basicConfig(level=logging.INFO)

# 0 ~ 1
to_tensor = transforms.ToTensor()
video_exts = ['.mp4', '.avi', '.mov', '.mkv']
fr_metrics = ['psnr', 'ssim', 'lpips', 'dists']


def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper


def is_video_file(filename):
    return any(filename.lower().endswith(ext) for ext in video_exts)


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(to_tensor(Image.fromarray(rgb)))
    cap.release()
    return torch.stack(frames)


def read_image_folder(folder_path):
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    frames = [to_tensor(Image.open(p).convert("RGB")) for p in image_files]
    return torch.stack(frames)


def load_sequence(path):
    # return a tensor of shape [F, C, H, W] // 0, 1
    if os.path.isdir(path):
        return read_image_folder(path)
    elif os.path.isfile(path):
        if is_video_file(path):
            return read_video_frames(path)
        elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Treat image as a single-frame video
            img = to_tensor(Image.open(path).convert("RGB"))
            return img.unsqueeze(0)  # [1, C, H, W]
    raise ValueError(f"Unsupported input: {path}")

@no_grad
def compute_metrics(pred_frames, gt_frames, metrics_model, metric_accumulator, file_name):

    print(f"\n\n[{file_name}] Metrics:", end=" ")
    for name, model in metrics_model.items():
        scores = []
        for i in range(pred_frames.shape[0]):
            pred = pred_frames[i].unsqueeze(0)
            if gt_frames != None:
                gt = gt_frames[i].unsqueeze(0)
            if name in fr_metrics:
                score = model(pred, gt).item()
            else:
                score = model(pred).item()
            scores.append(score)
        val = sum(scores) / len(scores)
        metric_accumulator[name].append(val)
        print(f"{name.upper()}={val:.4f}", end="  ")
    print()


def save_frames_as_png(video, output_dir, fps=8):
    """
    Save video frames as PNG sequence.

    Args:
        video (torch.Tensor): shape [B, C, F, H, W], float in [0, 1]
        output_dir (str): directory to save PNG files
        fps (int): kept for API compatibility
    """
    video = video[0]  # Remove batch dimension
    video = video.permute(1, 2, 3, 0)  # [F, H, W, C]

    os.makedirs(output_dir, exist_ok=True)
    frames = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    
    for i, frame in enumerate(frames):
        filename = os.path.join(output_dir, f"{i:03d}.png")
        Image.fromarray(frame).save(filename)


def save_video_with_imageio_lossless(video, output_path, fps=8):
    """
    Save a video tensor to .mkv using imageio.v3.imwrite with ffmpeg backend.

    Args:
        video (torch.Tensor): shape [B, C, F, H, W], float in [0, 1]
        output_path (str): where to save the .mkv file
        fps (int): frames per second
    """
    video = video[0]
    video = video.permute(1, 2, 3, 0)

    frames = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec='libx264rgb',
        pixelformat='rgb24',
        macro_block_size=None,
        ffmpeg_params=['-crf', '0'],
    )


def save_video_with_imageio(video, output_path, fps=8):
    """
    Save a video tensor to .mp4 using imageio.v3.imwrite with ffmpeg backend.

    Args:
        video (torch.Tensor): shape [B, C, F, H, W], float in [0, 1]
        output_path (str): where to save the .mp4 file
        fps (int): frames per second
    """
    video = video[0]
    video = video.permute(1, 2, 3, 0)

    frames = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec='libx264',
        pixelformat='yuv420p',
        macro_block_size=None,
        ffmpeg_params=['-crf', '10'],
    )


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
    original_shape = (F, H, W, C)
    
    pad_f = 0
    pad_h = 0
    pad_w = 0

    if is_match:
        remainder = (F - 1) % 8
        if remainder != 0:
            last_frame = frames[-1:]
            pad_f = 8 - remainder
            repeated_frames = last_frame.repeat(pad_f, 1, 1, 1)
            frames = torch.cat([frames, repeated_frames], dim=0)

        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        if pad_h > 0 or pad_w > 0:
            # pad = (w_left, w_right, h_top, h_bottom)
            frames = torch.nn.functional.pad(frames, pad=(0, 0, 0, pad_w, 0, pad_h))  # pad right and bottom

    # to F, C, H, W
    return frames.float().permute(0, 3, 1, 2).contiguous(), pad_f, pad_h, pad_w, original_shape


def remove_padding_and_extra_frames(video, pad_F, pad_H, pad_W):
    if pad_F > 0:
        video = video[:, :, :-pad_F, :, :]
    if pad_H > 0:
        video = video[:, :, :, :-pad_H, :]
    if pad_W > 0:
        video = video[:, :, :, :, :-pad_W]
    
    return video


def make_temporal_chunks(F, chunk_len, overlap_t=8):
    """
    Args:
        F: total number of frames
        chunk_len: int, chunk length in time (excluding overlap)
        overlap: int, number of overlapping frames between chunks
    Returns:
        time_chunks: List of (start_t, end_t) tuples
    """
    if chunk_len == 0:
        return [(0, F)]

    effective_stride = chunk_len - overlap_t
    if effective_stride <= 0:
        raise ValueError("chunk_len must be greater than overlap")

    chunk_starts = list(range(0, F - overlap_t, effective_stride))
    if chunk_starts[-1] + chunk_len < F:
        chunk_starts.append(F - chunk_len)

    time_chunks = []
    for i, t_start in enumerate(chunk_starts):
        t_end = min(t_start + chunk_len, F)
        time_chunks.append((t_start, t_end))

    if len(time_chunks) >= 2 and time_chunks[-1][1] - time_chunks[-1][0] < chunk_len:
        last = time_chunks.pop()
        prev_start, _ = time_chunks[-1]
        time_chunks[-1] = (prev_start, last[1])

    return time_chunks


def make_spatial_tiles(H, W, tile_size_hw, overlap_hw=(32, 32)):
    """
    Args:
        H, W: height and width of the frame
        tile_size_hw: Tuple (tile_height, tile_width)
        overlap_hw: Tuple (overlap_height, overlap_width)
    Returns:
        spatial_tiles: List of (start_h, end_h, start_w, end_w) tuples
    """
    tile_height, tile_width = tile_size_hw
    overlap_h, overlap_w = overlap_hw

    if tile_height == 0 or tile_width == 0:
        return [(0, H, 0, W)]

    tile_stride_h = tile_height - overlap_h
    tile_stride_w = tile_width - overlap_w

    if tile_stride_h <= 0 or tile_stride_w <= 0:
        raise ValueError("Tile size must be greater than overlap")

    h_tiles = list(range(0, H - overlap_h, tile_stride_h))
    if not h_tiles or h_tiles[-1] + tile_height < H:
        h_tiles.append(H - tile_height)
    
     # Merge last row if needed
    if len(h_tiles) >= 2 and h_tiles[-1] + tile_height > H:
        h_tiles.pop()

    w_tiles = list(range(0, W - overlap_w, tile_stride_w))
    if not w_tiles or w_tiles[-1] + tile_width < W:
        w_tiles.append(W - tile_width)
    
    # Merge last column if needed
    if len(w_tiles) >= 2 and w_tiles[-1] + tile_width > W:
        w_tiles.pop()

    spatial_tiles = []
    for h_start in h_tiles:
        h_end = min(h_start + tile_height, H)
        if h_end + tile_stride_h > H:
            h_end = H
        for w_start in w_tiles:
            w_end = min(w_start + tile_width, W)
            if w_end + tile_stride_w > W:
                w_end = W
            spatial_tiles.append((h_start, h_end, w_start, w_end))
    return spatial_tiles


def get_valid_tile_region(t_start, t_end, h_start, h_end, w_start, w_end,
                          video_shape, overlap_t, overlap_h, overlap_w):
    _, _, F, H, W = video_shape

    t_len = t_end - t_start
    h_len = h_end - h_start
    w_len = w_end - w_start

    valid_t_start = 0 if t_start == 0 else overlap_t // 2
    valid_t_end = t_len if t_end == F else t_len - overlap_t // 2
    valid_h_start = 0 if h_start == 0 else overlap_h // 2
    valid_h_end = h_len if h_end == H else h_len - overlap_h // 2
    valid_w_start = 0 if w_start == 0 else overlap_w // 2
    valid_w_end = w_len if w_end == W else w_len - overlap_w // 2

    out_t_start = t_start + valid_t_start
    out_t_end = t_start + valid_t_end
    out_h_start = h_start + valid_h_start
    out_h_end = h_start + valid_h_end
    out_w_start = w_start + valid_w_start
    out_w_end = w_start + valid_w_end

    return {
        "valid_t_start": valid_t_start, "valid_t_end": valid_t_end,
        "valid_h_start": valid_h_start, "valid_h_end": valid_h_end,
        "valid_w_start": valid_w_start, "valid_w_end": valid_w_end,
        "out_t_start": out_t_start, "out_t_end": out_t_end,
        "out_h_start": out_h_start, "out_h_end": out_h_end,
        "out_w_start": out_w_start, "out_w_end": out_w_end,
    }


@no_grad
def process_video(
    pipe: CogVideoXPipeline,
    video: torch.Tensor,
    prompt: str = '',
    noise_step: int = 0,
    sr_noise_step: int = 399,
):
    """
    Parameters:
    - prompt (str): The description of the video to be generated.
    - output_path (str): The path where the generated video will be saved.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - fps (int): The frames per second for the generated video.
    """
    # SR the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.

    # Decode video
    video = video.to(pipe.vae.device, dtype=pipe.vae.dtype)
    latent_dist = pipe.vae.encode(video).latent_dist
    latent = latent_dist.sample() * pipe.vae.config.scaling_factor

    patch_size_t = pipe.transformer.config.patch_size_t
    if patch_size_t is not None:
        ncopy = latent.shape[2] % patch_size_t
        # Copy the first frame ncopy times to match patch_size_t
        first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
        latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)

        assert latent.shape[2] % patch_size_t == 0

    batch_size, num_channels, num_frames, height, width = latent.shape

    # Get prompt embeddings
    prompt_token_ids = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.transformer.config.max_text_seq_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_token_ids = prompt_token_ids.input_ids
    prompt_embedding = pipe.text_encoder(
        prompt_token_ids.to(latent.device)
    )[0]
    _, seq_len, _ = prompt_embedding.shape
    prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

    latent = latent.permute(0, 2, 1, 3, 4)

    # Add noise to latent (Select)
    if noise_step != 0:
        noise = torch.randn_like(latent)
        add_timesteps = torch.full(
            (batch_size,),
            fill_value=noise_step,
            dtype=torch.long,
            device=latent.device,
        )
        latent = pipe.scheduler.add_noise(latent, noise, add_timesteps)
    
    timesteps = torch.full(
        (batch_size,),
        fill_value=sr_noise_step,
        dtype=torch.long,
        device=latent.device,
    )

    # Prepare rotary embeds
    vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    rotary_emb = (
        pipe._prepare_rotary_positional_embeddings(
            height=height * vae_scale_factor_spatial,
            width=width * vae_scale_factor_spatial,
            num_frames=num_frames,
            device=latent.device,
        )
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )

    predicted_noise = pipe.transformer(
        hidden_states=latent,
        encoder_hidden_states=prompt_embedding,
        timestep=timesteps,
        image_rotary_emb=rotary_emb,
        return_dict=False,
    )[0]
    
    latent_generate = pipe.scheduler.get_velocity(
        predicted_noise, latent, timesteps
    )

    # generate video
    if patch_size_t is not None and ncopy > 0:
        latent_generate = latent_generate[:, ncopy:, :, :, :]

    # [B, C, F, H, W]
    video_generate = pipe.decode_latents(latent_generate)
    video_generate = (video_generate * 0.5 + 0.5).clamp(0.0, 1.0)
    
    return video_generate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSR using DOVE")

    parser.add_argument("--input_dir", type=str)

    parser.add_argument("--input_json", type=str, default=None)

    parser.add_argument("--gt_dir", type=str, default=None)

    parser.add_argument("--eval_metrics", type=str, default='') # 'psnr,ssim,lpips,dists,clipiqa,musiq,maniqa,niqe'

    parser.add_argument("--model_path", type=str)

    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")

    parser.add_argument("--output_path", type=str, default="./results", help="The path save generated video")

    parser.add_argument("--fps", type=int, default=16, help="The frames per second for the generated video")

    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")

    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    parser.add_argument("--upscale_mode", type=str, default="bilinear")

    parser.add_argument("--upscale", type=int, default=4)

    parser.add_argument("--noise_step", type=int, default=0)

    parser.add_argument("--sr_noise_step", type=int, default=399)

    parser.add_argument("--is_cpu_offload", action="store_true", help="Enable CPU offload for the model")

    parser.add_argument("--is_vae_st", action="store_true", help="Enable VAE slicing and tiling")

    parser.add_argument("--png_save", action="store_true", help="Save output as PNG sequence")

    # Crop and Tiling Parameters
    parser.add_argument("--tile_size_hw", type=int, nargs=2, default=(0, 0), help="Tile size for spatial tiling (height, width)")

    parser.add_argument("--overlap_hw", type=int, nargs=2, default=(32, 32))

    parser.add_argument("--chunk_len", type=int, default=0, help="Chunk length for temporal chunking")

    parser.add_argument("--overlap_t", type=int, default=8)

    args = parser.parse_args()

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError("Invalid dtype. Choose from 'float16', 'bfloat16', or 'float32'.")
    
    if args.chunk_len > 0:
        print(f"Chunking video into {args.chunk_len} frames with {args.overlap_t} overlap")
        overlap_t = args.overlap_t
    else:
        overlap_t = 0
    if args.tile_size_hw != (0, 0):
        print(f"Tiling video into {args.tile_size_hw} frames with {args.overlap_hw} overlap")
        overlap_hw = args.overlap_hw
    else:
        overlap_hw = (0, 0)
    
    # Set seed
    set_seed(args.seed)

    if args.input_json is None:
        with open(args.input_json, 'r') as f:
            video_prompt_dict = json.load(f)
    else:
        video_prompt_dict = {}
    
    # Get all video files from input directory
    video_files = []
    for ext in video_exts:
        video_files.extend(glob.glob(os.path.join(args.input_dir, f'*{ext}')))
    video_files = sorted(video_files)  # Sort files for consistent ordering

    if not video_files:
        raise ValueError(f"No video files found in {args.input_dir}")
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)

    # If you're using with lora, add this code
    if args.lora_path:
        print(f"Loading LoRA weights from {args.lora_path}")
        pipe.load_lora_weights(
            args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1"
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0) # lora_scale = lora_alpha / rank

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    if args.is_cpu_offload:
        # pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    
    if args.is_vae_st:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    
    # pipe.transformer.eval()
    # torch.set_grad_enabled(False)

    # 4. Set the metircs
    if args.eval_metrics != '':
        metrics_list = [m.strip().lower() for m in args.eval_metrics.split(',')]
        metrics_models = {}
        for name in metrics_list:
            try:
                metrics_models[name] = pyiqa.create_metric(name).to(pipe.device).eval()
            except Exception as e:
                print(f"Failed to initialize metric '{name}': {e}")
        metric_accumulator = {name: [] for name in metrics_list}
    else:
        metrics_models = None
        metric_accumulator = None
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = os.path.basename(video_path)
        prompt = video_prompt_dict.get(video_name, "")
        if os.path.exists(video_path):
            # Read video
            # [F, C, H, W]
            video, pad_f, pad_h, pad_w, original_shape = preprocess_video_match(video_path, is_match=True)
            H_, W_ = video.shape[2], video.shape[3]
            video = torch.nn.functional.interpolate(video, size=(H_*args.upscale, W_*args.upscale), mode=args.upscale_mode, align_corners=False)
            __frame_transform = transforms.Compose(
                [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)] # -1, 1
            )
            video = torch.stack([__frame_transform(f) for f in video], dim=0)
            video = video.unsqueeze(0)
            # [B, C, F, H, W]
            video = video.permute(0, 2, 1, 3, 4).contiguous()

            _B, _C, _F, _H, _W = video.shape
            time_chunks = make_temporal_chunks(_F, args.chunk_len, overlap_t)
            spatial_tiles = make_spatial_tiles(_H, _W, args.tile_size_hw, overlap_hw)

            output_video = torch.zeros_like(video)
            write_count = torch.zeros_like(video, dtype=torch.int)

            # print(f"Process video: {video_name} | Prompt: {prompt} | Frame: {_F} (ori: {original_shape[0]}; pad: {pad_f}) | Target Resolution: {_H}, {_W} (ori: {original_shape[1]*4}, {original_shape[2]*4}; pad: {pad_h}, {pad_w}) | Chunk Num: {len(time_chunks)*len(spatial_tiles)}")

            for t_start, t_end in time_chunks:
                for h_start, h_end, w_start, w_end in spatial_tiles:
                    video_chunk = video[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
                    # print(f"video_chunk: {video_chunk.shape} | t: {t_start}:{t_end} | h: {h_start}:{h_end} | w: {w_start}:{w_end}")

                    # [B, C, F, H, W]
                    _video_generate = process_video(
                        pipe=pipe,
                        video=video_chunk,
                        prompt=prompt,
                        noise_step=args.noise_step,
                        sr_noise_step=args.sr_noise_step,
                    )

                    region = get_valid_tile_region(
                        t_start, t_end, h_start, h_end, w_start, w_end,
                        video_shape=video.shape,
                        overlap_t=overlap_t,
                        overlap_h=overlap_hw[0],
                        overlap_w=overlap_hw[1],
                    )
                    output_video[:, :, region["out_t_start"]:region["out_t_end"],
                                    region["out_h_start"]:region["out_h_end"],
                                    region["out_w_start"]:region["out_w_end"]] = \
                    _video_generate[:, :, region["valid_t_start"]:region["valid_t_end"],
                                    region["valid_h_start"]:region["valid_h_end"],
                                    region["valid_w_start"]:region["valid_w_end"]]
                    write_count[:, :, region["out_t_start"]:region["out_t_end"],
                                    region["out_h_start"]:region["out_h_end"],
                                    region["out_w_start"]:region["out_w_end"]] += 1
            
            video_generate = output_video

            if (write_count == 0).any():
                print("Error: Lack of write in region !!!")
                exit()
            if (write_count > 1).any():
                print("Error: Write count > 1 in region !!!")
                exit()

            video_generate = remove_padding_and_extra_frames(video_generate, pad_f, pad_h*4, pad_w*4)
            file_name = os.path.basename(video_path)
            output_path = os.path.join(args.output_path, file_name)

            if metrics_models is not None:
                #  [1, C, F, H, W] -> [F, C, H, W]
                pred_frames = video_generate[0]
                pred_frames = pred_frames.permute(1, 0, 2, 3).contiguous()
                if args.gt_dir is not None:
                    gt_frames = load_sequence(os.path.join(args.gt_dir, file_name))
                else:
                    gt_frames = None
                compute_metrics(pred_frames, gt_frames, metrics_models, metric_accumulator, file_name)

            if args.png_save:
                # Save as PNG sequence
                output_dir = output_path.rsplit('.', 1)[0]  # Remove extension
                save_frames_as_png(video_generate, output_dir, fps=args.fps)
            else:
                output_path = output_path.replace('.mkv', '.mp4')
                save_video_with_imageio(video_generate, output_path, fps=args.fps)
        else:
            print(f"Warning: {video_name} not found in {args.input_dir}")

    if metrics_models is not None:
        print("\n=== Overall Average Metrics ===")
        count = len(next(iter(metric_accumulator.values())))
        overall_avg = {metric: 0 for metric in metrics_list}
        out_name = 'metrics_'
        for metric in metrics_list:
            out_name += f"{metric}_"
            scores = metric_accumulator[metric]
            if scores:
                avg = sum(scores) / len(scores)
                overall_avg[metric] = avg
                print(f"{metric.upper()}: {avg:.4f}")

        out_name = out_name.rstrip('_') + '.json'
        out_path = os.path.join(args.output_path, out_name)
        output = {
            "per_sample": metric_accumulator,
            "average": overall_avg,
            "count": count
        }
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)

    print("All videos processed.")