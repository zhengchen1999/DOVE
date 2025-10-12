import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override

from finetune.constants import LOG_LEVEL, LOG_NAME
from finetune.datasets.degradation import RandomBlur, RandomResize, RandomNoise, RandomJPEGCompression, RandomVideoCompression, DegradationsWithShuffle
from finetune.datasets.utils import read_video_frames, random_crop_frames, paired_random_crop_video

from .utils import (
    load_prompts,
    load_videos,
    load_videos_with_root,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
    random_crop,
    resize_random_crop,
)

import yaml
import math
import numpy as np
import random

if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)

class RealSRDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_num_frames: int,
        height: int,
        width: int,
        video_column: str = None,
        caption_column: str = None,
        device: torch.device = None,
        trainer: "Trainer" = None,
        prompt_cache: str = "prompt_embeddings",
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        # video/prompt 顺序对应
        self.videos = load_videos_with_root(video_column, data_root)
        if caption_column is None:
            self.prompts = [''] * len(self.videos)
        else:
            self.prompts = load_prompts(caption_column)
        self.prompt_cache = prompt_cache
        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        self.trainer = trainer
        self.is_cache = trainer.args.is_cache
        self.is_latent = trainer.args.is_latent
        self.crop_mode = trainer.args.crop_mode
        empty_prompt_dir = self.trainer.args.data_root / "cache" / self.prompt_cache / "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"
        if trainer.args.empty_prompt and empty_prompt_dir.exists():
            self.empty_prompt = load_file(empty_prompt_dir)["prompt_embedding"]
        else:
            self.empty_prompt = None
        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transform = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)] # -1, 1
        )

        with open(trainer.args.degradation_config, 'r') as f:
            self.opt = yaml.safe_load(f)
        
        # Initialize degradation operations
        self.init_degradtion(self.opt)
        if 'youhq' in str(trainer.args.video_column).lower():
            self.inter_frames = min(self.max_num_frames + 10, 30)
        else:
            self.inter_frames = self.max_num_frames + 10 
        self.inter_height = math.ceil((self.height * 1.5) / 16) * 16
        self.inter_width = math.ceil((self.width * 1.5) / 16) * 16
        # self.inter_target_h = int(self.inter_height / 4)
        # self.inter_target_w = int(self.inter_width / 4)
        self.target_h = int(self.height / 4)
        self.target_w = int(self.width / 4)

        # Check if all video files exist
        if any(not path.is_file() for path in self.videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.videos if not path.is_file())}"
            )

        # Check if number of prompts matches number of videos
        if len(self.videos) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.videos)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )
        
    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        prompt = self.prompts[index]
        is_empty_propmt = random.random() < self.trainer.args.empty_ratio
        if is_empty_propmt:
            prompt = ''

        video = self.videos[index]
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        # Current shape of frames: [F, C, H, W]
        hq_frames, lq_frames = self.preprocess(video)
        H_, W_ = hq_frames.shape[2], hq_frames.shape[3]
        lq_frames_resize = F.interpolate(lq_frames, size=(H_, W_), mode="bilinear", align_corners=False)

        # Convert to [B, C, F, H, W]
        hq_frames = self.video_transform(hq_frames)
        hq_frames = hq_frames.unsqueeze(0)
        hq_frames = hq_frames.permute(0, 2, 1, 3, 4).contiguous()
        # Convert to [B, C, F, H, W]
        lq_frames_resize = self.video_transform(lq_frames_resize)
        lq_frames_resize = lq_frames_resize.unsqueeze(0)
        lq_frames_resize = lq_frames_resize.permute(0, 2, 1, 3, 4).contiguous()

        cache_dir = self.trainer.args.data_root / "cache"

        prompt_embeddings_dir = cache_dir / self.prompt_cache
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)
        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")

        if self.empty_prompt is not None:
            # print(f"Using empty prompt embedding")
            prompt_embedding = self.empty_prompt
        elif prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,
            )
        else:
            # 不能多进程处理
            prompt_embedding = self.encode_text(prompt)[0].to("cpu")
            if self.is_cache:
                save_file({"prompt_embedding": prompt_embedding.to("cpu")}, prompt_embedding_path)
                logger.info(
                    f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False
                )

        encoded_hq_video = None
        encoded_lq_video = None

        if self.is_latent:
            hq_video_latent_dir = (
                cache_dir / "video_latent" / "hq"  / self.trainer.args.model_name / train_resolution_str
            )
            lq_video_latent_dir = (
                cache_dir / "video_latent" / "lq" / self.trainer.args.model_name / train_resolution_str
            )

            hq_video_latent_dir.mkdir(parents=True, exist_ok=True)
            lq_video_latent_dir.mkdir(parents=True, exist_ok=True)
            encoded_hq_video_path = hq_video_latent_dir / (video.stem + ".safetensors")
            encoded_lq_video_path = lq_video_latent_dir / (video.stem + ".safetensors")

            if encoded_hq_video_path.exists():
                # encoded_video = torch.load(encoded_video_path, weights_only=True)
                encoded_hq_video = load_file(encoded_hq_video_path)["encoded_hq_video"]
                logger.debug(f"Loaded encoded hq video from {encoded_hq_video_path}", main_process_only=False)
            else:
                # 不能多进程处理
                encoded_hq_video = self.encode_video(hq_frames).to("cpu")
                if self.is_cache:
                    save_file({"encoded_hq_video": encoded_hq_video.to("cpu")}, encoded_hq_video_path)
                    logger.info(f"Saved encoded video to {encoded_hq_video_path}", main_process_only=False)

            if encoded_lq_video_path.exists():
                # encoded_video = torch.load(encoded_video_path, weights_only=True)
                encoded_lq_video = load_file(encoded_lq_video_path)["encoded_hq_video"]
                logger.debug(f"Loaded encoded lq video from {encoded_lq_video_path}", main_process_only=False)
            else:
                # 不能多进程处理
                encoded_lq_video = self.encode_video(lq_frames_resize).to("cpu")

                if self.is_cache:
                    save_file({"encoded_hq_video": encoded_lq_video.to("cpu")}, encoded_lq_video_path)
                    logger.info(f"Saved encoded video to {encoded_lq_video_path}", main_process_only=False)
        
        # shape of encoded_video: [C, F, H, W]
        # shape of video_frames: [B, C, F, H, W]
        return {
            "prompt": prompt,
            "hq_video": hq_frames[0],
            "lq_video": lq_frames_resize[0],
            "prompt_embedding": prompt_embedding,
            "encoded_hq_video": encoded_hq_video,
            "encoded_lq_video": encoded_lq_video,
            "video_metadata": {
                "num_frames": hq_frames.shape[2],
                "height": hq_frames.shape[3],
                "width": hq_frames.shape[4],
            },
            "encoded_video_metadata": (
                {
                    "num_frames": encoded_hq_video.shape[1],
                    "height": encoded_hq_video.shape[2],
                    "width": encoded_hq_video.shape[3],
                } if encoded_hq_video is not None else None
            ),
        }

    def preprocess(self, video_path: Path) -> torch.Tensor:
        """
        Loads and preprocesses a video.

        Args:
            video_path: Path to the video file to load.

        Returns:
            torch.Tensor: Video tensor of shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width
        """
        if self.crop_mode == 'random_crop':
            frame_list = read_video_frames(video_path, self.inter_frames)
            crop_frame_list = random_crop_frames(frame_list, self.inter_frames, self.inter_height, self.inter_width)
            inter_H, interW, _ = crop_frame_list[0].shape
            inter_target_h = int(inter_H / 4)
            inter_target_w = int(interW / 4)

            if isinstance(self.degradation_with_shuffle.degradations[0],list):
                self.degradation_with_shuffle.degradations[0][0].params['target_size'] = (inter_target_h, inter_target_w)
            else:
                self.degradation_with_shuffle.degradations[1][0].params['target_size'] = (inter_target_h, inter_target_w)
            
            input_dict = dict(lqs=crop_frame_list)
            deg_frame_list = self.degrade(input_dict)['lqs']

            hq_frame_list, lq_frame_list = paired_random_crop_video(crop_frame_list, deg_frame_list, self.max_num_frames, self.target_h, self.target_w, 4, video_path)

            hq_tensor_list = [self.to_tensor(f) for f in hq_frame_list]
            lq_tensor_list = [self.to_tensor(f) for f in lq_frame_list]
            
            # [F, C, H, W]
            hq_video_tensor = torch.stack(hq_tensor_list, dim=0)
            lq_video_tensor = torch.stack(lq_tensor_list, dim=0)

            return hq_video_tensor, lq_video_tensor
        else:
            # TODO
            raise NotImplementedError(f"Crop mode {self.crop_mode} not implemented")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor with the same shape as the input
        """
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)
    
    def init_degradtion(self, opt) -> None:
        # Initialize first degradation operations
        self.random_blur_1 = RandomBlur(
            params=opt['degradation_1']['random_blur']['params'],
            keys=opt['degradation_1']['random_blur']['keys']
        )
        self.random_resize_1 = RandomResize(
            params=opt['degradation_1']['random_resize']['params'],
            keys=opt['degradation_1']['random_resize']['keys']
        )
        self.random_noise_1 = RandomNoise(
            params=opt['degradation_1']['random_noise']['params'],
            keys=opt['degradation_1']['random_noise']['keys']
        )
        self.random_jpeg_1 = RandomJPEGCompression(
            params=opt['degradation_1']['random_jpeg']['params'],
            keys=opt['degradation_1']['random_jpeg']['keys']
        )
        self.random_mpeg_1 = RandomVideoCompression(
            params=opt['degradation_1']['random_mpeg']['params'],
            keys=opt['degradation_1']['random_mpeg']['keys']
        )
        
        # Initialize second degradation operations
        self.random_blur_2 = RandomBlur(
            params=opt['degradation_2']['random_blur']['params'],
            keys=opt['degradation_2']['random_blur']['keys']
        )
        self.random_resize_2 = RandomResize(
            params=opt['degradation_2']['random_resize']['params'],
            keys=opt['degradation_2']['random_resize']['keys']
        )
        self.random_noise_2 = RandomNoise(
            params=opt['degradation_2']['random_noise']['params'],
            keys=opt['degradation_2']['random_noise']['keys']
        )
        self.random_jpeg_2 = RandomJPEGCompression(
            params=opt['degradation_2']['random_jpeg']['params'],
            keys=opt['degradation_2']['random_jpeg']['keys']
        )
        self.degradation_with_shuffle = DegradationsWithShuffle(
            degradations=opt['degradation_2']['degradation_with_shuffle']['degradations'],
            keys=opt['degradation_2']['degradation_with_shuffle']['keys']
        )
        
        # Define degradation sequence
        self.first_stage = [
            self.random_blur_1,
            self.random_resize_1, 
            self.random_noise_1,
            self.random_jpeg_1,
            self.random_mpeg_1
        ]
        
        self.second_stage = [
            self.random_blur_2,
            self.random_resize_2,
            self.random_noise_2,
            self.random_jpeg_2,
            self.degradation_with_shuffle
        ]

    def degrade(self, data):
        """
        Apply degradation pipeline to input data
        
        Args:
            data: dict containing frames to be processed (e.g., {'lqs': frame_list})
        
        Returns:
            dict: processed data with degraded frames
        """
        # Apply first stage degradations
        for degradation in self.first_stage:
            data = degradation(data)
        
        # Apply second stage degradations
        for degradation in self.second_stage:
            data = degradation(data)
            
        return data
    
    def to_tensor(self, frame):
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame).float()
        else:
            frame = frame.float()
        if frame.ndim == 3 and frame.shape[-1] == 3:  # [H, W, C]
            frame = frame.permute(2, 0, 1).contiguous()  # [C, H, W]
        frame = torch.clamp(frame, 0.0, 255.0)
        return frame
