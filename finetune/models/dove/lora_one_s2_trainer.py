from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model
from finetune.utils.metric_utils import EdgeDetectionModel

from ..utils import register
from functools import partial
from torchvision import transforms
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput

import pyiqa
import random

class DOVES2Trainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "vae"]

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer"
        )

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXPipeline:
        pipe = CogVideoXPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device)
        )[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_hq_videos": [], "encoded_lq_videos": [], "hq_videos": [], "lq_videos": [], "hq_images": [], "lq_images": [], "prompts": [], "prompt_embeddings": [], "video_metadatas": [], "encoded_video_metadatas": []}

        for sample in samples:
            ret["hq_videos"].append(sample["hq_video"])
            ret["lq_videos"].append(sample["lq_video"])
            ret["hq_images"].append(sample["hq_image"])
            ret["lq_images"].append(sample["lq_image"])
            ret["prompts"].append(sample["prompt"])
            ret["prompt_embeddings"].append(sample["prompt_embedding"])
            ret["video_metadatas"].append(sample["video_metadata"])
            if sample["encoded_hq_video"] != None:
                ret["encoded_hq_videos"].append(sample["encoded_hq_video"])
            if sample["encoded_lq_video"] != None:
                ret["encoded_lq_videos"].append(sample["encoded_lq_video"])
            if sample["encoded_video_metadata"] != None:
                ret["encoded_video_metadatas"].append(sample["encoded_video_metadata"])

        ret["hq_videos"] = torch.stack(ret["hq_videos"])
        ret["lq_videos"] = torch.stack(ret["lq_videos"])
        ret["hq_images"] = torch.stack(ret["hq_images"])
        ret["lq_images"] = torch.stack(ret["lq_images"])
        ret["prompt_embeddings"] = torch.stack(ret["prompt_embeddings"])
        if len(ret["encoded_hq_videos"]) > 0:
            ret["encoded_hq_videos"] = torch.stack(ret["encoded_hq_videos"])
        if len(ret["encoded_lq_videos"]) > 0:
            ret["encoded_lq_videos"] = torch.stack(ret["encoded_lq_videos"])

        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        is_image_batch = random.random() < self.args.image_ratio

        prompt_embedding = batch["prompt_embeddings"]
        if self.args.is_latent:
            lq_latent = batch["encoded_lq_videos"]
        else:
            with torch.no_grad():
                if is_image_batch:
                    lq_videos = batch["lq_images"]
                    hq_videos = batch["hq_images"]
                else:
                    lq_videos = batch["lq_videos"]
                    hq_videos = batch["hq_videos"]

                self.components.vae.to(self.accelerator.device)
                latents = []
                for i in range(lq_videos.shape[2]):
                    frame = lq_videos[:, :, i:i+1, :, :]
                    latent_i = self.encode_video(frame)
                    latents.append(latent_i)
                lq_latent = torch.cat(latents, dim=2)

        # hq videos: [-1, 1], [B, C, F, H, W]
        hq_videos = (hq_videos * 0.5 + 0.5).clamp(0.0, 1.0)
        # GT: [0, 1], [B, C, F, H, W]
        hq_videos = hq_videos.to(self.accelerator.device)

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = lq_latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            lq_first_frame = lq_latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            lq_latent = torch.cat([lq_first_frame.repeat(1, 1, ncopy, 1, 1), lq_latent], dim=2)

            assert lq_latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = lq_latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=lq_latent.dtype)

        lq_latent = lq_latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W]

        # Noise Step (Select)
        if self.args.noise_step != 0:
            add_timesteps = torch.full(
                (batch_size,),
                fill_value=self.args.noise_step,
                dtype=torch.long,
                device=self.accelerator.device,
            )

            # Add noise to latent
            noise = torch.randn_like(lq_latent)
            lq_latent = self.components.scheduler.add_noise(lq_latent, noise, add_timesteps)
        
        timesteps = torch.full(
            (batch_size,),
            fill_value=self.args.sr_noise_step,
            dtype=torch.long,
            device=self.accelerator.device,
        )

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise (actual is velocity)
        predicted_noise = self.components.transformer(
            hidden_states=lq_latent,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]
        
        # Denoise: x0 (x0' = αt x_t - σt * vθ(z_t))
        latent_pred = self.components.scheduler.get_velocity(
            predicted_noise, lq_latent, timesteps
        )

        # generate video
        if patch_size_t is not None and ncopy > 0:
            latent_pred = latent_pred[:, ncopy:, :, :, :]
        # from [B, F, C, H, W] to [B, C, F, H, W]
        latent_pred = latent_pred.permute(0, 2, 1, 3, 4)
        latent_pred = 1 / self.components.vae.config.scaling_factor * latent_pred
        decoded_frames = []
        for i in range(latent_pred.shape[2]):
            latent_frame = latent_pred[:, :, i:i+1, :, :]
            frame_decoded = self.components.vae.decode(latent_frame).sample
            decoded_frames.append(frame_decoded)
        video_generate = torch.cat(decoded_frames, dim=2)
        video_generate = (video_generate * 0.5 + 0.5).clamp(0.0, 1.0)

        # Compute loss
        loss_dict = {}
        mse_loss = F.mse_loss(video_generate.float(), hq_videos.float(), reduction="mean")

        perceptual_loss = 0.0
        for f in range(video_generate.shape[2]):
            pred_frame = video_generate[:, :, f, :, :].to(dtype=torch.float32, device=self.accelerator.device)  # [B, C, H, W]
            gt_frame = hq_videos[:, :, f, :, :].to(dtype=torch.float32, device=self.accelerator.device)

            if self.args.ea_dists_weight > 0:
                dists_loss = self.dists_loss(pred_frame, gt_frame)
                edge_loss = self.dists_loss(
                    self.edge_detection_model(pred_frame), 
                    self.edge_detection_model(gt_frame)
                )
                perceptual_loss = perceptual_loss + dists_loss + edge_loss
            elif self.args.dists_weight > 0:
                dists_loss = self.dists_loss(pred_frame, gt_frame)
                perceptual_loss = perceptual_loss + dists_loss
            elif self.args.ea_lpips_weight > 0:
                lpips_loss = self.lpips_loss(pred_frame, gt_frame)
                edge_loss = self.lpips_loss(
                    self.edge_detection_model(pred_frame), 
                    self.edge_detection_model(gt_frame)
                )
                perceptual_loss = perceptual_loss + lpips_loss + edge_loss
            elif self.args.lpips_weight > 0:
                lpips_loss = self.lpips_loss(pred_frame, gt_frame)
                perceptual_loss = perceptual_loss + lpips_loss

        if self.args.ea_dists_weight > 0:
            perceptual_loss = perceptual_loss / (video_generate.shape[2] * 2)
            perceptual_loss = perceptual_loss * self.args.ea_dists_weight
        elif self.args.dists_weight > 0:
            perceptual_loss = perceptual_loss / video_generate.shape[2]
            perceptual_loss = perceptual_loss * self.args.dists_weight
        elif self.args.ea_lpips_weight > 0:
            perceptual_loss = perceptual_loss / (video_generate.shape[2] * 2)
            perceptual_loss = perceptual_loss * self.args.ea_lpips_weight
        elif self.args.lpips_weight > 0:
            perceptual_loss = perceptual_loss / video_generate.shape[2]
            perceptual_loss = perceptual_loss * self.args.lpips_weight
        
        if video_generate.shape[2] > 1:
            # video_generate, hq_videos: [B, C, F, H, W]
            video_generate = video_generate.to(dtype=torch.float32, device=self.accelerator.device)
            hq_videos = hq_videos.to(dtype=torch.float32, device=self.accelerator.device)
            diff_gen = video_generate[:, :, 1:, :, :] - video_generate[:, :, :-1, :, :]  # [B, C, F-1, H, W]
            diff_gt  = hq_videos[:, :, 1:, :, :]      - hq_videos[:, :, :-1, :, :]       # [B, C, F-1, H, W]
            frame_diff_loss = torch.nn.functional.l1_loss(diff_gen, diff_gt)
            frame_diff_loss = frame_diff_loss * self.args.frame_diff_weight
        else:
            frame_diff_loss = torch.tensor(0.0, device=self.accelerator.device)

        loss = mse_loss + perceptual_loss + frame_diff_loss

        loss_dict['perceptual_loss'] = perceptual_loss.detach().item() # original loss
        loss_dict['mse_loss'] = mse_loss.detach().item()
        loss_dict['frame_diff_loss'] = frame_diff_loss.detach().item()
        loss_dict['loss'] = loss.detach().item()

        return (loss, loss_dict)

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt, video, ref_video = eval_data["prompt"], eval_data["video_tensor"], eval_data["ref_video"]

        # [F, C, H, W]
        H_, W_ = video.shape[2], video.shape[3]
        video = torch.nn.functional.interpolate(video, size=(H_*4, W_*4), mode="bilinear", align_corners=False)
        self.__frame_transform = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)] # -1, 1
        )
        video = torch.stack([self.__frame_transform(f) for f in video], dim=0)
        video = video.unsqueeze(0)
        # [B, C, F, H, W]
        video = video.permute(0, 2, 1, 3, 4).contiguous()

        with torch.no_grad():
            self.components.vae.to(self.accelerator.device)
            latent = self.encode_video(video)

            patch_size_t = self.state.transformer_config.patch_size_t
            if patch_size_t is not None:
                ncopy = latent.shape[2] % patch_size_t
                # Copy the first frame ncopy times to match patch_size_t
                first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
                latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)

                assert latent.shape[2] % patch_size_t == 0

            batch_size, num_channels, num_frames, height, width = latent.shape

            # Get prompt embeddings
            self.components.text_encoder.to(self.accelerator.device)
            prompt_embedding = self.encode_text(prompt)
            _, seq_len, _ = prompt_embedding.shape
            prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

            latent = latent.permute(0, 2, 1, 3, 4)

            # Add noise to latent (Select)
            if self.args.noise_step != 0:
                noise = torch.randn_like(latent)
                add_timesteps = torch.full(
                    (batch_size,),
                    fill_value=self.args.noise_step,
                    dtype=torch.long,
                    device=self.accelerator.device,
                )
                latent = self.components.scheduler.add_noise(latent, noise, add_timesteps)

            timesteps = torch.full(
                (batch_size,),
                fill_value=self.args.sr_noise_step,
                dtype=torch.long,
                device=self.accelerator.device,
            )

            # Prepare rotary embeds
            vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
            transformer_config = self.state.transformer_config
            rotary_emb = (
                self.prepare_rotary_positional_embeddings(
                    height=height * vae_scale_factor_spatial,
                    width=width * vae_scale_factor_spatial,
                    num_frames=num_frames,
                    transformer_config=transformer_config,
                    vae_scale_factor_spatial=vae_scale_factor_spatial,
                    device=self.accelerator.device,
                )
                if transformer_config.use_rotary_positional_embeddings
                else None
            )

            # Predict noise (actual is velocity)
            predicted_noise = self.components.transformer(
                hidden_states=latent,
                encoder_hidden_states=prompt_embedding,
                timestep=timesteps,
                image_rotary_emb=rotary_emb,
                return_dict=False,
            )[0]
            
            # Denoise: x0 (x0' = αt x_t - σt * vθ(z_t))
            latent_generate = self.components.scheduler.get_velocity(
                predicted_noise, latent, timesteps
            )
            # get_velocity (function): velocity = sqrt_alpha_prod * input2 - sqrt_one_minus_alpha_prod * input1
            # get_velocity (here): latent_pred = sqrt_alpha_prod * latent - sqrt_one_minus_alpha_prod * predicted_noise

            # generate video
            if patch_size_t is not None and ncopy > 0:
                latent_generate = latent_generate[:, ncopy:, :, :, :]
            video = pipe.decode_latents(latent_generate)
            video = pipe.video_processor.postprocess_video(video=video, output_type='pil')
            video = CogVideoXPipelineOutput(frames=video).frames[0]
            
        return [("video", video)]

    # Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
    def get_resize_crop_region_for_grid(self, src, tgt_width, tgt_height):
        tw = tgt_width
        th = tgt_height
        h, w = src
        r = h / w
        if r > (th / tw):
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        crop_top = int(round((th - resize_height) / 2.0))
        crop_left = int(round((tw - resize_width) / 2.0))

        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        p = transformer_config.patch_size
        p_t = transformer_config.patch_size_t

        base_size_width = transformer_config.sample_width // p
        base_size_height = transformer_config.sample_height // p

        if p_t is None:
            # CogVideoX 1.0
            grid_crops_coords = self.get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=transformer_config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=transformer_config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
            )

        return freqs_cos, freqs_sin

register("dove-s2", "lora", DOVES2Trainer)
