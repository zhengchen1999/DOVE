#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "checkpoint/DOVE-s1/ckpt-10000-sft"
    --model_name "dove-s2"
    --model_type "real-sr-image-video"
    --training_type "sft"
)

# LORA_ARGS=(
#     --rank 64
#     --lora_alpha 64
# )

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "checkpoint/DOVE-s2"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "../datasets/train"
    --video_column "../datasets/train/HQ-VSR.txt"
    --image_data_root "../datasets/train"
    --image_column "../datasets/train/DIV2K_train_HR.txt"
    --train_resolution "2x320x640"  # (frames x height x width), frames should be 8N+1
    # --crop_mode "resize_random_crop"
    --image_ratio 0.8
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 10 # number of training epochs
    --train_steps 500
    --seed 42 # random seed
    --batch_size 2
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
    --learning_rate 5e-6
    --gradient_checkpointing true
    --max_grad_norm 0.1
    --lr_scheduler "constant_with_warmup"  # ["constant_with_warmup", "decay_with_warmup"]
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
    --stastic_frequency 100
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 100 # save checkpoint every x steps
    --checkpointing_limit 3 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "data/VideoSR/test/UDM10"
    --validation_steps 100  # should be multiple of checkpointing_steps
    --validation_videos "video_real_v0.txt"
    --validation_ref_videos "video.txt"
    # --validation_prompts "prompts.txt"
    --gen_fps 8
    --raw_test true
    --num_inference_steps 1
    --eval_metric_list "psnr,ssim,lpips,dists,clipiqa"  # ["psnr", "ssim", "lpips", "dists", "clipiqa", "musiq", "maniqa", 'niqe']
)

# SR parameters
SR_ARGS=(
    --is_latent false
    --is_cache true
    --empty_prompt true
    --prompt_cache "prompt_embeddings"
    --sr_noise_step 399
    --noise_step 0
    --degradation_config "configs/degradation_image_video.yaml"
)

# Perceptual Loss parameters
Per_ARGS=(
    --use_perceptual_loss true
    --dists_weight 1.0
    --frame_diff_weight 1.0
)

# Combine all arguments and launch training
accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${SR_ARGS[@]}" \
    "${Per_ARGS[@]}" \