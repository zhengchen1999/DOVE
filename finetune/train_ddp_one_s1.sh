#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX1.5-5B"
    --model_name "dove-s1"
    --model_type "real-sr"
    --training_type "sft"
)

# LORA_ARGS=(
#     --rank 64
#     --lora_alpha 64
# )

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "checkpoint/DOVE-s1"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "../datasets/train"
    --video_column "../datasets/train/HQ-VSR.txt"
    --train_resolution "25x320x640"  # (frames x height x width), frames should be 8N+1
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 1000 # number of training epochs
    --train_steps 10000
    --seed 42 # random seed
    --batch_size 2
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
    --learning_rate 2e-5
    --gradient_checkpointing true
    --max_grad_norm 0.1
    --lr_scheduler "constant_with_warmup"  # ["constant_with_warmup", "decay_with_warmup"]
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
    --stastic_frequency 500
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 1000 # save checkpoint every x steps
    --checkpointing_limit 3 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "../datasets/test/UDM10"
    --validation_steps 500  # should be multiple of checkpointing_steps
    --validation_videos "LQ-Video.txt"
    --validation_ref_videos "GT-Video.txt"
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
    --degradation_config "configs/degradation.yaml"
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
