#!/usr/bin/env bash

# UDM10
python inference_script.py \
    --input_dir datasets/test/UDM10/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/DOVE/UDM10 \
    --is_vae_st \
    --png_save

python eval_metrics.py \
    --gt datasets/test/UDM10/GT \
    --pred results/DOVE/UDM10 \
    --metrics psnr,ssim,lpips,dists,clipiqa

# SPMCS
python inference_script.py \
    --input_dir datasets/test/SPMCS/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/DOVE/SPMCS \
    --is_vae_st \
    --png_save

python eval_metrics.py \
    --gt datasets/test/UDM10/GT \
    --pred results/DOVE/SPMCS \
    --metrics psnr,ssim,lpips,dists,clipiqa

# YouHQ40
python inference_script.py \
    --input_dir datasets/test/YouHQ40/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/DOVE/YouHQ40 \
    --is_vae_st \
    --png_save

python eval_metrics.py \
    --gt datasets/test/UDM10/GT \
    --pred results/DOVE/YouHQ40 \
    --metrics psnr,ssim,lpips,dists,clipiqa

# RealVSR
python inference_script.py \
    --input_dir datasets/test/RealVSR/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/DOVE/RealVSR \
    --is_vae_st \
    --upscale 1 \
    --png_save

python eval_metrics.py \
    --gt datasets/test/UDM10/GT \
    --pred results/DOVE/RealVSR \
    --metrics psnr,ssim,lpips,dists,clipiqa

# MVSR4x
python inference_script.py \
    --input_dir datasets/test/MVSR4x/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/DOVE/MVSR4x \
    --is_vae_st \
    --upscale 1 \
    --png_save

python eval_metrics.py \
    --gt datasets/test/UDM10/GT \
    --pred results/DOVE/MVSR4x \
    --metrics psnr,ssim,lpips,dists,clipiqa

# VideoLQ
python inference_script.py \
    --input_dir datasets/test/VideoLQ/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/DOVE/VideoLQ \
    --is_vae_st \
    --png_save

python eval_metrics.py \
    --gt datasets/test/UDM10/GT \
    --pred results/DOVE/VideoLQ \
    --metrics clipiqa
