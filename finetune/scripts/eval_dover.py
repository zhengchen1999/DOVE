import os
import sys
import numpy as np
script_path = os.path.abspath(sys.argv[0])
script_directory = os.path.dirname(script_path)
os.chdir(script_directory)
import cv2
import json
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from functools import partial
import shutil
import subprocess
import re
import imageio.v3 as iio

# 0 ~ 1
to_tensor = transforms.ToTensor()
video_exts = ['.mp4', '.avi', '.mov', '.mkv']
video_metrics  = ['dover']


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

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def img2video(subfolder_path, output_path, fps=8):
    # 2025.4.19
    img_tensor = read_image_folder(subfolder_path)
    if img_tensor is None:
        print(f"Failed to read images from {subfolder_path}")
        return
    img_tensor = img_tensor.permute(0, 2, 3, 1)  # [F, H, W, C]
    frames = (img_tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()  # [F, H, W, C]
    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec='libx264rgb',
        pixelformat='rgb24',
        macro_block_size=None,
        ffmpeg_params=['-crf', '0'],
    )
    print(f"Video saved to {output_path}")

def process(pred_root, out_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 统计文件夹和视频文件的数量
    all_items = os.listdir(pred_root)
    folders_count = 0
    videos_count = 0
    
    for item in all_items:
        item_path = os.path.join(pred_root, item)
        if os.path.isdir(item_path):
            folders_count += 1
        elif is_video_file(item_path):
            videos_count += 1
    
    # 判断哪种类型更多，并设置is_folder_dominant变量
    is_folder_dominant = folders_count >= videos_count
    print(f"Found {folders_count} folders and {videos_count} videos. Folder dominant: {is_folder_dominant}")
    
    # 根据主导类型构建pred_files
    pred_files = {}
    if is_folder_dominant:
        # 如果文件夹更多，只收集文件夹
        for item in all_items:
            item_path = os.path.join(pred_root, item)
            if os.path.isdir(item_path):
                name = os.path.splitext(item)[0]
                pred_files[name] = item_path
    else:
        # 如果视频更多，只收集视频文件
        for item in all_items:
            item_path = os.path.join(pred_root, item)
            if is_video_file(item_path):
                name = os.path.splitext(item)[0]
                pred_files[name] = item_path
    
    # 如果没有找到有效文件，退出
    if not pred_files:
        print("No valid folders or video files found in the specified directory.")
        return
    
    pred_names = sorted(pred_files.keys())
    
    input_path = pred_root  # 默认输入对象是视频

    # 输入对象是文件夹，需要转换为视频
    if is_folder_dominant:
        input_path = os.path.join(out_path, "temp")
        os.makedirs(input_path, exist_ok=True)
        
        for name in pred_names:
            subfolder_path = pred_files[name]
            if os.path.isdir(subfolder_path):
                video_path = os.path.join(input_path, f"{name}.mp4")
                img2video(subfolder_path, video_path)
                pred_files[name] = video_path
    else:
        input_path = os.path.join(out_path, "temp")
        os.makedirs(input_path, exist_ok=True)
        for name in pred_names:
            video_path = pred_files[name]
            if is_video_file(video_path):
                new_video_path = os.path.join(input_path, f"{name}.mp4")
                shutil.copy(video_path, new_video_path)
                pred_files[name] = new_video_path
    
    from DOVER.evaluate_a_set_of_videos import evaluate_set as dover

    input_path = os.path.abspath(input_path)

    dover_results = dover(input_path, device=device)

    results = {}

    for i, name in enumerate(pred_names):
        results[name] = dover_results[i-1]

    count = len(results)
    
    if count > 0:
        overall_avg = np.mean([results[name] for name in results])
        print(results)
    else:
        overall_avg = {}
        print("No valid samples were processed.")

    print(f"\nProcessed {count} samples.")
    print(f"Average score: {overall_avg}")
    output = {
        "per_sample": results,
        "average": overall_avg,
        "count": count
    }

    os.makedirs(out_path, exist_ok=True)
    out_name = 'metrics_dover.json'
    file_out_path = os.path.join(out_path, out_name)

    with open(file_out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    output_folder = os.path.join(out_path,"temp")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='Path to predicted results folder')
    parser.add_argument('--out', type=str, default='', help='Path to save JSON output (as directory)')
    args = parser.parse_args()

    if args.out == '':
        out = args.pred
    else:
        out = args.out
    process(args.pred, out)