import os
import cv2
import json
import torch
import pyiqa
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# 0 ~ 1
to_tensor = transforms.ToTensor()
video_exts = ['.mp4', '.avi', '.mov', '.mkv']
fr_metrics = ['psnr', 'ssim', 'lpips', 'dists']


def is_video_file(filename):
    return any(filename.lower().endswith(ext) for ext in video_exts)

def rgb_to_y(img):
    # Assumes img is [1, 3, H, W] in [0,1], returns [1, 1, H, W]
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 0.0625
    return y

def crop_border(img, crop):
    return img[:, :, crop:-crop, crop:-crop]

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


def crop_img_center(img, target_h, target_w):
    _, h, w = img.shape
    top = max((h - target_h) // 2, 0)
    left = max((w - target_w) // 2, 0)
    return img[:, top:top+target_h, left:left+target_w]

def crop_img_top_left(img, target_h, target_w):
    # Crop image from top-left corner to (target_h, target_w)
    return img[:, :target_h, :target_w]

def match_resolution(gt_frames, pred_frames, is_center=False, name=None):
    t = min(gt_frames.shape[0], pred_frames.shape[0])
    gt_frames = gt_frames[:t]
    pred_frames = pred_frames[:t]
    _, _, h_g, w_g = gt_frames.shape
    _, _, h_p, w_p = pred_frames.shape

    target_h = min(h_g, h_p)
    target_w = min(w_g, w_p)

    if (h_g != h_p or w_g != w_p) and name:
        if is_center:
            print(f"[{name}] Resolution mismatch detected: GT is ({h_g}, {w_g}), Pred is ({h_p}, {w_p}). Both GT and Pred were center cropped to ({target_h}, {target_w}).")
        else:
            print(f"[{name}] Resolution mismatch detected: GT is ({h_g}, {w_g}), Pred is ({h_p}, {w_p}). Both GT and Pred were top-left cropped to ({target_h}, {target_w}).")

    if is_center:
        gt_frames = torch.stack([crop_img_center(f, target_h, target_w) for f in gt_frames])
        pred_frames = torch.stack([crop_img_center(f, target_h, target_w) for f in pred_frames])
    else:
        gt_frames = torch.stack([crop_img_top_left(f, target_h, target_w) for f in gt_frames])
        pred_frames = torch.stack([crop_img_top_left(f, target_h, target_w) for f in pred_frames])

    return gt_frames, pred_frames


def init_models(metrics, device):
    models = {}
    for name in metrics:
        try:
            models[name] = pyiqa.create_metric(name).to(device).eval()
        except Exception as e:
            print(f"Failed to initialize metric '{name}': {e}")
    return models

def compute_metrics(pred_frames, gt_frames, models, device, batch_mode, crop, test_y_channel):
    if batch_mode:
        pred_batch = pred_frames.to(device)  # [F, C, H, W]
        gt_batch = gt_frames.to(device)      # [F, C, H, W]

        results = {}
        for name, model in models.items():
            if name in fr_metrics:
                pred_eval = pred_batch
                gt_eval = gt_batch
                if crop > 0:
                    pred_eval = crop_border(pred_eval, crop)
                    gt_eval = crop_border(gt_eval, crop)
                if test_y_channel:
                    pred_eval = rgb_to_y(pred_eval)
                    gt_eval = rgb_to_y(gt_eval)
                values = model(pred_eval, gt_eval)  # [F]
            else:
                values = model(pred_batch)  # no-reference
            results[name] = round(values.mean().item(), 4)
        return results

    else:
        results = {name: [] for name in models}
        for pred, gt in zip(pred_frames, gt_frames):
            pred = pred.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)

            for name, model in models.items():
                if name in fr_metrics:
                    pred_eval = pred
                    gt_eval = gt
                    if crop > 0:
                        pred_eval = crop_border(pred_eval, crop)
                        gt_eval = crop_border(gt_eval, crop)
                    if test_y_channel:
                        pred_eval = rgb_to_y(pred_eval)
                        gt_eval = rgb_to_y(gt_eval)
                    value = model(pred_eval, gt_eval).item()
                else:
                    value = model(pred).item()
                results[name].append(value)

        return {k: round(np.mean(v), 4) for k, v in results.items()}


def process(gt_root, pred_root, out_path, metrics, batch_mode, crop, test_y_channel, is_center):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    models = init_models(metrics, device)

    has_gt = bool(gt_root and os.path.exists(gt_root))

    if has_gt:
        gt_files = {os.path.splitext(f)[0]: os.path.join(gt_root, f) for f in os.listdir(gt_root)}
    pred_files = {os.path.splitext(f)[0]: os.path.join(pred_root, f) for f in os.listdir(pred_root)}

    pred_names = sorted(pred_files.keys())
    results = {}
    aggregate = {metric: [] for metric in metrics}

    for name in tqdm(pred_names, desc="Evaluating"):
        # # valida
        # name_hr = name.replace('_CAT_A_x4', '').replace('img_', 'img')
        name_hr = name
        if has_gt and name_hr not in gt_files:
            print(f"Skipping {name_hr}: no matching GT file.")
            continue

        pred_path = pred_files[name]
        gt_path = gt_files[name_hr] if has_gt else None

        try:
            pred_frames = load_sequence(pred_path)

            if has_gt:
                gt_frames = load_sequence(gt_path)
                gt_frames, pred_frames = match_resolution(gt_frames, pred_frames, is_center=is_center, name=name)
                scores = compute_metrics(pred_frames, gt_frames, models, device, batch_mode, crop, test_y_channel)
            else:
                nr_models = {k: v for k, v in models.items() if k not in fr_metrics}
                if not nr_models:
                    print(f"Skipping {name}: GT is not provided and no NR-IQA metrics found.")
                    continue
                dummy_gt = pred_frames
                scores = compute_metrics(pred_frames, dummy_gt, nr_models, device, batch_mode, crop, test_y_channel)

            results[name] = scores
            for k in scores:
                aggregate[k].append(scores[k])
        except Exception as e:
            print(f"Error processing {name}: {e}")

    print("\nPer-sample Results:")
    for name in sorted(results):
        print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    print("\nOverall Average Results:")
    count = len(results)
    if count > 0:
        overall_avg = {k: round(np.mean(v), 4) for k, v in aggregate.items()}
        for k, v in overall_avg.items():
            print(f"{k.upper()}: {v:.4f}")
    else:
        overall_avg = {}
        print("No valid samples were processed.")

    print(f"\nProcessed {count} samples.")

    output = {
        "per_sample": results,
        "average": overall_avg,
        "count": count
    }

    os.makedirs(out_path, exist_ok=True)
    out_name = 'metrics_'
    for metric in metrics:
        out_name += f"{metric}_"
    out_name = out_name.rstrip('_') + '.json'
    out_path = os.path.join(out_path, out_name)

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='', help='Path to GT folder (optional for NR-IQA)')
    parser.add_argument('--pred', type=str, required=True, help='Path to predicted results folder')
    parser.add_argument('--out', type=str, default='', help='Path to save JSON output (as directory)')
    parser.add_argument('--metrics', type=str, default='psnr,ssim,clipiqa',
                        help='Comma-separated list of metrics: psnr,ssim,clipiqa,lpips,...')
    parser.add_argument('--batch_mode', action='store_true', help='Use batch mode for metrics computation')
    parser.add_argument('--crop', type=int, default=0, help='Crop border size for PSNR/SSIM')
    parser.add_argument('--test_y_channel', action='store_true', help='Use Y channel for PSNR/SSIM')
    parser.add_argument('--is_center', action='store_true', help='Use center crop for PSNR/SSIM')

    args = parser.parse_args()

    if args.out == '':
        out = args.pred
    else:
        out = args.out
    metric_list = [m.strip().lower() for m in args.metrics.split(',')]
    process(args.gt, args.pred, out, metric_list, args.batch_mode, args.crop, args.test_y_channel, args.is_center)
