"""RAFT-based eWarp with RGB inputs in [0, 255] for flow estimation."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .core.raft import RAFT
from .core.utils.utils import InputPadder
from . import warp_utils

METRIC_PROTOCOL = 'raft_rgb255'
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

# Adapted from EvalCrafter/metrics/RAFT/optical_flow_scores.py, itself adapted
# from https://github.com/phoenix104104/fast_blind_video_consistency
def compute_video_warping_error(video_path, model, device):

    cap = cv2.VideoCapture(video_path)
    frames = []
    warping_error = 0
    err = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)
        frames.append(frame)

    # Num = 5
    Num = len(frames)
    tensor_frames = torch.stack([torch.from_numpy(frame) for frame in frames])
    # for i in range(Num):
    N = len(tensor_frames)
    indices = torch.linspace(0, N - 1, Num).long()
    extracted_frames = torch.index_select(tensor_frames, 0, indices)
    with torch.no_grad():
        for i in range(Num - 1):
            frame1 = extracted_frames[i]
            frame2 = extracted_frames[i + 1]

            # Calculate optical flow using Farneback method
            img1 = frame1.permute(2,0,1).float().unsqueeze(0).to(device)/ 255.0
            img2 = frame2.permute(2,0,1).float().unsqueeze(0).to(device)/ 255.0
            # img1 = torch.tensor(img2tensor(frame1)).float().to(device)
            # img2 = torch.tensor(img2tensor(frame2)).float().to(device)

            # Downsample the images by a factor of 2
            img1 = F.interpolate(img1, scale_factor=0.5, mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, scale_factor=0.5, mode='bilinear', align_corners=False)

            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)

            ### compute fw flow

            _, fw_flow = model(img1 * 255.0, img2 * 255.0, iters=20, test_mode=True) # with optical flow model: RAFT
            fw_flow = warp_utils.tensor2img(fw_flow)
            # Clear cache and temporary data
            torch.cuda.empty_cache()

            ### compute bw flow
            _, bw_flow = model(img2 * 255.0, img1 * 255.0, iters=20, test_mode=True) # with optical flow model: RAFT
            bw_flow = warp_utils.tensor2img(bw_flow)
            torch.cuda.empty_cache()

            ### compute occlusion
            fw_occ, warp_img2 = warp_utils.detect_occlusion(bw_flow, fw_flow, img2)
            warp_img2 = torch.tensor(warp_img2).float().to(device)
            fw_occ = torch.tensor(fw_occ).float().to(device)

            ### load flow
            flow = fw_flow

            ### load occlusion mask
            occ_mask = fw_occ
            noc_mask = 1 - occ_mask

            # ipdb.set_trace()
            diff = (warp_img2- img1) * noc_mask
            diff_squared = diff ** 2

            # Calculate the sum and mean
            N = torch.sum(noc_mask)
            if N == 0:
                N = diff_squared.numel()
            # ipdb.set_trace()
            err += torch.sum(diff_squared) / N

    warping_error = err / (len(extracted_frames) - 1)

    return warping_error


def metric_metadata(args):
    return {
        'protocol': METRIC_PROTOCOL,
        'implementation_base': 'EvalCrafter/metrics/RAFT/optical_flow_scores.py',
        'change_from_legacy': 'multiply both inputs by 255 at the two RAFT calls only',
        'flow_source': 'prediction',
        'flow_model': str(Path(args.model).resolve()),
        'raft_input_range': [0, 255],
        'photometric_input_range': [0, 1],
        'resize_scale': 0.5,
        'raft_iterations': 20,
        'error': 'sum_RGB_squared_error / visible_pixels',
        'occlusion': 'forward_backward_squared_consistency_and_motion_boundary',
        'exclude_padding': False,
        'exclude_out_of_bounds': False,
        'zero_visible_pair': 'count as zero; retain in pair average',
        'aggregation': 'mean_all_adjacent_pairs_per_video_then_mean_videos',
        'score_scale': 1000,
    }


def Ewarp(args):
    """Evaluate videos, keeping the project's historical x1000 output scale."""
    if args.metric != 'warping_error':
        raise ValueError(f'Unsupported metric: {args.metric}')
    from natsort import natsorted

    video_paths = natsorted(
        p for p in Path(args.pred).iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_paths:
        raise ValueError(f'No videos found: {args.pred}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model = model.module.to(device).eval()
    model.args.mixed_precision = False
    results = {}
    for video_path in tqdm(video_paths):
        # Fail explicitly on malformed inputs; the metric function stays intact.
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.read()[0] or not cap.read()[0]:
                raise ValueError(f'At least two readable frames are required: {video_path}')
        finally:
            cap.release()
        score = compute_video_warping_error(str(video_path), model, device).item() * 1000
        if not np.isfinite(score):
            raise ValueError(f'Non-finite score: {video_path}')
        if video_path.stem in results:
            raise ValueError(f'Duplicate video stem: {video_path.stem}')
        results[video_path.stem] = score
        print(f'Vid: {video_path.name}, {METRIC_PROTOCOL}: {score}', flush=True)
    average_score = sum(results.values()) / len(results)
    print(f'Final average {METRIC_PROTOCOL}: {average_score}, Total videos: {len(results)}')
    return results, average_score
