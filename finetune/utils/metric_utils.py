import torch
import numpy as np
import pyiqa
import torchvision.transforms as transforms
from torch import nn 

fr_metrics = ['psnr', 'ssim', 'lpips', 'dists']  # full-reference metrics

def rgb_to_y(img):
    # img: [B, 3, H, W] in [0,1]
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 0.0625
    return y

def crop_border(img, crop):
    # C, H, W
    return img[:, :, crop:-crop, crop:-crop]

def crop_img_center(img, target_h, target_w):
    # C, H, W
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

@torch.no_grad()
def evaluate_video_metrics(
    pred_video: torch.Tensor,
    ref_video: torch.Tensor = None,
    models: dict = {},
    crop: int = 0,
    test_y_channel: bool = False,
    device: torch.device = None,
    batch_mode: bool = False,
    name: str = None,
) -> dict:
    """
    Compute quality metrics between predicted and reference video.
    
    Args:
        pred_video: [F, C, H, W] in [0,1]
        ref_video: [F, C, H, W] in [0,1] or None
        models: dict of {metric_name: metric_model}
        crop: border cropping before metric
        test_y_channel: convert RGB to Y
        device: torch.device
        batch_mode: whether to compute in batch

    Returns:
        dict: {metric_name: value}
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # If ref_video exists, match resolution and assert shape
    if ref_video is not None:
        ref_video, pred_video = match_resolution(ref_video, pred_video, name=name)
        assert pred_video.shape == ref_video.shape, f"Shape mismatch: Pred {pred_video.shape}, Ref {ref_video.shape}"
        ref_video = ref_video.to(device)

    pred_video = pred_video.to(device)
    results = {}

    if batch_mode:
        for name, model in models.items():
            pred_eval = pred_video
            gt_eval = ref_video if ref_video is not None else None

            if crop > 0:
                pred_eval = crop_border(pred_eval, crop)
                if gt_eval is not None:
                    gt_eval = crop_border(gt_eval, crop)

            if test_y_channel and gt_eval is not None and name in fr_metrics:
                pred_eval = rgb_to_y(pred_eval)
                gt_eval = rgb_to_y(gt_eval)

            if name in fr_metrics and gt_eval is not None:
                score = model(pred_eval, gt_eval)
            else:
                score = model(pred_eval)

            results[name] = round(score.mean().item(), 4)

    else:
        res_dict = {name: [] for name in models}
        for pred_idx in range(pred_video.shape[0]):
            pred = pred_video[pred_idx].unsqueeze(0)
            gt = ref_video[pred_idx].unsqueeze(0) if ref_video is not None else None

            for name, model in models.items():
                pred_eval = pred
                gt_eval = gt

                if crop > 0:
                    pred_eval = crop_border(pred_eval, crop)
                    if gt_eval is not None:
                        gt_eval = crop_border(gt_eval, crop)

                if test_y_channel and gt_eval is not None and name in fr_metrics:
                    pred_eval = rgb_to_y(pred_eval)
                    gt_eval = rgb_to_y(gt_eval)

                if name in fr_metrics and gt_eval is not None:
                    val = model(pred_eval, gt_eval).item()
                else:
                    val = model(pred_eval).item()

                res_dict[name].append(val)

        results = {k: round(np.mean(v), 4) for k, v in res_dict.items()}

    return results

if __name__ == '__main__':
    pred_path = 'results/Val-Metric-Demo-Retest/000.mp4'
    gt_path = 'dataset/VideoSR/UDM10/GT_video/000.mp4'

    import decord
    decord.bridge.set_bridge("torch")

    def read_video_with_decord(video_path):
        video_reader = decord.VideoReader(video_path)
        frames = video_reader.get_batch(list(range(len(video_reader))))  # [F, H, W, C], uint8
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [F, C, H, W], float32
        return frames
    
    import cv2
    def read_video_tensor(path):
        """Load video from mp4 and return tensor in [F, C, H, W], normalized to [0,1]."""
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))  # [C, H, W]
        cap.release()
        return torch.stack(frames, dim=0)  # [F, C, H, W]

    pred_video1 = read_video_with_decord(pred_path)
    gt_video1 = read_video_with_decord(gt_path)

    pred_video = read_video_tensor(pred_path)
    gt_video = read_video_tensor(gt_path)

    def compare_videos(t1: torch.Tensor, t2: torch.Tensor, name=''):
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"{name}  Max pixel diff  : {max_diff:.6f}")
        print(f"{name}  Mean pixel diff : {mean_diff:.6f}")

    compare_videos(pred_video1, pred_video, name='Predicted Video')

    compare_videos(gt_video1, gt_video, name='Ground Truth Video')

    models = {
        'psnr': pyiqa.create_metric('psnr').cuda().eval(),
        'ssim': pyiqa.create_metric('ssim').cuda().eval(),
        'lpips': pyiqa.create_metric('lpips').cuda().eval(),
        'dists': pyiqa.create_metric('dists').cuda().eval(),
        'clipiqa': pyiqa.create_metric('clipiqa').cuda().eval(),
        'musiq': pyiqa.create_metric('musiq').cuda().eval(),
        'niqe': pyiqa.create_metric('niqe').cuda().eval(),
    }
    results = evaluate_video_metrics(
        pred_video=pred_video,
        ref_video=gt_video,
        models=models,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        name='test_video'
    )

    print("Evaluation Results:")
    for k, v in results.items():
        print(f"{k.upper()}: {v}")

class EdgeDetectionModel(nn.Module):
    def __init__(self):
        super(EdgeDetectionModel, self).__init__()
        # Sobel filters for edge detection
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        sobel_x_kernel = torch.tensor([[-1., 0., 1.],
                                       [-2., 0., 2.],
                                       [-1., 0., 1.]])
        sobel_y_kernel = torch.tensor([[-1., -2., -1.],
                                       [ 0.,  0.,  0.],
                                       [ 1.,  2.,  1.]])
        
        self.sobel_x.weight = nn.Parameter(sobel_x_kernel.view(1, 1, 3, 3))
        self.sobel_y.weight = nn.Parameter(sobel_y_kernel.view(1, 1, 3, 3))
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def forward(self, x):
        # Convert to grayscale if needed
        if x.shape[1] == 3:
            x = transforms.Grayscale()(x)
        
        # Apply Sobel filters
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        
        # Calculate gradient magnitude (edge detection result)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        
        return edges