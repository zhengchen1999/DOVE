"""Frame metrics following the supplied eval_metric bundle."""
import warnings

FULL_REFERENCE = {"psnr", "ssim", "lpips", "dists"}
SUPPORTED = FULL_REFERENCE | {"clipiqa", "musiq", "niqe", "brisque"}


def align(pred, gt, alignment, name):
    if pred.shape == gt.shape:
        return pred, gt
    if alignment == "strict":
        raise ValueError(
            f"{name}: prediction {tuple(pred.shape)} != GT {tuple(gt.shape)}; "
            "use --alignment top-left or center only when intentional."
        )
    t = min(len(pred), len(gt))
    h, w = min(pred.shape[-2], gt.shape[-2]), min(pred.shape[-1], gt.shape[-1])
    warnings.warn(f"{name}: aligning {tuple(pred.shape)} and {tuple(gt.shape)} "
                  f"to {t} frames, {h}x{w}, {alignment} crop")

    def crop(x):
        top = (x.shape[-2] - h) // 2 if alignment == "center" else 0
        left = (x.shape[-1] - w) // 2 if alignment == "center" else 0
        return x[:t, :, top:top + h, left:left + w]
    return crop(pred), crop(gt)


def evaluation_inputs(pred, gt, metric, crop, test_y_channel):
    if metric in FULL_REFERENCE:
        if crop:
            if min(pred.shape[-2:]) <= 2 * crop:
                raise ValueError("Crop border removes the entire image")
            pred = pred[:, :, crop:-crop, crop:-crop]
            gt = gt[:, :, crop:-crop, crop:-crop]
        # LPIPS and DISTS require RGB even when PSNR/SSIM use luminance.
        if test_y_channel and metric in {"psnr", "ssim"}:
            def to_y(x):
                return (0.257 * x[:, 0:1] + 0.504 * x[:, 1:2]
                        + 0.098 * x[:, 2:3] + 0.0625)
            pred, gt = to_y(pred), to_y(gt)
    return pred, gt


def evaluate(sequences, references, metrics, args, device):
    import pyiqa
    import torch
    from pyiqa.default_model_configs import DEFAULT_CONFIGS
    from .io import load_sequence

    models, configurations = {}, {}
    for name in metrics:
        options = {}
        if args.test_y_channel and name in {"psnr", "ssim"}:
            # The supplied script converts Y itself; prevent a second conversion.
            options["test_y_channel"] = False
        models[name] = pyiqa.create_metric(name, device=device, **options).eval()
        configurations[name] = dict(DEFAULT_CONFIGS[name]["metric_opts"], **options)
    results, details = {}, {}
    with torch.no_grad():
        for name, path in sequences.items():
            pred = load_sequence(path)
            original_shape = list(pred.shape)
            gt = load_sequence(references[name]) if references else None
            gt_shape = list(gt.shape) if gt is not None else None
            if gt is not None:
                pred, gt = align(pred, gt, args.alignment, name)
            details[name] = {
                "prediction_shape": original_shape, "gt_shape": gt_shape,
                "evaluated_shape": list(pred.shape),
            }
            results[name] = {}
            for metric, model in models.items():
                values = []
                for start in range(0, len(pred), args.batch_size):
                    p = pred[start:start + args.batch_size].to(device)
                    g = gt[start:start + args.batch_size].to(device) if gt is not None else None
                    p, g = evaluation_inputs(p, g, metric, args.crop, args.test_y_channel)
                    score = model(p, g) if metric in FULL_REFERENCE else model(p)
                    if score.numel() != len(p):
                        raise ValueError(f"{metric} did not return one score per frame")
                    values.extend(score.detach().double().cpu().reshape(-1).tolist())
                results[name][metric] = sum(values) / len(values)
            print(f"{name}: {results[name]}", flush=True)
    return results, {"pyiqa_version": pyiqa.__version__, "sequences": details,
                     "metric_options": configurations,
                     "aggregation": "mean_frames_per_video_then_mean_videos",
                     "y_channel_metrics": ["psnr", "ssim"] if args.test_y_channel else [],
                     "y_formula": "0.257 R + 0.504 G + 0.098 B + 0.0625",
                     "crop_border": args.crop, "alignment": args.alignment}
