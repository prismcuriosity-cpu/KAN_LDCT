"""Metrics. Pure torch — no scikit-image dependency, runs on GPU, batched.

Reported per slice so that mean +/- std and paired significance tests are
possible.  Two protocols:

``full``   normalised [-1, 1] over the whole configured HU range (data_range 2)
``window`` the abdomen soft-tissue display window [-160, 240] HU, which is what
           the LDCT literature (RED-CNN, WGAN-VGG, CoreDiff, ...) reports.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _gauss_kernel(ws=11, sigma=1.5, device="cpu", dtype=torch.float32):
    c = torch.arange(ws, device=device, dtype=dtype) - (ws - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).view(1, 1, ws, ws)


def ssim(pred, target, data_range=2.0, ws=11, sigma=1.5):
    """(B,1,H,W) -> (B,) mean SSIM per image."""
    pred, target = pred.float(), target.float()
    k = _gauss_kernel(ws, sigma, pred.device, pred.dtype)
    pad = ws // 2
    mu1 = F.conv2d(pred, k, padding=pad)
    mu2 = F.conv2d(target, k, padding=pad)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(pred * pred, k, padding=pad) - mu1s
    s2 = F.conv2d(target * target, k, padding=pad) - mu2s
    s12 = F.conv2d(pred * target, k, padding=pad) - mu12
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1s + mu2s + c1) * (s1 + s2 + c2))
    return m.flatten(1).mean(1)


def psnr(pred, target, data_range=2.0):
    mse = (pred.float() - target.float()).pow(2).flatten(1).mean(1)
    return 10.0 * torch.log10((data_range ** 2) / mse.clamp_min(1e-12))


def rmse(pred, target):
    return (pred.float() - target.float()).pow(2).flatten(1).mean(1).sqrt()


def to_window(x, hu_min, hu_max, lo=-160.0, hi=240.0):
    """[-1,1] over [hu_min,hu_max]  ->  [0,1] over the display window."""
    hu = (x + 1.0) * 0.5 * (hu_max - hu_min) + hu_min
    return ((hu - lo) / (hi - lo)).clamp(0.0, 1.0)


def slice_metrics(pred, target, cfg):
    """dict of per-slice tensors: PSNR/SSIM/RMSE in both protocols + RMSE in HU."""
    hu_span = cfg.hu_max - cfg.hu_min
    pw = to_window(pred, cfg.hu_min, cfg.hu_max, *cfg.eval_window)
    tw = to_window(target, cfg.hu_min, cfg.hu_max, *cfg.eval_window)
    return {
        "psnr": psnr(pred, target, 2.0),
        "ssim": ssim(pred, target, 2.0),
        "rmse_hu": rmse(pred, target) * hu_span / 2.0,
        "psnr_w": psnr(pw, tw, 1.0),
        "ssim_w": ssim(pw, tw, 1.0),
        "rmse_w_hu": rmse(pw, tw) * (cfg.eval_window[1] - cfg.eval_window[0]),
    }


def summarize(rows: dict[str, list[torch.Tensor]]):
    out = {}
    for k, v in rows.items():
        t = torch.cat(v).float()
        out[k] = (t.mean().item(), t.std().item())
    return out


def wilcoxon(a: torch.Tensor, b: torch.Tensor):
    """Paired Wilcoxon signed-rank p-value (scipy if present, else None)."""
    try:
        from scipy.stats import wilcoxon as _w
        return float(_w(a.cpu().numpy(), b.cpu().numpy()).pvalue)
    except Exception:
        return float("nan")
