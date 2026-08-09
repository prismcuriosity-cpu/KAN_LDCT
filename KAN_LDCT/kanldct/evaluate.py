"""Full held-out evaluation: table, significance tests, figures, CSV.

Unlike v1 (which scored a single batch of 8 slices) this runs the whole
held-out patient set and reports mean +/- std per slice, plus a paired Wilcoxon
test of KAN-PGSD against the strongest baseline.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import BASELINES
from .config import Cfg, cfg_from_args
from .data import make_loaders
from .diffusion import KANPGSD, MeanPreservingSchedule
from .metrics import slice_metrics, summarize, to_window, wilcoxon
from .physics import ParallelBeamRadon

METHOD_ORDER = ["LD (input)", "RED-CNN", "EDCNN", "UNet", "KAN-PGSD",
                "KAN-PGSD+DC", "KAN-PGSD (ens)"]


def _load(cfg, dev):
    run = cfg.run_dir()
    model = KANPGSD(cfg).to(dev)
    for name in ("ckpt_physics.pt", "ckpt_pretrain.pt"):
        p = run / name
        if p.exists():
            ck = torch.load(p, map_location=dev, weights_only=False)
            model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
            print(f"[eval] loaded {name}")
            break
    else:
        raise FileNotFoundError(f"no checkpoint in {run}")
    model.eval()
    bl = {}
    for name, ctor in BASELINES.items():
        p = run / f"baseline_{name}.pt"
        if p.exists():
            net = ctor().to(dev)
            net.load_state_dict(torch.load(p, map_location=dev, weights_only=True))
            bl[name] = net.eval()
        else:
            print(f"[eval] skipping {name} (no checkpoint)")
    return model, bl


@torch.no_grad()
def run_eval(cfg: Cfg, dev, split="test"):
    run = cfg.run_dir()
    _, (tl, vl, sl) = make_loaders(cfg, verbose=False)
    loader = {"val": vl, "test": sl}[split]
    if len(loader.dataset) == 0:
        print(f"[eval] {split} split empty; using val")
        loader = vl
    model, bl = _load(cfg, dev)
    sched = MeanPreservingSchedule(cfg.n_steps, cfg.bridge_sigma, dev)
    radon = ParallelBeamRadon(cfg.img_size, cfg.n_angles, device=dev,
                              chunk=cfg.angle_chunk)

    rows: dict[str, dict[str, list]] = {}
    timing: dict[str, float] = {}
    n_done, first, unc = 0, None, None
    for bi, (ld, nd) in enumerate(loader):
        ld, nd = ld.to(dev), nd.to(dev)
        preds = {"LD (input)": ld}
        for name, net in bl.items():
            t0 = time.time()
            preds[name] = net(ld).clamp(-1, 1)
            timing[name] = timing.get(name, 0.0) + time.time() - t0
        t0 = time.time()
        preds["KAN-PGSD"] = model.sample(ld, sched, cfg, None)
        timing["KAN-PGSD"] = timing.get("KAN-PGSD", 0.0) + time.time() - t0
        if cfg.dc_step > 0:
            t0 = time.time()
            preds["KAN-PGSD+DC"] = model.sample(ld, sched, cfg, radon)
            timing["KAN-PGSD+DC"] = timing.get("KAN-PGSD+DC", 0.0) + time.time() - t0
        if bi == 0 and cfg.unc_samples > 1:
            m, s = model.sample_with_uncertainty(ld, sched, cfg, None,
                                                 cfg.unc_samples)
            preds["KAN-PGSD (ens)"] = m
            unc = s
        for name, p in preds.items():
            d = rows.setdefault(name, {})
            for k, v in slice_metrics(p, nd, cfg).items():
                d.setdefault(k, []).append(v.cpu())
        if first is None:
            first = ({k: v.cpu() for k, v in preds.items()}, nd.cpu(),
                     unc.cpu() if unc is not None else None)
        n_done += ld.size(0)
        if cfg.max_eval_slices and n_done >= cfg.max_eval_slices:
            break

    # ------------------------------------------------------------- table --
    summary = {k: summarize(v) for k, v in rows.items()}
    order = [m for m in METHOD_ORDER if m in summary] + \
            [m for m in summary if m not in METHOD_ORDER]
    hdr = (f"{'Method':<16}{'PSNR (dB)':>16}{'SSIM':>16}{'RMSE (HU)':>14}"
           f"{'win-PSNR':>12}{'win-SSIM':>12}{'ms/slice':>10}")
    lines = [f"held-out slices: {n_done}", hdr, "-" * len(hdr)]
    for m in order:
        s = summary[m]
        ms = timing.get(m, 0.0) / max(n_done, 1) * 1e3
        lines.append(f"{m:<16}{s['psnr'][0]:>9.2f}+-{s['psnr'][1]:<5.2f}"
                     f"{s['ssim'][0]:>10.4f}+-{s['ssim'][1]:<5.4f}"
                     f"{s['rmse_hu'][0]:>10.1f}+-{s['rmse_hu'][1]:<3.1f}"
                     f"{s['psnr_w'][0]:>12.2f}{s['ssim_w'][0]:>12.4f}"
                     f"{ms:>10.1f}")
    # significance vs best baseline
    base_names = [m for m in summary if m in BASELINES or m == "LD (input)"]
    if base_names and "KAN-PGSD" in rows:
        best = max(base_names, key=lambda m: summary[m]["psnr"][0])
        a = torch.cat(rows["KAN-PGSD"]["psnr"])
        b = torch.cat(rows[best]["psnr"])
        lines.append("")
        lines.append(f"KAN-PGSD vs {best}: dPSNR = {(a-b).mean():+.2f} dB, "
                     f"paired Wilcoxon p = {wilcoxon(a, b):.3g}")
    report = "\n".join(lines)
    print(report)
    (run / f"results_{split}.txt").write_text(report)

    with open(run / f"results_{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = sorted(next(iter(summary.values())).keys())
        w.writerow(["method"] + [f"{k}_{s}" for k in keys for s in ("mean", "std")])
        for m in order:
            w.writerow([m] + [f"{summary[m][k][i]:.6f}" for k in keys
                              for i in (0, 1)])
    (run / f"results_{split}.json").write_text(json.dumps(summary, indent=2))
    if first is not None:
        make_figures(cfg, *first, run, split)
    kan_figures(model, cfg, run)
    return summary


def kan_figures(model, cfg, run: Path):
    """Interpretability outputs — the reason to use a KAN at all."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = run / "figures"
    fig_dir.mkdir(exist_ok=True)
    if model.acc is not None:
        labels = ["timestep t/T", "log |A x - p|", "log sigma_t", "log TV(x)"]
        fig, ax = plt.subplots(1, 4, figsize=(14, 3), squeeze=False)
        for i, lab in enumerate(labels):
            x, y = model.acc.sweep(i)
            ax[0, i].plot(x.numpy(), y.numpy(), lw=2)
            ax[0, i].set_xlabel(lab + " (normalised)")
            ax[0, i].set_ylabel("lambda")
            ax[0, i].grid(alpha=0.3)
        fig.suptitle("KAN-ACC: learned data-consistency step size")
        fig.tight_layout()
        fig.savefig(fig_dir / "kan_acc_sweeps.png", dpi=160)
        plt.close(fig)
    if model.noise is not None:
        best = model.noise.fit_symbolic()
        fig, a = plt.subplots(figsize=(5.5, 3.5))
        a.plot(best["p"], best["y"], lw=2, label="KAN-NM (learned)")
        a.plot(best["p"], best["fit"], "--",
               label=f"{best['form']}  R2={best['r2']:.3f}")
        a.set_xlabel("line integral p")
        a.set_ylabel("log sigma^2(p)")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "kan_noise_model.png", dpi=160)
        plt.close(fig)
        print(f"[eval] KAN-NM symbolic form: {best['form']}  R2={best['r2']:.4f}"
              f"  (Poisson theory predicts log sigma^2 = p - log I0)")


# -------------------------------------------------------------- figures ---
def make_figures(cfg, preds, nd, unc, run: Path, split: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = run / "figures"
    fig_dir.mkdir(exist_ok=True)
    names = [m for m in METHOD_ORDER if m in preds]
    n = min(3, nd.size(0))
    win = lambda x: to_window(x, cfg.hu_min, cfg.hu_max, *cfg.eval_window)

    # gallery in the diagnostic window (the only fair way to look at CT)
    fig, ax = plt.subplots(n, len(names) + 1, figsize=(2.2 * (len(names) + 1), 2.4 * n),
                           squeeze=False)
    for i in range(n):
        for j, m in enumerate(names):
            ax[i, j].imshow(win(preds[m][i, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(m, fontsize=8)
        ax[i, -1].imshow(win(nd[i, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
        ax[i, -1].axis("off")
        if i == 0:
            ax[i, -1].set_title("ND (GT)", fontsize=8)
    fig.suptitle(f"Soft-tissue window {cfg.eval_window[0]:.0f}/{cfg.eval_window[1]:.0f} HU")
    fig.tight_layout()
    fig.savefig(fig_dir / f"gallery_{split}.png", dpi=160)
    plt.close(fig)

    # absolute error, HU
    span = (cfg.hu_max - cfg.hu_min) / 2
    fig, ax = plt.subplots(n, len(names), figsize=(2.3 * len(names), 2.4 * n),
                           squeeze=False)
    for i in range(n):
        for j, m in enumerate(names):
            e = (preds[m][i, 0] - nd[i, 0]).abs().numpy() * span
            im = ax[i, j].imshow(e, cmap="inferno", vmin=0, vmax=120)
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(f"|err| {m}", fontsize=8)
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.6, label="HU")
    fig.savefig(fig_dir / f"error_{split}.png", dpi=160)
    plt.close(fig)

    # radial power spectrum: over-smoothing shows up as an early roll-off
    def radial(img):
        f = np.abs(np.fft.fftshift(np.fft.fft2(img)))
        H, W = img.shape
        yy, xx = np.indices((H, W))
        r = np.hypot(yy - H // 2, xx - W // 2).astype(int)
        return np.bincount(r.ravel(), f.ravel()) / np.bincount(r.ravel())

    fig, a = plt.subplots(figsize=(7, 4))
    for m in names + ["ND (GT)"]:
        img = (nd if m == "ND (GT)" else preds[m])[0, 0].numpy()
        a.plot(radial(img)[:cfg.img_size // 2], lw=1.4, label=m)
    a.set_yscale("log")
    a.set_xlabel("radial frequency (px)")
    a.set_ylabel("|F|")
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
    a.set_title("Radial power spectrum")
    fig.tight_layout()
    fig.savefig(fig_dir / f"spectrum_{split}.png", dpi=160)
    plt.close(fig)

    # uncertainty vs error
    if unc is not None and "KAN-PGSD (ens)" in preds:
        mean = preds["KAN-PGSD (ens)"]
        fig, ax = plt.subplots(n, 3, figsize=(9, 2.5 * n), squeeze=False)
        rs = []
        for i in range(n):
            e = (mean[i, 0] - nd[i, 0]).abs().numpy() * span
            s = unc[i, 0].numpy() * span
            ax[i, 0].imshow(win(mean[i, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
            ax[i, 1].imshow(s, cmap="magma")
            ax[i, 2].imshow(e, cmap="inferno", vmin=0, vmax=120)
            for j in range(3):
                ax[i, j].axis("off")
            rs.append(np.corrcoef(s.ravel(), e.ravel())[0, 1])
        ax[0, 0].set_title("ensemble mean", fontsize=8)
        ax[0, 1].set_title("std (HU)", fontsize=8)
        ax[0, 2].set_title("|error| (HU)", fontsize=8)
        fig.suptitle(f"uncertainty vs error, mean Pearson r = {np.nanmean(rs):+.3f}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"uncertainty_{split}.png", dpi=160)
        plt.close(fig)
        print(f"[eval] uncertainty-error correlation r = {np.nanmean(rs):+.3f}")
    print(f"[eval] figures -> {fig_dir}")


def main(argv=None):
    cfg, args = cfg_from_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_eval(cfg, dev, "test")


if __name__ == "__main__":
    main()
