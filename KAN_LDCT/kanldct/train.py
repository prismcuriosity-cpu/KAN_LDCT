"""Training entry point.

    python -m kanldct.train --stage all --data_root "D:\\...\\256"

Stages
------
pretrain   restoration cold-diffusion (KAN-UNet), no physics.  This is where
           nearly all of the quality comes from.
physics    freezes nothing but only optimises the two auxiliary KAN heads
           (KAN-NM noise model, KAN-ACC step size) against the PWLS operator.
baselines  RED-CNN / EDCNN / U-Net under an identical budget.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .baselines import BASELINES
from .config import Cfg, cfg_from_args
from .data import make_loaders
from .diffusion import KANPGSD, MeanPreservingSchedule
from .kan import kan_regularization
from .metrics import slice_metrics, summarize
from .models import count_params
from .physics import ParallelBeamRadon


# ------------------------------------------------------------------ setup --
def setup(cfg: Cfg):
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        if 0 < cfg.cuda_mem_fraction < 1:
            torch.cuda.set_per_process_memory_fraction(cfg.cuda_mem_fraction, 0)
        print(f"[env] {p.name} sm_{p.major}{p.minor} {p.total_memory/2**30:.1f} GiB "
              f"| cap {cfg.cuda_mem_fraction:.0%} = "
              f"{p.total_memory*cfg.cuda_mem_fraction/2**30:.1f} GiB | torch {torch.__version__}")
        amp_dtype = torch.bfloat16 if p.major >= 8 else torch.float16
    else:
        print(f"[env] CPU | torch {torch.__version__}")
        amp_dtype = torch.bfloat16
    return dev, amp_dtype


class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1.0 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()


def cosine_lr(step, total, base, min_lr, warmup):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * min(p, 1.0)))


def autocast(dev, dtype, on):
    return torch.autocast(device_type=dev.type, dtype=dtype,
                          enabled=on and dev.type == "cuda")


# --------------------------------------------------------------- validate --
@torch.no_grad()
def quick_val(model, sched, cfg, loader, dev, radon=None, max_batches=4):
    was = model.training
    model.eval()
    rows: dict[str, list] = {}
    for i, (ld, nd) in enumerate(loader):
        if i >= max_batches:
            break
        ld, nd = ld.to(dev), nd.to(dev)
        out = model.sample(ld, sched, cfg, radon)
        for k, v in slice_metrics(out, nd, cfg).items():
            rows.setdefault(k, []).append(v.cpu())
    model.train(was)
    return summarize(rows)


# ------------------------------------------------------------- diffusion ---
def train_diffusion(cfg, dev, amp_dtype, loaders, stage="pretrain",
                    model=None, ema=None, radon=None):
    tl, vl, _ = loaders
    if model is None:
        model = KANPGSD(cfg).to(dev)
        if cfg.channels_last:
            model.net = model.net.to(memory_format=torch.channels_last)
        print(f"[model] KAN-UNet {count_params(model.net)/1e6:.2f} M | "
              f"modulator {count_params(model.modulator)/1e3 if model.modulator else 0:.1f} k | "
              f"KAN-ACC {count_params(model.acc) if model.acc else 0} | "
              f"KAN-NM {count_params(model.noise) if model.noise else 0}")
    sched = MeanPreservingSchedule(cfg.n_steps, cfg.bridge_sigma, dev)
    if ema is None:
        ema = EMA(model, cfg.ema_decay)

    if stage == "physics":
        params = [p for m in (model.acc, model.noise) if m is not None
                  for p in m.parameters()]
        epochs, lr = cfg.epochs_physics, cfg.lr
        if not params:
            print("[physics] both KAN heads disabled — nothing to train")
            return model, ema, []
    else:
        params = list(model.parameters())
        epochs, lr = cfg.epochs_pretrain, cfg.lr

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.99))
    total = max(epochs * len(tl) // cfg.grad_accum, 1)
    scaler = torch.amp.GradScaler(dev.type, enabled=(amp_dtype == torch.float16
                                                     and dev.type == "cuda"))
    hist, gstep, t0 = [], 0, time.time()
    run = cfg.run_dir()

    for ep in range(epochs):
        model.train()
        pbar = tqdm(tl, desc=f"{stage} {ep+1}/{epochs}", leave=False, dynamic_ncols=True)
        for it, (ld, nd) in enumerate(pbar):
            ld = ld.to(dev, non_blocking=True)
            nd = nd.to(dev, non_blocking=True)
            if cfg.channels_last:
                ld, nd = ld.contiguous(memory_format=torch.channels_last), \
                         nd.contiguous(memory_format=torch.channels_last)

            for g in opt.param_groups:
                g["lr"] = cosine_lr(gstep, total, lr, cfg.lr_min, cfg.warmup_steps)

            if stage == "physics":
                # The Radon branch stays in fp32.  A sinogram is a sum over 256
                # rows; in bf16 (8 mantissa bits) that accumulates several
                # percent of error, which is larger than the residual the PWLS
                # term is trying to measure.
                with torch.no_grad(), autocast(dev, amp_dtype, cfg.amp):
                    t = torch.randint(1, sched.T + 1, (ld.size(0),), device=dev)
                    x_t = sched.degrade(nd, ld, t)
                    x0_hat = model.predict_x0(x_t, ld, t)
                n = min(cfg.phys_batch, ld.size(0))
                d = model.physics_losses(nd[:n].float(), ld[:n].float(),
                                         x0_hat[:n].float(), t[:n], sched,
                                         radon, cfg)
                loss = sum(v for k, v in d.items() if k.startswith("L_"))
                logs = {k: float(v.detach()) for k, v in d.items()}
            else:
                with autocast(dev, amp_dtype, cfg.amp):
                    d, x0_hat, t = model.restoration_loss(nd, ld, sched)
                    reg = kan_regularization(model.net)
                    loss = d["L_total"] + cfg.kan_reg_weight * reg
                logs = {k: float(v.detach()) for k, v in d.items()}
                logs["reg"] = float(reg.detach())

            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                print(f"[warn] non-finite loss at step {gstep}; batch skipped")
                continue

            scaler.scale(loss / cfg.grad_accum).backward()
            if (it + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if stage != "physics":
                    ema.update(model)
                gstep += 1

            if it % cfg.log_every == 0:
                logs.update(ep=ep, step=gstep, lr=opt.param_groups[0]["lr"],
                            sec=time.time() - t0)
                hist.append(logs)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()
                                  if k.startswith(("L_", "lam", "dc_"))})

        if stage == "physics":
            # the physics stage optimises the aux heads directly (no EMA), so
            # mirror them into the shadow before it is validated or saved.
            for src, dst in ((model.acc, ema.shadow.acc),
                             (model.noise, ema.shadow.noise)):
                if src is not None:
                    dst.load_state_dict(src.state_dict())

        if (ep + 1) % cfg.ckpt_every == 0 or ep == epochs - 1:
            m = quick_val(ema.shadow, sched, cfg, vl, dev,
                          radon if stage == "physics" else None)
            print(f"[{stage}] epoch {ep+1}: PSNR {m['psnr'][0]:.2f}+-{m['psnr'][1]:.2f} "
                  f"| SSIM {m['ssim'][0]:.4f} | RMSE {m['rmse_hu'][0]:.1f} HU "
                  f"| win-PSNR {m['psnr_w'][0]:.2f}")
            torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                        "ema": ema.state_dict(), "epoch": ep, "stage": stage},
                       run / f"ckpt_{stage}.pt")
            (run / f"history_{stage}.json").write_text(json.dumps(hist, indent=1))
    return model, ema, hist


# ------------------------------------------------------------- baselines ---
def train_baselines(cfg, dev, amp_dtype, loaders):
    tl, vl, _ = loaders
    run = cfg.run_dir()
    out = {}
    for name, ctor in BASELINES.items():
        net = ctor().to(dev)
        if cfg.channels_last:
            net = net.to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        total = max(cfg.epochs_baseline * len(tl), 1)
        scaler = torch.amp.GradScaler(dev.type,
                                      enabled=(amp_dtype == torch.float16
                                               and dev.type == "cuda"))
        print(f"[baseline] {name}: {count_params(net)/1e6:.2f} M params")
        step = 0
        for ep in range(cfg.epochs_baseline):
            net.train()
            pbar = tqdm(tl, desc=f"{name} {ep+1}/{cfg.epochs_baseline}",
                        leave=False, dynamic_ncols=True)
            for ld, nd in pbar:
                ld, nd = ld.to(dev, non_blocking=True), nd.to(dev, non_blocking=True)
                for g in opt.param_groups:
                    g["lr"] = cosine_lr(step, total, cfg.lr, cfg.lr_min,
                                        cfg.warmup_steps)
                with autocast(dev, amp_dtype, cfg.amp):
                    loss = F.mse_loss(net(ld), nd)
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                pbar.set_postfix(mse=f"{loss.item():.5f}")
        torch.save(net.state_dict(), run / f"baseline_{name}.pt")
        out[name] = net
    return out


# ------------------------------------------------------------------ main --
def main(argv=None):
    cfg, args = cfg_from_args(argv)
    dev, amp_dtype = setup(cfg)
    cfg.save()
    _, loaders = make_loaders(cfg)
    radon = ParallelBeamRadon(cfg.img_size, cfg.dc_angles, device=dev,
                              chunk=cfg.angle_chunk)
    model = ema = None
    run = cfg.run_dir()

    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model = KANPGSD(cfg).to(dev)
        model.load_state_dict(ck["model"])
        ema = EMA(model, cfg.ema_decay)
        ema.shadow.load_state_dict(ck["ema"])
        print(f"[resume] {args.resume} (stage={ck.get('stage')} epoch={ck.get('epoch')})")

    if args.stage in ("all", "pretrain"):
        model, ema, _ = train_diffusion(cfg, dev, amp_dtype, loaders, "pretrain",
                                        model, ema)
    if args.stage in ("all", "physics"):
        if model is None:
            ck = torch.load(run / "ckpt_pretrain.pt", map_location=dev,
                            weights_only=False)
            model = KANPGSD(cfg).to(dev)
            model.load_state_dict(ck["model"])
            ema = EMA(model, cfg.ema_decay)
            ema.shadow.load_state_dict(ck["ema"])
        train_diffusion(cfg, dev, amp_dtype, loaders, "physics", model, ema, radon)
    if args.stage in ("all", "baselines"):
        train_baselines(cfg, dev, amp_dtype, loaders)
    if args.stage in ("all", "eval"):
        from .evaluate import run_eval
        run_eval(cfg, dev)
    print("[done]", run.resolve())


if __name__ == "__main__":
    main()
