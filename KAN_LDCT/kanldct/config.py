"""Configuration for KAN-PGSD v2 (KAN-CoreDiff for low-dose CT).

Defaults are tuned for a single RTX 5090 (32 GB) with ~30 % of VRAM held in
reserve, paired with an 8-core / 16-thread Ryzen 7 9800X3D.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cfg:
    # ---------------------------------------------------------------- data --
    data_root: str = r"D:\datasets\ct-low-dose-reconstruction\Preprocessed_256x256\256"
    ld_dirname: str = "Quarter Dose"
    nd_dirname: str = "Full Dose"
    img_size: int = 256
    # Fixed HU window used to map raw pixel values -> [-1, 1].  The SAME
    # transform is applied to the low-dose and the normal-dose slice, which is
    # what makes the pair comparable.  Per-image percentile scaling (the v1
    # behaviour) breaks the pairing and caps every metric.
    hu_min: float = -1024.0
    hu_max: float = 3071.0
    # Source encoding of the preprocessed PNG/TIFF files.  "auto" inspects the
    # first file: 16-bit -> assumed HU + 1024 offset, 8-bit -> assumed already
    # windowed to [-1024, 3071] linearly.
    src_encoding: str = "auto"          # auto | uint16_hu_offset | uint8_linear | raw_hu
    val_patients: int = 2               # hold out N whole patients (never slices)
    test_patients: int = 1
    max_train_slices: int = 0           # 0 = use all
    augment: bool = True

    # -------------------------------------------------------------- model --
    base_channels: int = 64
    channel_mult: tuple = (1, 1, 2, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: tuple = (32, 16)
    kan_resolutions: tuple = (32, 16)   # where tokenized-KAN blocks are inserted
    time_dim: int = 256
    dropout: float = 0.0

    # ---------------------------------------------------------------- KAN --
    kan_impl: str = "rbf"               # rbf (FastKAN) | bspline
    kan_grid: int = 8
    kan_range: tuple = (-2.0, 2.0)
    kan_hidden_mult: int = 2
    use_kan_unet: bool = True
    use_kan_time: bool = True
    use_kan_acc: bool = True            # learned data-consistency step size
    use_kan_noise: bool = True          # learned projection noise model
    kan_reg_weight: float = 1e-5        # L1 on spline coefficients (sparsity)

    # ---------------------------------------------------------- diffusion --
    n_steps: int = 10                   # CoreDiff-style: 10 is enough
    bridge_sigma: float = 0.10          # stochastic bridge strength, [-1,1] units
    two_stage: bool = True              # CoreDiff stage-II error modulation
    stochastic_sampling: bool = False   # True only for uncertainty ensembles

    # ------------------------------------------------------------ physics --
    n_angles: int = 128
    dc_angles: int = 64                 # cheaper operator during training
    angle_chunk: int = 32               # angles processed per grid_sample call
    mu_water: float = 0.0192            # mm^-1 @ 70 keV
    I0: float = 1e4
    dc_every: int = 4                   # apply DC on every k-th sampling step
    dc_step: float = 0.30               # global scale on the KAN-predicted lambda
    lambda_max: float = 1.0

    # ----------------------------------------------------------- training --
    # Conservative default: effective batch 16 (8 x 2) fits comfortably under a
    # 22.4 GB cap even with two_stage=True and KAN blocks at 32x32.  If
    # nvidia-smi shows headroom, --batch_size 16 --grad_accum 1 is ~15 % faster.
    batch_size: int = 8
    phys_batch: int = 4                 # sub-batch for the Radon branch
    lr: float = 2e-4
    lr_min: float = 1e-6
    warmup_steps: int = 500
    weight_decay: float = 1e-5
    grad_accum: int = 2
    grad_clip: float = 1.0
    epochs_pretrain: int = 30
    epochs_physics: int = 5
    epochs_baseline: int = 30
    ema_decay: float = 0.9995
    num_workers: int = 8
    amp: bool = True
    channels_last: bool = True
    compile: bool = False               # torch.compile: needs triton-windows
    cuda_mem_fraction: float = 0.70     # honour the 30 % VRAM reserve
    seed: int = 0

    # --------------------------------------------------------------- eval --
    eval_window: tuple = (-160.0, 240.0)   # abdomen soft-tissue window (HU)
    eval_batch: int = 8
    unc_samples: int = 8
    max_eval_slices: int = 0            # 0 = whole held-out set

    # --------------------------------------------------------------- misc --
    out_dir: str = "runs/kan_corediff"
    log_every: int = 50
    ckpt_every: int = 1                 # epochs

    # ------------------------------------------------------------ helpers --
    def ld_root(self) -> str:
        return os.path.join(self.data_root, self.ld_dirname)

    def nd_root(self) -> str:
        return os.path.join(self.data_root, self.nd_dirname)

    def run_dir(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, path=None):
        path = Path(path or (self.run_dir() / "config.json"))
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2, default=list))
        return path


_TUPLE_FIELDS = {"channel_mult", "attn_resolutions", "kan_resolutions",
                 "kan_range", "eval_window"}


def cfg_from_args(argv=None, **overrides) -> tuple[Cfg, argparse.Namespace]:
    """Build a Cfg from CLI flags.  Every dataclass field becomes ``--field``."""
    base = Cfg(**overrides)
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--stage", default="all",
                    choices=["all", "pretrain", "physics", "baselines", "eval"])
    ap.add_argument("--resume", default="", help="checkpoint path to resume from")
    for f in dataclasses.fields(base):
        cur = getattr(base, f.name)
        if f.name in _TUPLE_FIELDS:
            ap.add_argument(f"--{f.name}", type=str, default=None,
                            help=f"comma-separated (default {cur})")
        elif isinstance(cur, bool):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in
                            ("1", "true", "yes", "y"), default=None)
        else:
            ap.add_argument(f"--{f.name}", type=type(cur), default=None)
    args = ap.parse_args(argv)
    for f in dataclasses.fields(base):
        v = getattr(args, f.name, None)
        if v is None:
            continue
        if f.name in _TUPLE_FIELDS:
            v = tuple(float(x) if "." in x else int(x) for x in v.split(","))
        setattr(base, f.name, v)
    return base, args
