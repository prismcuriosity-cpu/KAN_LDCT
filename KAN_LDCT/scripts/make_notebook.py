"""Regenerate KAN_LDCT_v2.ipynb (thin driver over the kanldct package)."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s.splitlines(True)}


CELLS = [
md(r"""# KAN-PGSD v2 - KAN-guided cold diffusion for low-dose CT

Thin driver over the `kanldct` package. All logic lives in the package so it is
testable; this notebook only orchestrates and shows pictures.

Read `README.md` first - it lists the twelve defects in the v1 notebook and what
replaced each one.

**Run the self-check cell before anything else. If it fails, do not train.**
"""),

md("## 0. Environment + self-check"),
code(r'''import os, sys, subprocess
sys.path.insert(0, os.path.abspath("."))          # run from the repo root
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  sm_{p.major}{p.minor}  {p.total_memory/2**30:.1f} GiB")
    if (p.major, p.minor) >= (12, 0):
        assert (torch.version.cuda or "0") >= "12.8", (
            "RTX 50-series is sm_120: pip install torch "
            "--index-url https://download.pytorch.org/whl/cu128")
else:
    print("!! no CUDA - this will be unusably slow")

r = subprocess.run([sys.executable, "tests/test_smoke.py"], capture_output=True,
                   text=True)
print(r.stdout or r.stderr)
assert r.returncode == 0, "self-check failed"'''),

md(r"""## 1. Configuration

Point `DATA_ROOT` at the folder that directly contains `Full Dose` and
`Quarter Dose`. Everything else is tuned for a 32 GB RTX 5090 at 70 % VRAM."""),
code(r'''from kanldct.config import Cfg

DATA_ROOT = r"D:\datasets\ct-low-dose-reconstruction\Preprocessed_256x256\256"

cfg = Cfg(
    data_root=DATA_ROOT,
    out_dir="runs/nb_v2",
    img_size=256,
    batch_size=16,
    num_workers=8,
    cuda_mem_fraction=0.70,
    epochs_pretrain=30,
    epochs_physics=5,
    epochs_baseline=30,
)
cfg.save()
print(cfg)'''),

md(r"""## 2. Data - pairing check first

The v1 notebook printed `No filename-key overlap; using sorted-zip (16628 pairs)`
and trained anyway. `verify_pairing` reports the mean Pearson r between paired
LD/ND slices; correct Mayo quarter-dose pairs sit above 0.98.
**If this prints SUSPECT, stop and fix the folder layout.**"""),
code(r'''from kanldct.train import setup
from kanldct.data import make_loaders

dev, amp_dtype = setup(cfg)
(train_ds, val_ds, test_ds), (train_dl, val_dl, test_dl) = make_loaders(cfg)'''),

code(r'''import matplotlib.pyplot as plt
from kanldct.metrics import to_window, psnr

fig, ax = plt.subplots(2, 4, figsize=(12, 6))
for j in range(4):
    ld, nd = train_ds[j * 37 % len(train_ds)]
    for i, (img, name) in enumerate(((ld, "LD"), (nd, "ND"))):
        ax[i, j].imshow(to_window(img[0], cfg.hu_min, cfg.hu_max, *cfg.eval_window),
                        cmap="gray", vmin=0, vmax=1)
        ax[i, j].set_title(f"{name} #{j}", fontsize=9)
        ax[i, j].axis("off")
    print(f"slice {j}: LD-vs-ND PSNR = {psnr(ld[None], nd[None]).item():.2f} dB")
plt.suptitle(f"soft-tissue window {cfg.eval_window}")
plt.tight_layout(); plt.show()'''),

md("## 3. Model"),
code(r'''from kanldct.diffusion import KANPGSD, MeanPreservingSchedule
from kanldct.models import count_params
from kanldct.physics import ParallelBeamRadon

model = KANPGSD(cfg).to(dev)
model.net = model.net.to(memory_format=torch.channels_last)
sched = MeanPreservingSchedule(cfg.n_steps, cfg.bridge_sigma, dev)
radon = ParallelBeamRadon(cfg.img_size, cfg.dc_angles, device=dev,
                          chunk=cfg.angle_chunk)

print(f"KAN-UNet   {count_params(model.net)/1e6:7.2f} M")
if model.modulator: print(f"modulator  {count_params(model.modulator)/1e6:7.2f} M")
if model.acc:       print(f"KAN-ACC    {count_params(model.acc):7d}")
if model.noise:     print(f"KAN-NM     {count_params(model.noise):7d}")

# sanity: an untrained model is the identity on the LD input, so it can never
# score below it -- the failure mode that gave v1 9.31 dB is structurally gone
ld, nd = next(iter(val_dl))
with torch.no_grad():
    out = model.sample(ld[:2].to(dev), sched, cfg, None)
print("untrained |out - ld| max =", (out.cpu() - ld[:2]).abs().max().item())'''),

md(r"""## 4. Stage 1 - restoration pre-training

Where essentially all of the quality comes from. Validation PSNR is printed each
epoch and starts at the LD-input PSNR, never below it."""),
code(r'''from kanldct.train import train_diffusion

model, ema, hist1 = train_diffusion(cfg, dev, amp_dtype,
                                    (train_dl, val_dl, test_dl), "pretrain", model)'''),

code(r'''fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot([h["L_stage1"] for h in hist1], label="stage I")
if hist1 and "L_stage2" in hist1[0]:
    ax[0].plot([h["L_stage2"] for h in hist1], label="stage II")
ax[0].set_yscale("log"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title("restoration MSE")
ax[1].plot([h["lr"] for h in hist1]); ax[1].set_title("lr"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()'''),

md(r"""## 5. Stage 2 - physics / KAN heads

Trains only `KAN-NM` (projection noise model) and `KAN-ACC` (data-consistency
step size); the restoration network is untouched. Watch `dc_gain` - it is
MSE-before minus MSE-after the correction, so **positive means the physics term
is actually helping**. If it stays negative on your data, the honest move is
`dc_step=0` and to report that."""),
code(r'''model, ema, hist2 = train_diffusion(cfg, dev, amp_dtype,
                                    (train_dl, val_dl, test_dl), "physics",
                                    model, ema, radon)
if hist2:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3))
    for a, k, t in zip(ax, ("L_nm", "lam", "dc_gain"),
                       ("KAN-NM NLL", "lambda", "DC gain (MSE before - after)")):
        a.plot([h[k] for h in hist2 if k in h]); a.set_title(t); a.grid(alpha=.3)
    ax[2].axhline(0, color="r", lw=.8)
    plt.tight_layout(); plt.show()'''),

md("## 6. Baselines - RED-CNN / EDCNN / U-Net, identical budget"),
code(r'''from kanldct.train import train_baselines
nets = train_baselines(cfg, dev, amp_dtype, (train_dl, val_dl, test_dl))'''),

md(r"""## 7. Held-out evaluation

Whole held-out patient, per-slice mean +- std, paired Wilcoxon against the
strongest baseline, both metric protocols, and every figure."""),
code(r'''from kanldct.evaluate import run_eval
summary = run_eval(cfg, dev, "test")'''),

code(r'''from IPython.display import Image, display
from pathlib import Path
for f in ["gallery_test.png", "error_test.png", "spectrum_test.png",
          "uncertainty_test.png", "kan_acc_sweeps.png", "kan_noise_model.png"]:
    p = Path(cfg.out_dir) / "figures" / f
    if p.exists():
        print(f); display(Image(filename=str(p)))'''),

md(r"""## 8. Reading the results

* `LD (input)` is the floor. Every method must beat it; v1's did not (9.31 vs 37.08 dB).
* `win-PSNR` / `win-SSIM` use the abdomen soft-tissue window `[-160, 240] HU` -
  the protocol the LDCT literature reports.
* The radial spectrum shows over-smoothing as an early high-frequency roll-off.
  A method can win on PSNR and lose here; report both.
* `KAN-PGSD+DC` vs `KAN-PGSD` isolates the projection-consistency term. If the
  gap is inside the noise, say so - see the scope note in README section 2.3.
* `kan_acc_sweeps.png` and `kan_noise_model.png` are the interpretability
  outputs. `KAN-NM` should approach `log sigma^2(p) = p - log I0` where Poisson
  statistics dominate.

Ablations: `scripts/ablation.ps1`, or set the flags on `Cfg` and re-run."""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = ROOT / "KAN_LDCT_v2.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")
