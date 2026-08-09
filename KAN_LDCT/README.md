# KAN-PGSD v2 — KAN-guided cold diffusion for low-dose CT

Rebuild of `kan-pgsd-fixed-20th-may-26.ipynb`. That notebook ran without crashing
but produced a reconstruction **28 dB worse than doing nothing**:

```
Method           PSNR (dB)      SSIM      RMSE      <- v1 notebook output
LD (input)       37.08±0.40   0.928     0.0280
RED-CNN          36.79±0.50   0.927     0.0290
UNet             30.93±0.21   0.930     0.0568
KAN-PGSD          9.31±1.33   0.167     0.6926     <- the model
KAN-PGSD(4)      11.85±0.54   0.162     0.5119
FBP               8.97±0.03   0.020     0.7124
```

The clamps and `nan_to_num` guards in v1 removed the NaNs but not the reason the
numbers were bad. This version fixes the reasons. It is a package, not a
notebook, so the pieces are individually testable — `python tests/test_smoke.py`
checks every non-trivial component and must pass before you train.

---

## 1. Diagnosis — why v1 scored 9 dB

Ordered by how much damage each one did.

| # | Root cause | Where | Effect |
|---|---|---|---|
| **1** | **Unconditional prior + full-noise start.** `reconstruct()` began at `t0 = n_steps-1 = 999`, where `alpha_bar ≈ 4e-5`. `x = sqrt(a)*ld + sqrt(1-a)*noise` is **pure Gaussian noise** — the low-dose image was numerically erased. 50 DDIM steps of an *unconditional* U-Net then had to re-invent the anatomy, steered only by a `0.05`-scaled gradient. | `KANPGSD.reconstruct`, cell 14 | The model hallucinated. This alone explains 9 dB. |
| **2** | **Per-image percentile normalisation, applied separately to LD and ND.** `_to_unit` rescaled each slice by *its own* 0.5/99.5 percentiles. The pair received two different affine maps, so LD and ND no longer live in one intensity space. | `MayoLDCTDataset._to_unit`, cell 16 | Caps every method's ceiling; `hu_min`/`hu_max` were accepted as arguments and never used. |
| **3** | **Image-domain "data consistency" pulls toward the noise.** With `radon=None`, `residual = ld − x0_hat` and the loss minimises it — i.e. Phase 2 explicitly trained the model to reproduce the *noisy* low-dose slice. | `diffusion_loss`, cell 14 | Phase 2 actively undid Phase 1. |
| **4** | **Time-blind embedding.** `t_in = t / n_steps ∈ [0,1)` fed into frequencies `exp(-k·log(10000)/half)` ∈ `[1e-4, 1]`. Every argument landed in `[0, 1]`, where `sin(x) ≈ x` — all 128 frequency channels collapsed to the same near-linear ramp. | `SinusoidalTimeEmbedding`, cell 10 | The denoiser could not tell `t=10` from `t=990`. |
| **5** | **KAN inputs clamped into the knot grid.** `x.clamp(grid_lo, grid_hi)` before Cox–de Boor. Clamping is zero-gradient: every saturated unit stopped learning permanently. | `KANLinear.b_splines`, cell 6 | KAN layers were partly frozen. |
| **6** | **`KANAdaptiveConsistency` was degenerate.** λ scaled the very loss it was trained on, so the optimum is λ → 0. | cell 8 | The "adaptive" head learned to switch itself off. |
| **7** | **`KANNoiseModel` was never connected to any loss.** `fit_symbolic()` then reported `quadratic R²=0.9959` — a fit to a randomly initialised function. | cell 8 / cell 29 | The headline interpretability result was noise. |
| **8** | **Pairing was an unverified sorted zip.** The run printed `[WARN] No filename-key overlap; using sorted-zip (16628 pairs)` and proceeded anyway. | `MayoLDCTDataset.__init__`, cell 16 | Silent mis-pairing risk on the whole dataset. |
| **9** | **Slice-level split.** Last 10 % of a slice list — adjacent slices of the same patient on both sides. | cell 16 | Optimistic validation. |
| **10** | **`adjoint` was not the adjoint of `forward`.** Back-projection re-used the *same* rotation grid rather than the inverse, so `⟨Ax, s⟩ ≠ ⟨x, Aᵀs⟩`. FBP output had to be rescued by max-normalising. | `ParallelBeamRadon`, cell 12 | FBP scored 8.97 dB. |
| **11** | **Beer–Lambert on a `[-1,1]` display image.** Sinograms of a signed display image are not line integrals of attenuation. v1 patched this with `clamp(0,20)`. | `simulate_ldct`, cell 12 | Physics term was dimensionally meaningless. |
| **12** | Broken `grad_accum` (zeroed grads every step), one-batch evaluation (8 slices), `num_workers=0`, mis-wired RED-CNN skips, `channels_last` on weights only. | cells 20, 22, 33 | Slow, noisy, unfair baselines. |

---

## 2. What v2 does instead

### 2.1 Mean-preserving cold diffusion (the fix that matters)

Following **CoreDiff** (Gao et al., *IEEE TMI* 43(2):745–759, 2024), the diffusion
endpoint is the **measured low-dose image**, not Gaussian noise:

```
x_t = a_t·x0 + (1 − a_t)·y_LD + σ·sqrt(a_t(1 − a_t))·ε ,   a_t = 1 − t/T
```

* `a_0 = 1` → clean; `a_T = 0` → exactly `y_LD`.
* The noise term is a Brownian bridge: zero at both ends, so
  `E[x_t] = a_t·x0 + (1−a_t)·y_LD`. The CT number is never shifted.
* Sampling **starts from the measurement** and needs `T = 10` steps, not 1000.

Reverse step (improved cold-diffusion update):

```
x_{t−1} = x_t − D(x̂0, y, t) + D(x̂0, y, t−1) = x_t + (a_{t−1} − a_t)(x̂0 − y)
```

The network is **conditional** — it sees `cat([x_t, y_LD])` — and predicts a
**residual on top of `y`**:

```
x̂0 = y + net([x_t, y], t)
```

so an untrained model is exactly the identity. `tests/test_smoke.py` asserts
this. It is why v2 cannot score below the low-dose input the way v1 did.

**Two-stage refinement (CLEAR-Net)**: the stage-I estimate is re-degraded one
step and refined, with the timestep embedding modulated by
`(β, γ) = F_φ(x̂0, y)`. Loss `= ‖x̂0 − x0‖² + ‖x̂0' − x0‖²`.

### 2.2 Where the KAN actually goes

Three concrete roles, each with its own gradient — not decoration.

**(a) Tokenized KAN blocks in the U-Net**, following **U-KAN**
(Li et al., *AAAI* 2025). The KAN acts on the channel vector of each spatial
token at the two lowest resolutions (32×32, 16×16); a depth-wise 3×3 conv
between the two KAN layers restores the spatial inductive bias:

```
GroupNorm → KANLinear(C→2C) → DWConv → KANLinear(2C→C) → γ·residual   (γ zero-init)
```

**(b) `KANAdaptiveConsistency`** — the step size `λ(t, ‖Ax̂−p‖, σ_t, TV(x̂))` for
the physics correction. v1's version was degenerate. Here λ is trained by
**differentiating through one data-consistency step and scoring the corrected
image against the ground truth**:

```
x̂0_dc = x̂0 − dc_step · λ · ĝ ,   ĝ = A^T[w(Ax̂0 − p_LD)] / ‖·‖
L_acc = ‖x̂0_dc − x0‖²
```

λ → 0 is now a *bad* solution whenever the correction helps, so the head learns
a real schedule. `dc_gain` (MSE before − after) is printed every log step, so
whether the physics term earns its place is measured, never assumed.

**(c) `KANNoiseModel`** — learns `log σ²(p)` in the projection domain via a
heteroscedastic Gaussian NLL on the projection residual, and its output *is* the
PWLS statistical weight `w = 1/σ²`. Poisson statistics predict
`log σ²(p) = p − log I₀`; `fit_symbolic()` now fits candidate closed forms to a
head that was actually trained, so `R²` means something.

**Basis**: Gaussian RBF (**FastKAN**, Li 2024) by default — no Cox–de Boor
recursion, no division, ~3× faster than B-splines. Both bases put a
`LayerNorm` in front, which *keeps* inputs inside the grid while staying
differentiable — the correct fix for v1's clamp. Basis evaluation is forced to
fp32 even under autocast (bf16's 8 mantissa bits are not enough).
`--kan_impl bspline` runs the B-spline ablation.

### 2.3 Physics that is dimensionally coherent

* image → HU → **linear attenuation** `μ = μ_water(1 + HU/1000)`, non-negative,
  before any projection.
* `adjoint` is the **exact transpose** of `forward`, computed as the projector's
  vector-Jacobian product. The smoke test asserts `⟨Ax,s⟩ = ⟨x,Aᵀs⟩` to 1e-4
  (measured: 1.1e-6).
* `fbp` = zero-padded Ram-Lak × Hann, correct `π/A` measure. No max-normalisation
  hack.
* Projector is vectorised over angles in chunks — v1 ran one `grid_sample` per
  angle per sample (180 × B calls per FBP).

> **Honest scope note.** The Mayo *preprocessed image* release ships
> reconstructed slices, not raw projections. The consistency term therefore
> compares the re-projection of the estimate against the re-projection of the
> measured LD slice. That is a statistically weighted (PWLS) regulariser, **not**
> true measurement fidelity — it cannot recover information the LD
> reconstruction already destroyed. Expect it to be worth a few tenths of a dB,
> not a headline. Run `--dc_step 0` for the ablation. If you want a genuine
> sinogram-domain claim, use the **LDCT-and-Projection-Data** release (TCIA), which
> ships the raw projections; `physics.simulate_ldct_sino` is the matching
> Poisson+Gaussian forward model.

### 2.4 Data and evaluation

* One **fixed, shared** HU→[−1,1] map for both images of a pair.
* Pairing tries four structural keys (relative path → parent+stem → digit tuple →
  stem) and then **verifies by correlation** on a random sample. Anything under
  `r = 0.90` prints a loud warning.
* **Patient-level** train/val/test split.
* Whole held-out set evaluated (not one batch), per-slice metrics, mean ± std,
  paired **Wilcoxon** test against the strongest baseline.
* Two protocols: full range, and the abdomen soft-tissue window
  `[−160, 240] HU` that the LDCT literature reports.
* SSIM is implemented in torch — no scikit-image, runs on GPU.
* Baselines under an identical budget: **RED-CNN** (correctly wired),
  **EDCNN**, **U-Net**, and **LD (input)** — the identity row that any honest
  LDCT table must contain.

---

## 3. Setup

### 3.1 PyTorch for the RTX 5090

The 5090 is Blackwell **sm_120**. Stock PyPI wheels do not carry that
architecture and fail with `no kernel image is available for execution`.

```bash
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
```

You want `(12, 0)`. Python 3.11–3.13 is the safest range.

### 3.2 Dataset

```bash
pip install kaggle
kaggle datasets download -d andrewmvd/ct-low-dose-reconstruction -p D:\datasets
tar -xf D:\datasets\ct-low-dose-reconstruction.zip -C D:\datasets\ct-low-dose-reconstruction
```

Point `--data_root` at the folder that directly contains `Full Dose` and
`Quarter Dose`, e.g.

```
D:\datasets\ct-low-dose-reconstruction\Preprocessed_256x256\256
  ├─ Full Dose\...
  └─ Quarter Dose\...
```

If your folders are named differently, pass `--ld_dirname` / `--nd_dirname`.
The first thing training prints is the pairing check — **do not proceed if it
says SUSPECT.**

### 3.3 Hardware profile (5090 32 GB @ 70 %, Ryzen 7 9800X3D)

| Setting | Value | Why |
|---|---|---|
| `--cuda_mem_fraction` | `0.70` | hard cap at 22.4 GB, honouring your 30 % reserve |
| `--batch_size` / `--grad_accum` | `8` / `2` | effective batch 16; conservative so the first run cannot OOM |
| `--num_workers` | `8` | 8 physical cores; PNG decode is the bottleneck |
| `--amp` | `true` | bf16 (sm_120 native); KAN bases stay fp32 |
| `--channels_last` | `true` | applied to weights *and* inputs |
| `--compile` | `false` | `torch.compile` needs `triton-windows`; enable only after a clean eager run |

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
```

Watch `nvidia-smi` during the first epoch:

* **headroom left** → `--batch_size 16 --grad_accum 1` (about 15 % faster), then
  `--base_channels 96` if there is still room.
* **OOM** → in this order: `--kan_resolutions 16` (drops the 32×32 KAN blocks,
  which hold the largest basis tensors: 16 384 tokens × C × 8 grid points),
  then `--two_stage false`, then `--batch_size 4 --grad_accum 4`.

The 32×32 KAN blocks are the memory hot spot, not the attention.

---

## 4. Run it

```powershell
python tests\test_smoke.py
.\scripts\run_all.ps1 -DataRoot "D:\datasets\ct-low-dose-reconstruction\Preprocessed_256x256\256"
```

Or stage by stage:

```bash
python -m kanldct.train --stage pretrain  --data_root <ROOT> --out_dir runs/v2
python -m kanldct.train --stage physics   --data_root <ROOT> --out_dir runs/v2
python -m kanldct.train --stage baselines --data_root <ROOT> --out_dir runs/v2
python -m kanldct.evaluate                --data_root <ROOT> --out_dir runs/v2
```

Resume: `--resume runs/v2/ckpt_pretrain.pt`.
Every dataclass field in `kanldct/config.py` is a `--flag`.

**Rough wall-clock on the 5090** (~15 k train slices, 256², batch 16):
pretrain ≈ 3–5 min/epoch → ~2–3 h for 30 epochs; physics ≈ 5 epochs (the Radon
branch dominates); baselines ≈ 45 min total; evaluation minutes.
Inference is **10 network passes/slice**, ~0.1–0.2 s/slice — v1 needed 50 passes
*plus* 4 more full reconstructions for its uncertainty map.

Outputs in `--out_dir`:

```
config.json  ckpt_pretrain.pt  ckpt_physics.pt  baseline_*.pt
history_*.json  results_test.{txt,csv,json}
figures/gallery_test.png  error_test.png  spectrum_test.png
        uncertainty_test.png  kan_acc_sweeps.png  kan_noise_model.png
```

---

## 5. Ablations

```powershell
.\scripts\ablation.ps1 -DataRoot <ROOT> -Epochs 15
```

| Flag | Question it answers |
|---|---|
| `--use_kan_unet false` | Do the tokenized KAN blocks beat plain ResBlocks at equal params? |
| `--use_kan_time false` | Is the KAN time-MLP worth it over a plain MLP? |
| `--kan_impl bspline` | RBF vs B-spline basis |
| `--two_stage false` | Value of CLEAR-Net stage II |
| `--dc_step 0` | Value of the PWLS consistency term |
| `--use_kan_acc false` | Learned λ vs fixed λ = 0.5 |
| `--n_steps 1 / 10 / 20` | Sampling-step budget |

For a fair KAN-vs-MLP claim, match parameter counts — a KAN layer with `G` grid
points carries roughly `G+1` times the weights of the `nn.Linear` it replaces.
Compensate with `--base_channels`, and report the counts.

---

## 6. What to expect, and how to compare

**Sanity floor.** v2's residual parameterisation makes the untrained model the
identity, so `KAN-PGSD ≥ LD (input)` from step 0. If a trained run is *below*
the LD row, something is wrong with the data, not the model — check the pairing
correlation first.

**Do not judge at 10 epochs.** The supervised CNNs converge in a handful of
epochs; the diffusion model does not, and it is *behind* them early. Measured on
the synthetic end-to-end check shipped with this repo (same data, same budget
for all methods):

| epochs | LD input | RED-CNN | KAN-PGSD |
|---|---|---|---|
| 8 | 25.85 | **31.26** | 27.09 |
| 25 | 25.85 | 32.72 | **33.79** (+1.08 dB, p = 5e-4) |

The crossover is real. Run the full `--epochs_pretrain 30` before drawing any
conclusion. (Those are toy-phantom numbers used only to verify the pipeline —
they say nothing about Mayo performance, only about the convergence shape.)

**Reference numbers from the literature** (Mayo 2016 abdominal, 25 % dose,
original DICOM with the standard `[-1024, 3072]` HU preprocessing — from the
CoreDiff paper's Table):

| Method | PSNR (dB) | SSIM | RMSE (HU) |
|---|---|---|---|
| RED-CNN | 39.29 ± 1.53 | 0.9599 | 22.1 |
| WGAN-VGG | 40.12 ± 0.98 | 0.9419 | 19.9 |
| DU-GAN | 41.50 ± 1.22 | 0.9591 | 17.0 |
| IDDPM (50 steps) | 41.49 ± 1.18 | 0.9582 | 17.0 |
| **CoreDiff (10 steps)** | **43.92 ± 1.33** | **0.9744** | **12.9** |

> These are **not** directly comparable to a run on the Kaggle *preprocessed
> image* release: different source encoding, different HU mapping, different
> split, and 8-bit quantisation in some folders. Use them to sanity-check the
> *ordering* and the *size* of the gaps, and always report your own `LD (input)`
> row so a reader can normalise. If you need numbers that are comparable to the
> table, run on the original AAPM Mayo 2016 DICOMs with the same split as the
> paper you cite.

**Report these, not just PSNR:** the soft-tissue-window metrics, the radial
power spectrum (over-smoothing shows as an early roll-off — the standard failure
of MSE-trained denoisers), the LD-minus-output residual (structure in it means
anatomy was removed), and the uncertainty↔error correlation.

---

## 7. Layout

```
kanldct/config.py     dataclass config + CLI (every field is a --flag)
kanldct/kan.py        RBF and B-spline KAN layers, tokenized KAN block
kanldct/models.py     conditional KAN-UNet, error modulator, KAN-ACC, KAN-NM
kanldct/diffusion.py  mean-preserving schedule, training losses, sampler
kanldct/physics.py    vectorised Radon / exact adjoint / FBP / PWLS
kanldct/data.py       pairing + verification, patient split, shared HU map
kanldct/baselines.py  RED-CNN, EDCNN, U-Net
kanldct/metrics.py    torch SSIM/PSNR/RMSE, windowing, Wilcoxon
kanldct/train.py      three stages, EMA, cosine LR, AMP, checkpoints
kanldct/evaluate.py   held-out table, significance, all figures
tests/test_smoke.py   runnable self-check — run this first
```

---

## 8. Known limitations

1. The projection-consistency term is a re-projection regulariser, not
   measurement fidelity (§2.3). Do not describe it as sinogram-domain data
   consistency in a paper written on this dataset.
2. `hu_min`/`hu_max` for 8-bit source folders are an *assumed* window; HU-unit
   RMSE inherits that assumption. 16-bit folders (HU + 1024) are exact.
3. The uncertainty map is an ensemble spread over the bridge noise. It is a
   useful correlate of error, not a calibrated posterior — do not report coverage
   intervals from it without a calibration study.
4. `torch.compile` is untested on Windows here; left off by default.
5. KAN layers have more parameters per unit than the `nn.Linear` they replace.
   Any "KAN wins" claim needs the parameter-matched ablation from §5.

## 9. References

- Gao et al., *CoreDiff: Contextual Error-Modulated Generalized Diffusion Model
  for Low-Dose CT Denoising and Generalization*, IEEE TMI 43(2), 2024.
  [arXiv:2304.01814](https://arxiv.org/abs/2304.01814)
- Li et al., *U-KAN Makes Strong Backbone for Medical Image Segmentation and
  Generation*, AAAI 2025. [arXiv:2406.02918](https://arxiv.org/pdf/2406.02918) ·
  [code](https://github.com/CUHK-AIM-Group/U-KAN)
- Li, *Kolmogorov-Arnold Networks are Radial Basis Function Networks* (FastKAN).
  [arXiv:2405.06721](https://arxiv.org/pdf/2405.06721) ·
  [code](https://github.com/ZiyaoLi/fast-kan)
- Liu et al., *KAN: Kolmogorov–Arnold Networks*, ICLR 2025.
  [paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/afaed89642ea100935e39d39a4da602c-Paper-Conference.pdf)
- Chen et al., *Low-Dose CT with a Residual Encoder-Decoder CNN (RED-CNN)*,
  IEEE TMI 36(12), 2017.
- Sauer & Bouman, *A local update strategy for iterative reconstruction from
  projections* (PWLS statistical weighting), IEEE TSP 41(2), 1993.
- McCollough et al., *Low-dose CT image and projection dataset*, Med. Phys. 2021.
  [PMC7985836](https://pmc.ncbi.nlm.nih.gov/articles/PMC7985836/)
- Dataset: [CT Low Dose Reconstruction (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/ct-low-dose-reconstruction)
