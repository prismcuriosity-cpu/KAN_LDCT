"""Runnable self-check.  `python tests/test_smoke.py` — no pytest needed.

Covers every non-trivial piece of logic: layer shapes, gradient flow, the
Radon/FBP adjoint identity, the mean-preserving schedule endpoints, the fact
that an untrained model is the identity (so it can never score below the
low-dose input), and that training on a toy problem actually improves PSNR.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kanldct.baselines import BASELINES
from kanldct.config import Cfg
from kanldct.data import MayoPairs, build_pairs, raw_to_norm, verify_pairing
from kanldct.diffusion import KANPGSD, MeanPreservingSchedule
from kanldct.kan import RBFKANLinear, BSplineKANLinear, TokenKANBlock, kan_regularization
from kanldct.metrics import psnr, slice_metrics, ssim
from kanldct.models import KANUNet, count_params
from kanldct.physics import ParallelBeamRadon, norm_to_mu, poisson_weights

torch.manual_seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tiny_cfg(**kw):
    c = Cfg(img_size=32, base_channels=8, channel_mult=(1, 2), num_res_blocks=1,
            attn_resolutions=(16,), kan_resolutions=(16,), time_dim=32,
            kan_grid=4, n_steps=4, n_angles=16, dc_angles=16, angle_chunk=8,
            batch_size=2, phys_batch=2)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def t_kan_layers():
    for cls in (RBFKANLinear, BSplineKANLinear):
        m = cls(6, 5, num_grids=4).to(DEV)
        x = (torch.randn(3, 7, 6, device=DEV) * 50).requires_grad_(True)  # out of grid
        y = m(x)
        assert y.shape == (3, 7, 5), y.shape
        assert torch.isfinite(y).all(), f"{cls.__name__} produced non-finite output"
        y.sum().backward()
        g = m.spline.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
            f"{cls.__name__}: dead spline gradient (the v1 clamp bug)"
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0, \
            f"{cls.__name__}: no gradient reaches the input"
        assert m.regularization_loss().item() >= 0
    blk = TokenKANBlock(8, "rbf", 4).to(DEV)
    z = torch.randn(2, 8, 16, 16, device=DEV)
    out = blk(z)
    assert out.shape == z.shape
    assert torch.allclose(out, z, atol=1e-6), "zero-init residual should start as identity"
    print("  kan layers            OK")


def t_unet_shapes():
    cfg = tiny_cfg()
    net = KANUNet(in_channels=2, out_channels=1, base_channels=cfg.base_channels,
                  channel_mult=cfg.channel_mult, num_res_blocks=cfg.num_res_blocks,
                  img_size=cfg.img_size, attn_resolutions=cfg.attn_resolutions,
                  kan_resolutions=cfg.kan_resolutions, time_dim=cfg.time_dim,
                  kan_grid=cfg.kan_grid).to(DEV)
    x = torch.randn(2, 2, 32, 32, device=DEV)
    t = torch.randint(0, 4, (2,), device=DEV)
    y = net(x, t)
    assert y.shape == (2, 1, 32, 32), y.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    n_none = sum(1 for p in net.parameters() if p.requires_grad and p.grad is None)
    assert n_none == 0, f"{n_none} parameters received no gradient"
    # the timestep must actually change the output (v1's embedding was time-blind).
    # zero-initialised output convs are deliberate (identity at init), so give
    # them a value before probing the architecture's time sensitivity.
    with torch.no_grad():
        for p in net.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.05)
        a = net(x, torch.zeros(2, dtype=torch.long, device=DEV))
        b = net(x, torch.full((2,), 3, dtype=torch.long, device=DEV))
    assert (a - b).abs().mean() > 1e-6, "network is time-blind"
    print(f"  unet shapes/grads     OK ({count_params(net)/1e3:.0f}k params)")

    # a full-size model must also build (this is where v1's skip bookkeeping broke)
    big = KANUNet(img_size=256)
    assert count_params(big) > 1e6
    print(f"  unet 256 build        OK ({count_params(big)/1e6:.1f}M params)")


def t_radon():
    N = 64
    r = ParallelBeamRadon(N, 64, device=DEV, chunk=16)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, N), torch.linspace(-1, 1, N),
                            indexing="ij")
    disk = ((xx ** 2 + yy ** 2) < 0.35 ** 2).float()[None, None].to(DEV)
    sino = r(disk)
    assert sino.shape == (1, 1, 64, N), sino.shape
    assert torch.isfinite(sino).all() and sino.min() >= -1e-4
    # <A x, s> == <x, A^T s>  (adjoint identity)
    x = torch.randn(2, 1, N, N, device=DEV)
    s = torch.randn(2, 1, 64, N, device=DEV)
    lhs = (r(x) * s).sum()
    rhs = (x * r.adjoint(s)).sum()
    rel = ((lhs - rhs).abs() / lhs.abs().clamp_min(1e-8)).item()
    assert rel < 1e-4, f"adjoint mismatch rel={rel:.2e}"
    # FBP recovers the disk far better than plain back-projection
    fb = r.fbp(sino)
    fb = fb / fb.amax().clamp_min(1e-8)
    bp = r.backproject(sino)
    bp = bp / bp.amax().clamp_min(1e-8)
    e_fbp = F.mse_loss(fb, disk).item()
    e_bp = F.mse_loss(bp, disk).item()
    assert e_fbp < e_bp, f"FBP ({e_fbp:.4f}) no better than backprojection ({e_bp:.4f})"
    assert e_fbp < 0.05, f"FBP error too high: {e_fbp:.4f}"
    print(f"  radon/fbp             OK (adjoint rel={rel:.1e}, fbp mse={e_fbp:.4f})")


def t_schedule():
    s = MeanPreservingSchedule(8, 0.1, DEV)
    x0 = torch.randn(3, 1, 8, 8, device=DEV)
    y = torch.randn(3, 1, 8, 8, device=DEV)
    z = torch.zeros(3, dtype=torch.long, device=DEV)
    assert torch.allclose(s.degrade(x0, y, z), x0, atol=1e-6), "a_0 must be clean"
    assert torch.allclose(s.degrade(x0, y, z + 8), y, atol=1e-6), "a_T must be the LD image"
    t = torch.tensor([4, 4, 4], device=DEV)
    m = torch.stack([s.degrade(x0, y, t) for _ in range(400)]).mean(0)
    assert (m - s.mean(x0, y, t)).abs().mean() < 0.02, "degradation is not mean-preserving"
    print("  schedule              OK")


def t_identity_and_learning():
    cfg = tiny_cfg(two_stage=True, use_kan_acc=True, use_kan_noise=True)
    m = KANPGSD(cfg).to(DEV)
    sched = MeanPreservingSchedule(cfg.n_steps, cfg.bridge_sigma, DEV)
    x0 = torch.randn(2, 1, 32, 32, device=DEV).clamp(-1, 1)
    y = (x0 + 0.2 * torch.randn_like(x0)).clamp(-1, 1)

    out = m.sample(y, sched, cfg, None)
    assert torch.allclose(out, y.clamp(-1, 1), atol=1e-5), \
        "an untrained model must be the identity on the LD input"
    p_before = psnr(out, x0).mean().item()

    # overfit this one pair; PSNR must go up
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(120):
        d, _, _ = m.restoration_loss(x0, y, sched)
        loss = d["L_total"] + 1e-5 * kan_regularization(m.net)
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), "loss went non-finite"
    m.eval()
    with torch.no_grad():
        p_after = psnr(m.sample(y, sched, cfg, None), x0).mean().item()
    assert p_after > p_before + 1.0, f"no learning: {p_before:.2f} -> {p_after:.2f} dB"
    print(f"  identity + learning   OK ({p_before:.2f} -> {p_after:.2f} dB)")
    return m, sched, cfg, x0, y


def t_physics_heads(m, sched, cfg, x0, y):
    m.train()
    m.zero_grad(set_to_none=True)
    radon = ParallelBeamRadon(cfg.img_size, cfg.dc_angles, device=DEV,
                              chunk=cfg.angle_chunk)
    t = torch.full((2,), 2, device=DEV, dtype=torch.long)
    with torch.no_grad():
        x0_hat = m.predict_x0(sched.degrade(x0, y, t), y, t)
    d = m.physics_losses(x0, y, x0_hat, t, sched, radon, cfg)
    assert {"L_nm", "L_acc", "lam", "dc_gain"} <= set(d)
    total = d["L_nm"] + d["L_acc"]
    assert torch.isfinite(total)
    total.backward()
    for name, mod in (("acc", m.acc), ("noise", m.noise)):
        g = [p.grad for p in mod.parameters() if p.grad is not None]
        assert g and any(x.abs().sum() > 0 for x in g), f"KAN-{name} got no gradient"
    # the restoration net must be untouched by the physics stage
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m.net.parameters()), \
        "physics stage leaked gradients into the restoration network"
    with torch.no_grad():
        s = m.sample(y, sched, cfg, radon)
    assert torch.isfinite(s).all() and s.shape == y.shape
    mu = norm_to_mu(x0, cfg.hu_min, cfg.hu_max, cfg.mu_water)
    assert mu.min() >= 0, "attenuation must be non-negative"
    w = poisson_weights(radon(mu), cfg.I0)
    assert torch.isfinite(w).all() and w.max() <= 1.0 + 1e-6
    print(f"  physics heads         OK (lambda={d['lam']:.3f}, "
          f"dc_gain={d['dc_gain']:+.2e})")


def t_metrics():
    x = torch.rand(4, 1, 32, 32) * 2 - 1
    assert psnr(x, x).min() > 100, "psnr of identical images must be huge"
    assert (ssim(x, x) - 1).abs().max() < 1e-3, "ssim of identical images must be 1"
    n = (x + 0.1 * torch.randn_like(x)).clamp(-1, 1)
    assert (psnr(n, x) < psnr(x, x)).all()
    d = slice_metrics(n, x, tiny_cfg())
    assert set(d) == {"psnr", "ssim", "rmse_hu", "psnr_w", "ssim_w", "rmse_w_hu"}
    assert all(torch.isfinite(v).all() for v in d.values())
    print("  metrics               OK")


def t_baselines():
    x = torch.randn(2, 1, 32, 32, device=DEV)
    for name, ctor in BASELINES.items():
        net = ctor().to(DEV)
        y = net(x)
        assert y.shape == x.shape, f"{name}: {y.shape}"
        assert torch.isfinite(y).all(), f"{name} non-finite"
        y.sum().backward()
    print("  baselines             OK")


def t_data():
    """Synthetic mirror of the Mayo layout: pairing, split and normalisation."""
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        for dose, off in (("Quarter Dose", 40), ("Full Dose", 0)):
            for pat in ("L067", "L096", "L109", "L143"):
                sub = os.path.join(d, dose, f"1mm B30 {pat}")
                os.makedirs(sub)
                for i in range(6):
                    rng = np.random.RandomState(hash((pat, i)) % 2 ** 31)
                    base = rng.randint(0, 200, (16, 16)).astype(np.uint8)
                    a = np.clip(base + rng.randint(0, off + 1, (16, 16)), 0, 255)
                    Image.fromarray(a.astype(np.uint8)).save(
                        os.path.join(sub, f"slice_{i:04d}.png"))
        cfg = tiny_cfg(data_root=d, img_size=16, val_patients=1, test_patients=1,
                       num_workers=0)
        pairs = build_pairs(cfg.ld_root(), cfg.nd_root(), verbose=False)
        assert len(pairs) == 24, len(pairs)
        for lp, np_ in pairs:      # same patient folder and same slice index
            assert os.path.basename(lp) == os.path.basename(np_)
            assert os.path.basename(os.path.dirname(lp)).split()[-1] == \
                   os.path.basename(os.path.dirname(np_)).split()[-1]
        from kanldct.data import make_datasets
        tr, va, te = make_datasets(cfg, verbose=False)
        assert len(tr) == 12 and len(va) == 6 and len(te) == 6, (len(tr), len(va), len(te))
        tr_pat = {os.path.dirname(p[0]) for p in tr.pairs}
        va_pat = {os.path.dirname(p[0]) for p in va.pairs}
        assert not (tr_pat & va_pat), "patient leaked across the split"
        ld, nd = tr[0]
        assert ld.shape == (1, 16, 16) and -1 <= ld.min() <= ld.max() <= 1
        # one shared affine map, so a constant-offset LD stays a constant offset
        a = raw_to_norm(np.full((4, 4), 100, np.uint8), "uint8_linear", -1024, 3071)
        b = raw_to_norm(np.full((4, 4), 120, np.uint8), "uint8_linear", -1024, 3071)
        assert np.allclose(b - a, (20 / 255) * 2), "normalisation is not shared/linear"
        r = verify_pairing(tr, n=12, thresh=0.5)
        assert r > 0.5, f"pairing verification failed on synthetic data (r={r:.3f})"
    print("  data pipeline         OK")


if __name__ == "__main__":
    print(f"device: {DEV}")
    t_kan_layers()
    t_unet_shapes()
    t_radon()
    t_schedule()
    m, sched, cfg, x0, y = t_identity_and_learning()
    t_physics_heads(m, sched, cfg, x0, y)
    t_metrics()
    t_baselines()
    t_data()
    print("\nALL CHECKS PASSED")
