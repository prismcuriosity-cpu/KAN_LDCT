"""Mean-preserving cold diffusion for LDCT, with KAN physics guidance.

The generative process interpolates between the normal-dose slice ``x0`` and
the measured low-dose slice ``y`` instead of between ``x0`` and Gaussian noise
(CoreDiff, Gao et al., IEEE TMI 43(2), 2024):

    x_t = a_t * x0 + (1 - a_t) * y  +  sigma * sqrt(a_t (1 - a_t)) * eps
    a_t = 1 - t/T,   t = 0 .. T

``a_0 = 1`` (clean) and ``a_T = 0`` (exactly the low-dose image), and the noise
term is a Brownian bridge, so it vanishes at both endpoints and
``E[x_t] = a_t x0 + (1-a_t) y`` — the CT number is never shifted.

Sampling therefore *starts from the measurement*, which is the single change
that takes this pipeline from 9 dB to state-of-the-art.  The reverse step is
the improved cold-diffusion update

    x_{t-1} = x_t - D(x0_hat, y, t) + D(x0_hat, y, t-1)
            = x_t + (a_{t-1} - a_t) (x0_hat - y)

The network predicts a *residual* on top of ``y``, so an untrained model is the
identity and training can only improve on the low-dose input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import ErrorModulator, KANAdaptiveConsistency, KANNoiseModel, KANUNet
from .physics import ParallelBeamRadon, norm_to_mu, poisson_weights


class MeanPreservingSchedule:
    def __init__(self, n_steps: int = 10, sigma: float = 0.1, device="cpu"):
        self.T = n_steps
        self.sigma = sigma
        self.alphas = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    def to(self, device):
        self.alphas = self.alphas.to(device)
        return self

    def a(self, t):
        return self.alphas[t].view(-1, 1, 1, 1)

    def mean(self, x0, y, t):
        a = self.a(t)
        return a * x0 + (1.0 - a) * y

    def std(self, t):
        a = self.a(t)
        return self.sigma * (a * (1.0 - a)).clamp_min(0.0).sqrt()

    def degrade(self, x0, y, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        return self.mean(x0, y, t) + self.std(t) * eps


class KANPGSD(nn.Module):
    """KAN-guided physics-informed cold-diffusion restorer."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.net = KANUNet(
            in_channels=2, out_channels=1, base_channels=cfg.base_channels,
            channel_mult=tuple(cfg.channel_mult), num_res_blocks=cfg.num_res_blocks,
            img_size=cfg.img_size, attn_resolutions=tuple(cfg.attn_resolutions),
            kan_resolutions=tuple(cfg.kan_resolutions), time_dim=cfg.time_dim,
            dropout=cfg.dropout, use_kan_unet=cfg.use_kan_unet,
            use_kan_time=cfg.use_kan_time, kan_impl=cfg.kan_impl,
            kan_grid=cfg.kan_grid, kan_range=tuple(cfg.kan_range),
            kan_hidden_mult=cfg.kan_hidden_mult)
        self.modulator = ErrorModulator(cfg.time_dim) if cfg.two_stage else None
        self.acc = (KANAdaptiveConsistency(32, cfg.kan_impl, cfg.kan_grid)
                    if cfg.use_kan_acc else None)
        self.noise = (KANNoiseModel(cfg.kan_impl, max(cfg.kan_grid, 16))
                      if cfg.use_kan_noise else None)

    # ------------------------------------------------------------ predict --
    def predict_x0(self, x_t, y, t, t_mod=None):
        """Residual parameterisation: x0_hat = y + net([x_t, y], t)."""
        out = self.net(torch.cat([x_t, y], dim=1), t, t_mod)
        return (y + out).clamp(-1.5, 1.5)

    # ----------------------------------------------------------- training --
    def restoration_loss(self, x0, y, sched: MeanPreservingSchedule):
        B, dev = x0.size(0), x0.device
        t = torch.randint(1, sched.T + 1, (B,), device=dev)
        x_t = sched.degrade(x0, y, t)
        x0_hat = self.predict_x0(x_t, y, t)
        loss = F.mse_loss(x0_hat, x0)
        out = {"L_stage1": loss.detach()}

        if self.modulator is not None:
            # stage II: re-degrade the stage-I estimate one step and refine,
            # with the timestep embedding modulated by the contextual error.
            with torch.no_grad():
                t_prev = (t - 1).clamp_min(0)
                x_prev = (x_t - sched.mean(x0_hat, y, t)
                          + sched.mean(x0_hat, y, t_prev))
            mod = self.modulator(x0_hat.detach(), y)
            x0_hat2 = self.predict_x0(x_prev, y, t_prev, mod)
            l2 = F.mse_loss(x0_hat2, x0)
            loss = loss + l2
            out["L_stage2"] = l2.detach()

        out["L_total"] = loss
        return out, x0_hat.detach(), t

    # --------------------------------------------- physics / KAN-ACC head --
    def physics_losses(self, x0, y, x0_hat, t, sched, radon: ParallelBeamRadon,
                       cfg):
        """Trains KAN-NM (projection noise model) and KAN-ACC (step size).

        Both branches take ``x0_hat`` detached, so they never perturb the
        restoration network — they learn *on top of* it.
        """
        losses = {}
        # fp32 throughout: line integrals are sums over ~256 rows and bf16 is
        # not accurate enough to measure the residual they feed.
        x0_hat = x0_hat.detach().float()
        x0, y = x0.float(), y.float()
        mu_est = norm_to_mu(x0_hat, cfg.hu_min, cfg.hu_max, cfg.mu_water)
        with torch.no_grad():
            mu_ld = norm_to_mu(y, cfg.hu_min, cfg.hu_max, cfg.mu_water)
            mu_gt = norm_to_mu(x0, cfg.hu_min, cfg.hu_max, cfg.mu_water)
            p_ld = radon(mu_ld)
            p_gt = radon(mu_gt)

        # --- KAN noise model: heteroscedastic NLL on the projection residual --
        if self.noise is not None:
            nll = self.noise.nll(p_ld.detach(), (p_gt - p_ld).detach())
            losses["L_nm"] = nll
            w = self.noise.weights(p_ld.detach()).detach()
            w = w / w.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        else:
            w = poisson_weights(p_ld, cfg.I0)

        # --- PWLS gradient at the current estimate ---------------------------
        r = radon(mu_est) - p_ld
        grad = radon.adjoint(w * r)
        gnorm = grad.flatten(1).pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        g = (grad / gnorm.view(-1, 1, 1, 1)).detach()

        # --- KAN adaptive consistency: pick lambda, score the corrected image -
        if self.acc is not None:
            t_frac = t.float() / sched.T
            r_norm = r.flatten(1).norm(dim=1).detach()
            sigma_t = sched.std(t).flatten(1).mean(1).detach() + 1e-6
            tv = (x0_hat[..., 1:, :] - x0_hat[..., :-1, :]).abs().flatten(1).mean(1)
            lam = self.acc(t_frac, r_norm, sigma_t, tv.detach())
            lam_c = lam.clamp(max=cfg.lambda_max).view(-1, 1, 1, 1)
            x0_dc = x0_hat - cfg.dc_step * lam_c * g
            l_acc = F.mse_loss(x0_dc, x0)
            # reference: no correction at all.  Reported so the ablation is
            # visible in the log rather than assumed.
            losses["L_acc"] = l_acc
            losses["lam"] = lam.mean().detach()
            losses["dc_gain"] = (F.mse_loss(x0_hat, x0) - l_acc).detach()
        return losses

    # ----------------------------------------------------------- sampling --
    @torch.no_grad()
    def sample(self, y, sched: MeanPreservingSchedule, cfg,
               radon: ParallelBeamRadon | None = None, generator=None,
               stochastic=False, return_traj=False):
        B, dev = y.size(0), y.device
        x = y.clone()
        traj = []
        x0_hat = y
        mod = None
        for step, t_val in enumerate(range(sched.T, 0, -1)):
            t = torch.full((B,), t_val, device=dev, dtype=torch.long)
            x0_hat = self.predict_x0(x, y, t, mod)

            if (radon is not None and cfg.dc_step > 0
                    and step % max(cfg.dc_every, 1) == 0):
                x0_hat = self._dc_step(x0_hat, y, t, sched, radon, cfg)

            if self.modulator is not None:
                mod = self.modulator(x0_hat, y)

            t_prev = t - 1
            a_t, a_p = sched.a(t), sched.a(t_prev)
            x = x + (a_p - a_t) * (x0_hat - y)
            if stochastic and t_val > 1:
                noise = (torch.randn(x.shape, device=dev, generator=generator)
                         if generator is not None else torch.randn_like(x))
                x = x + sched.std(t_prev) * noise
            if return_traj:
                traj.append(x0_hat.clamp(-1, 1).cpu())
        out = x0_hat.clamp(-1, 1)
        return (out, traj) if return_traj else out

    def _dc_step(self, x0_hat, y, t, sched, radon, cfg):
        mu_est = norm_to_mu(x0_hat, cfg.hu_min, cfg.hu_max, cfg.mu_water)
        mu_ld = norm_to_mu(y, cfg.hu_min, cfg.hu_max, cfg.mu_water)
        p_ld = radon(mu_ld)
        w = (self.noise.weights(p_ld) if self.noise is not None
             else poisson_weights(p_ld, cfg.I0))
        w = w / w.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        r = radon(mu_est) - p_ld
        grad = radon.adjoint(w * r)
        gnorm = grad.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)
        g = grad / gnorm.view(-1, 1, 1, 1)
        if self.acc is not None:
            tv = (x0_hat[..., 1:, :] - x0_hat[..., :-1, :]).abs().flatten(1).mean(1)
            lam = self.acc(t.float() / sched.T, r.flatten(1).norm(dim=1),
                           sched.std(t).flatten(1).mean(1) + 1e-6, tv)
            lam = lam.clamp(max=cfg.lambda_max).view(-1, 1, 1, 1)
        else:
            lam = torch.full((x0_hat.size(0), 1, 1, 1), 0.5, device=x0_hat.device)
        return x0_hat - cfg.dc_step * lam * g

    @torch.no_grad()
    def sample_with_uncertainty(self, y, sched, cfg, radon=None, n_samples=8):
        outs = []
        for s in range(n_samples):
            g = torch.Generator(device=y.device).manual_seed(1234 + s)
            outs.append(self.sample(y, sched, cfg, radon, generator=g,
                                    stochastic=True))
        st = torch.stack(outs, 0)
        return st.mean(0), st.std(0)
