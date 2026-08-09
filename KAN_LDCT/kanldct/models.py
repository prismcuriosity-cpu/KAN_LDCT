"""KAN-UNet restoration network + the two auxiliary KAN heads.

Differences from the v1 notebook that actually matter:

1.  The network is **conditional**: it receives ``cat([x_t, y_ld], dim=1)``.
    v1 trained an unconditional score model and then tried to steer it back to
    the anatomy with a 0.05-scaled gradient — that is why it hallucinated.
2.  The timestep embedding is built from the **integer** timestep.  v1 fed
    ``t / n_steps`` into ``exp(-k log(10000)/half)`` frequencies, so every
    frequency channel collapsed into the same near-linear ramp and the network
    was effectively time-blind.
3.  Self-attention at 32x32 and 16x16 (standard for diffusion UNets, absent
    in v1).
4.  Tokenized KAN blocks (U-KAN style) at the two lowest resolutions instead of
    a single bottleneck KAN.
5.  DDPM-style skip *stack* so encoder/decoder channel bookkeeping is correct
    by construction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan import TokenKANBlock, groups_for, make_kan


# ----------------------------------------------------------------- blocks --
class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of the *integer* timestep, then MLP (or KAN-MLP)."""

    def __init__(self, dim: int, use_kan: bool = True, impl: str = "rbf",
                 grid: int = 8):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer("freqs", freqs, persistent=False)
        if use_kan:
            self.mlp = nn.Sequential(
                make_kan(impl, dim, dim * 2, num_grids=grid, grid_range=(-2.0, 2.0)),
                nn.SiLU(),
                nn.Linear(dim * 2, dim))
        else:
            self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(),
                                     nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timesteps.  Scaled by 1000/T-independent constant so
        # the frequency ladder spans several decades regardless of n_steps.
        args = t.float().unsqueeze(-1) * self.freqs.unsqueeze(0) * 100.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.dim:                       # odd dim
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups_for(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch * 2)         # FiLM: scale + shift
        self.norm2 = nn.GroupNorm(groups_for(out_ch), out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.t_proj(F.silu(t_emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        # heads must divide ch and leave a head dim of at least 16
        self.heads = next((h for h in range(min(heads, max(ch // 16, 1)), 0, -1)
                           if ch % h == 0), 1)
        self.norm = nn.GroupNorm(groups_for(ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(
            B, 3, self.heads, C // self.heads, H * W).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2),
                                           v.transpose(-1, -2))
        o = o.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(o)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, t=None):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, t=None):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


# ------------------------------------------------------------------ U-Net --
class KANUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=64,
                 channel_mult=(1, 1, 2, 2, 4), num_res_blocks=2, img_size=256,
                 attn_resolutions=(32, 16), kan_resolutions=(32, 16),
                 time_dim=256, dropout=0.0, use_kan_unet=True, use_kan_time=True,
                 kan_impl="rbf", kan_grid=8, kan_range=(-2.0, 2.0),
                 kan_hidden_mult=2):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim, use_kan_time, kan_impl, kan_grid)
        attn_resolutions = set(int(r) for r in attn_resolutions)
        kan_resolutions = set(int(r) for r in kan_resolutions) if use_kan_unet else set()

        def kanblk(ch):
            return TokenKANBlock(ch, kan_impl, kan_grid, kan_range, kan_hidden_mult)

        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down = nn.ModuleList()
        skip_chs = [base_channels]
        ch, res = base_channels, img_size
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_dim, dropout)]
                if res in attn_resolutions:
                    layers.append(AttnBlock(out_ch))
                if res in kan_resolutions:
                    layers.append(kanblk(out_ch))
                self.down.append(nn.ModuleList(layers))
                ch = out_ch
                skip_chs.append(ch)
            if level != len(channel_mult) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                skip_chs.append(ch)
                res //= 2

        self.mid = nn.ModuleList([ResBlock(ch, ch, time_dim, dropout),
                                  AttnBlock(ch),
                                  kanblk(ch) if kan_resolutions else nn.Identity(),
                                  ResBlock(ch, ch, time_dim, dropout)])

        self.up = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                layers = [ResBlock(ch + skip_chs.pop(), out_ch, time_dim, dropout)]
                if res in attn_resolutions:
                    layers.append(AttnBlock(out_ch))
                if res in kan_resolutions:
                    layers.append(kanblk(out_ch))
                ch = out_ch
                if level != 0 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    res *= 2
                self.up.append(nn.ModuleList(layers))

        self.out_norm = nn.GroupNorm(groups_for(ch), ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, t_mod=None):
        """x: (B, 2, H, W) = [x_t, y_ld].  t: (B,) int.  t_mod: (beta, gamma)."""
        t_emb = self.time_embed(t)
        if t_mod is not None:
            beta, gamma = t_mod
            t_emb = beta * t_emb + gamma
        h = self.stem(x)
        hs = [h]
        for group in self.down:
            for layer in group:
                h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
        for group in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in group:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, Upsample):
                    h = layer(h)
                else:
                    h = layer(h)
        return self.out_conv(F.silu(self.out_norm(h)))


class ErrorModulator(nn.Module):
    """CLEAR-Net style contextual modulation (CoreDiff, IEEE TMI 2024).

    Looks at the stage-I estimate and the measured low-dose image, and emits a
    (scale, shift) pair applied to the timestep embedding in stage II.
    """

    def __init__(self, time_dim: int, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width * 2, time_dim * 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x0_hat, y):
        beta, gamma = self.net(torch.cat([x0_hat, y], dim=1)).chunk(2, dim=-1)
        return 1.0 + beta, gamma


# ------------------------------------------------------------- KAN heads --
class KANAdaptiveConsistency(nn.Module):
    """lambda(t, r, sigma, grad) -> data-consistency step size.

    v1 trained this head *inside* the objective it scaled, which is degenerate:
    the loss is minimised by lambda -> 0.  Here it is trained by
    differentiating through one data-consistency step and scoring the result
    against the normal-dose ground truth, so lambda has a genuine optimum.
    """

    def __init__(self, hidden=32, impl="rbf", grid=8):
        super().__init__()
        self.k1 = make_kan(impl, 4, hidden, num_grids=grid, grid_range=(-2.0, 2.0))
        self.k2 = make_kan(impl, hidden, 1, num_grids=grid, grid_range=(-2.0, 2.0))
        self.act = nn.Softplus(beta=2.0)
        self.register_buffer("log_scales", torch.zeros(3), persistent=True)
        self.register_buffer("seen", torch.zeros((), dtype=torch.long), persistent=True)

    def _features(self, t_frac, r_norm, sigma_t, grad_norm):
        raw = torch.stack([r_norm, sigma_t, grad_norm], dim=-1).clamp_min(1e-8).log()
        if self.training:                                  # running mean of log-scale
            with torch.no_grad():
                m = raw.mean(0)
                w = 0.01 if self.seen > 0 else 1.0
                self.log_scales.mul_(1 - w).add_(w * m)
                self.seen += 1
        z = (raw - self.log_scales).clamp(-4.0, 4.0) * 0.5
        return torch.cat([(t_frac * 2.0 - 1.0).unsqueeze(-1), z], dim=-1)

    def forward(self, t_frac, r_norm, sigma_t, grad_norm):
        x = self._features(t_frac, r_norm, sigma_t, grad_norm)
        return self.act(self.k2(self.k1(x)).squeeze(-1))

    @torch.no_grad()
    def sweep(self, axis: int, n: int = 128):
        """1-D sweep of lambda along one normalised input axis (for figures)."""
        dev = self.log_scales.device
        x = torch.zeros(n, 4, device=dev)
        x[:, axis] = torch.linspace(-2.0, 2.0, n, device=dev)
        was = self.training
        self.eval()
        y = self.act(self.k2(self.k1(x)).squeeze(-1))
        self.train(was)
        return x[:, axis].cpu(), y.cpu()


class KANNoiseModel(nn.Module):
    """Learned projection-domain noise model  log sigma^2 = f_KAN(p).

    Trained with a heteroscedastic Gaussian NLL on the projection residual
    between the normal-dose and low-dose re-projections.  Its output *is* the
    PWLS statistical weight w = 1/sigma^2, so unlike v1 (where the head was
    never connected to any loss) it has a real job and its symbolic form is a
    genuine result: it should recover sigma^2 ~ exp(p)/I0.
    """

    def __init__(self, impl="rbf", grid=16, p_max: float = 8.0):
        super().__init__()
        self.p_max = p_max
        self.kan = make_kan(impl, 1, 1, num_grids=grid, grid_range=(-2.0, 2.0),
                            use_layernorm=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def log_var(self, p: torch.Tensor) -> torch.Tensor:
        u = (p.clamp(0.0, self.p_max) / self.p_max * 4.0 - 2.0).reshape(-1, 1)
        return (self.kan(u).reshape(p.shape) + self.bias).clamp(-12.0, 12.0)

    def weights(self, p: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.log_var(p))

    def nll(self, p: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
        lv = self.log_var(p)
        return (0.5 * (resid.pow(2) * torch.exp(-lv) + lv)).mean()

    @torch.no_grad()
    def fit_symbolic(self, n=512):
        """Least-squares fit of candidate closed forms to the learned sigma^2(p)."""
        import numpy as np
        dev = self.bias.device
        p = torch.linspace(0.0, self.p_max, n, device=dev)
        y = self.log_var(p).cpu().numpy()
        pn = p.cpu().numpy()
        cands = {                       # log sigma^2(p) = a * f(p) + b
            "poisson   log s2 = p + c":      pn,
            "constant  log s2 = c":          np.zeros_like(pn),
            "quadratic log s2 = p^2 + c":    pn ** 2,
            "log       log s2 = log(1+p)+c": np.log1p(pn),
            "sqrt      log s2 = sqrt(p)+c":  np.sqrt(pn),
        }
        best = {"form": None, "r2": -float("inf")}
        for name, f in cands.items():
            A = np.stack([f, np.ones_like(f)], 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = 1.0 - ((y - A @ coef) ** 2).sum() / (
                ((y - y.mean()) ** 2).sum() + 1e-12)
            if r2 > best["r2"]:
                best = {"form": name, "r2": float(r2), "a": float(coef[0]),
                        "b": float(coef[1]), "p": pn, "fit": A @ coef, "y": y}
        return best


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
