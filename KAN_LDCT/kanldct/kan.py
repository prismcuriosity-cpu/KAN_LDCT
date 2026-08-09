"""Kolmogorov-Arnold layers used throughout the model.

Two bases are provided:

``RBFKANLinear``  - Gaussian radial basis functions (FastKAN, Li 2024).  No
                    Cox-de Boor recursion, no division, no clamping.  ~3x
                    faster than the B-spline form and numerically far safer.
``BSplineKANLinear`` - the classic efficient-KAN basis, kept for ablations.

Both put a ``LayerNorm`` in front of the basis expansion.  That is the fix for
the failure mode in the v1 notebook: there the inputs were *clamped* into the
knot range, which silently zeroes the gradient for every saturated unit.  A
LayerNorm instead *keeps* the inputs inside the grid while staying
differentiable everywhere.

Both layers evaluate the basis in float32 even under autocast — bf16 has 8
mantissa bits, which is not enough for stable spline/RBF interpolation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def groups_for(ch: int, target: int = 32) -> int:
    """Largest group count <= ``target`` that divides ``ch``.

    ``min(32, ch)`` (the obvious version) blows up on 48, 96, 192 ... — any
    channel width that is not a multiple of 32.  Concatenated skip connections
    hit exactly those widths.
    """
    for g in (target, 16, 8, 4, 2):
        if ch % g == 0:
            return g
    return 1


class RBFKANLinear(nn.Module):
    """y = W_base @ SiLU(LN(x)) + W_spline @ phi(LN(x)), phi = Gaussian RBFs."""

    def __init__(self, in_features: int, out_features: int, num_grids: int = 8,
                 grid_range=(-2.0, 2.0), use_layernorm: bool = True,
                 spline_scale: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids
        grid = torch.linspace(grid_range[0], grid_range[1], num_grids)
        self.register_buffer("grid", grid, persistent=False)
        # RBF width = one grid spacing -> neighbouring bumps overlap at ~e^-1
        self.denom = (grid_range[1] - grid_range[0]) / max(num_grids - 1, 1)
        self.norm = nn.LayerNorm(in_features) if use_layernorm else nn.Identity()
        self.base = nn.Linear(in_features, out_features)
        self.spline = nn.Linear(in_features * num_grids, out_features, bias=False)
        nn.init.trunc_normal_(
            self.spline.weight, std=spline_scale / math.sqrt(in_features * num_grids))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            h = self.norm(x.float())
            phi = torch.exp(-((h.unsqueeze(-1) - self.grid) / self.denom) ** 2)
            y = self.base(F.silu(h)) + self.spline(phi.flatten(-2))
        return y.to(dt)

    def regularization_loss(self) -> torch.Tensor:
        return self.spline.weight.abs().mean()



class BSplineKANLinear(nn.Module):
    """Classic efficient-KAN B-spline basis (ablation path)."""

    def __init__(self, in_features: int, out_features: int, num_grids: int = 8,
                 grid_range=(-2.0, 2.0), spline_order: int = 3,
                 use_layernorm: bool = True, spline_scale: float = 0.1):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = num_grids, spline_order
        h = (grid_range[1] - grid_range[0]) / num_grids
        grid = (torch.arange(-spline_order, num_grids + spline_order + 1) * h
                + grid_range[0])
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous(),
                             persistent=False)
        self.norm = nn.LayerNorm(in_features) if use_layernorm else nn.Identity()
        self.base = nn.Linear(in_features, out_features)
        self.spline = nn.Linear(in_features * (num_grids + spline_order),
                                out_features, bias=False)
        nn.init.trunc_normal_(
            self.spline.weight,
            std=spline_scale / math.sqrt(in_features * (num_grids + spline_order)))

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        g = self.grid
        x = x.unsqueeze(-1)
        b = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            lo = (x - g[:, : -(k + 1)]) / (g[:, k:-1] - g[:, : -(k + 1)])
            hi = (g[:, k + 1:] - x) / (g[:, k + 1:] - g[:, 1:-k])
            b = lo * b[..., :-1] + hi * b[..., 1:]
        return b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            h = self.norm(x.float())
            y = self.base(F.silu(h)) + self.spline(self._b_splines(h).flatten(-2))
        return y.to(dt)

    def regularization_loss(self) -> torch.Tensor:
        return self.spline.weight.abs().mean()


def make_kan(impl: str, *a, **kw) -> nn.Module:
    if impl == "rbf":
        return RBFKANLinear(*a, **kw)
    if impl == "bspline":
        return BSplineKANLinear(*a, **kw)
    raise ValueError(f"unknown kan_impl {impl!r}")


def kan_regularization(module: nn.Module) -> torch.Tensor:
    """Sum the L1 spline penalty over every KAN layer inside ``module``."""
    terms = [m.regularization_loss() for m in module.modules()
             if isinstance(m, (RBFKANLinear, BSplineKANLinear))]
    if not terms:
        return torch.zeros((), device=next(module.parameters()).device)
    return torch.stack(terms).mean()


# ---------------------------------------------------------------------------
# Tokenized KAN block (U-KAN, Li et al., AAAI 2025).
#
# The KAN acts on the channel vector of every spatial token; a depth-wise 3x3
# convolution between the two KAN layers restores the local spatial inductive
# bias that a purely per-token operator would throw away.
# ---------------------------------------------------------------------------
class DWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.bn = nn.GroupNorm(1, dim)

    def forward(self, x, H, W):                 # x: (B, N, C)
        B, N, C = x.shape
        h = x.transpose(1, 2).reshape(B, C, H, W)
        h = F.silu(self.bn(self.dw(h)))
        return h.flatten(2).transpose(1, 2)


class TokenKANBlock(nn.Module):
    """LN -> KAN -> DWConv -> KAN -> residual, over (B, C, H, W) feature maps."""

    def __init__(self, dim: int, impl: str = "rbf", grid: int = 8,
                 grid_range=(-2.0, 2.0), hidden_mult: int = 2):
        super().__init__()
        hidden = dim * hidden_mult
        self.norm = nn.GroupNorm(groups_for(dim), dim)
        self.fc1 = make_kan(impl, dim, hidden, num_grids=grid, grid_range=grid_range)
        self.dw = DWConv(hidden)
        self.fc2 = make_kan(impl, hidden, dim, num_grids=grid, grid_range=grid_range)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))   # zero-init residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)            # (B, N, C)
        h = self.fc1(h)
        h = self.dw(h, H, W)
        h = self.fc2(h)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return x + self.gamma * h
