"""Parallel-beam Radon / FBP and the PWLS data-consistency term.

Fixes over the v1 notebook:

* ``forward``/``adjoint`` are vectorised over angles (chunked) instead of a
  Python loop per angle per sample — two orders of magnitude faster.
* ``fbp`` uses a properly scaled Ram-Lak ramp with a Hann apodisation and
  zero-padding to twice the detector width, so it produces a real
  reconstruction instead of something that has to be max-normalised afterwards.
* The image is converted to linear attenuation coefficients before projection
  (mu = mu_water * (1 + HU/1000)).  Projecting a [-1, 1] display image, as v1
  did, is dimensionally meaningless and was the source of the Beer-Lambert
  "incoherence" the old notebook papered over with clamps.

HONEST SCOPE NOTE
-----------------
The Mayo *preprocessed image* release ships reconstructed slices, not raw
projections.  The consistency term below therefore compares the re-projection
of the estimate against the re-projection of the measured low-dose slice.  That
is a statistically weighted (PWLS) regulariser, **not** true measurement
fidelity — it cannot recover information the LDCT reconstruction already
destroyed.  Every claim in the report is phrased accordingly, and
``--dc_step 0`` runs the ablation without it.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def norm_to_hu(x: torch.Tensor, hu_min: float, hu_max: float) -> torch.Tensor:
    return (x + 1.0) * 0.5 * (hu_max - hu_min) + hu_min


def hu_to_norm(hu: torch.Tensor, hu_min: float, hu_max: float) -> torch.Tensor:
    return (hu - hu_min) / (hu_max - hu_min) * 2.0 - 1.0


def norm_to_mu(x: torch.Tensor, hu_min: float, hu_max: float,
               mu_water: float = 0.0192) -> torch.Tensor:
    """[-1, 1] display value -> linear attenuation coefficient (mm^-1), >= 0."""
    hu = norm_to_hu(x, hu_min, hu_max)
    return (mu_water * (1.0 + hu / 1000.0)).clamp_min(0.0)


class ParallelBeamRadon:
    """Rotate-and-sum parallel-beam projector on a square grid."""

    def __init__(self, image_size=256, n_angles=128, angle_range=math.pi,
                 device="cpu", chunk=32, dtype=torch.float32):
        self.image_size = image_size
        self.n_angles = n_angles
        self.chunk = chunk
        self.device = device
        angles = torch.linspace(0.0, angle_range, n_angles + 1, device=device)[:-1]
        c, s = torch.cos(angles), torch.sin(angles)
        zero = torch.zeros_like(c)
        mat = torch.stack([torch.stack([c, s, zero], -1),
                           torch.stack([-s, c, zero], -1)], dim=-2)   # (A,2,3)
        grid = F.affine_grid(mat, [n_angles, 1, image_size, image_size],
                             align_corners=False).to(dtype)
        self.grids = grid                                             # (A,H,W,2)
        # each detector bin integrates one pixel row => pixel-size scaling
        self.pixel = 1.0

    def to(self, device):
        self.grids = self.grids.to(device)
        self.device = device
        return self

    # ------------------------------------------------------------------ A --
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,H,W) attenuation image -> sinogram (B,1,A,W)."""
        B = x.size(0)
        outs = []
        for lo in range(0, self.n_angles, self.chunk):
            g = self.grids[lo:lo + self.chunk]                        # (a,H,W,2)
            a = g.size(0)
            xb = x.unsqueeze(1).expand(B, a, -1, -1, -1).reshape(B * a, 1,
                                                                 *x.shape[-2:])
            gb = g.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * a,
                                                                  *g.shape[-3:])
            rot = F.grid_sample(xb, gb.to(xb.dtype), mode="bilinear",
                                padding_mode="zeros", align_corners=False)
            outs.append(rot.sum(-2).reshape(B, a, -1))
        return torch.cat(outs, dim=1).unsqueeze(1) * self.pixel        # (B,1,A,W)

    __call__ = forward

    # ---------------------------------------------------------------- A^T --
    def adjoint(self, sino: torch.Tensor) -> torch.Tensor:
        """Exact transpose of :meth:`forward`: (B,1,A,W) -> (B,1,H,W).

        Computed as the vector-Jacobian product of the (linear) projector, so
        it is the true adjoint down to floating point.  Writing a second
        rotate-and-sum loop — what v1 did — only *approximates* the transpose
        and silently biases every PWLS gradient.
        """
        x = torch.zeros(sino.size(0), 1, self.image_size, self.image_size,
                        device=sino.device, dtype=torch.float32,
                        requires_grad=True)
        with torch.enable_grad():
            y = self.forward(x)
        (g,) = torch.autograd.grad(y, x, grad_outputs=sino.float(),
                                   create_graph=bool(sino.requires_grad))
        return g.to(sino.dtype)

    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        """Back-projection carrying the FBP angular measure pi/A."""
        return self.adjoint(sino) * (math.pi / sino.size(-2))

    # ---------------------------------------------------------------- FBP --
    def _ramp(self, W: int, device, dtype, window="hann"):
        n = 1
        while n < 2 * W:
            n *= 2
        f = torch.fft.rfftfreq(n, d=self.pixel, device=device)
        ramp = 2.0 * f                                     # Ram-Lak
        if window == "hann":
            ramp = ramp * (0.5 + 0.5 * torch.cos(math.pi * f / f[-1].clamp_min(1e-12)))
        return n, ramp.to(dtype)

    def fbp(self, sino: torch.Tensor, window="hann") -> torch.Tensor:
        W = sino.size(-1)
        n, ramp = self._ramp(W, sino.device, torch.float32)
        s = F.pad(sino.float(), (0, n - W))
        filt = torch.fft.irfft(torch.fft.rfft(s, dim=-1) * ramp, n=n, dim=-1)[..., :W]
        return self.backproject(filt.to(sino.dtype))


# ------------------------------------------------------------------ PWLS --
def poisson_weights(p: torch.Tensor, I0: float = 1e4) -> torch.Tensor:
    """Analytic PWLS weight w = I0 exp(-p) (used when KAN-NM is disabled)."""
    w = I0 * torch.exp(-p.clamp(0.0, 12.0))
    return w / w.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)


# --------------------------------------------------------- LDCT simulator --
@torch.no_grad()
def simulate_ldct_sino(clean_sino: torch.Tensor, I0: float = 1e4,
                       sigma_e: float = 10.0) -> torch.Tensor:
    """Poisson + Gaussian degradation of a *physically valid* sinogram."""
    p = clean_sino.clamp(0.0, 12.0)
    lam = I0 * torch.exp(-p)
    counts = torch.poisson(lam) + torch.randn_like(lam) * sigma_e
    return -torch.log(counts.clamp_min(1.0) / I0)
