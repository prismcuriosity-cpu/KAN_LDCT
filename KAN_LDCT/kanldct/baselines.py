"""Supervised LDCT denoising baselines, trained on the same split and budget.

* RED-CNN (Chen et al., IEEE TMI 2017) — corrected residual wiring; the v1
  notebook popped its skip stack in the wrong order.
* EDCNN (Liang et al., 2020) — Sobel edge-enhancement front end + dense convs.
* U-Net denoiser — plain supervised baseline.
* Identity — the low-dose input itself.  This is the number every method has to
  beat, and v1's "KAN-PGSD" (9.31 dB vs 37.08 dB) did not.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class REDCNN(nn.Module):
    def __init__(self, ch=96):
        super().__init__()
        k = dict(kernel_size=5, padding=2)
        self.c1 = nn.Conv2d(1, ch, **k)
        self.c2 = nn.Conv2d(ch, ch, **k)
        self.c3 = nn.Conv2d(ch, ch, **k)
        self.c4 = nn.Conv2d(ch, ch, **k)
        self.c5 = nn.Conv2d(ch, ch, **k)
        self.t1 = nn.ConvTranspose2d(ch, ch, **k)
        self.t2 = nn.ConvTranspose2d(ch, ch, **k)
        self.t3 = nn.ConvTranspose2d(ch, ch, **k)
        self.t4 = nn.ConvTranspose2d(ch, ch, **k)
        self.t5 = nn.ConvTranspose2d(ch, 1, **k)

    def forward(self, x):
        r0 = x
        h1 = F.relu(self.c1(x))
        h2 = F.relu(self.c2(h1))          # residual 2
        h3 = F.relu(self.c3(h2))
        h4 = F.relu(self.c4(h3))          # residual 3
        h5 = F.relu(self.c5(h4))
        d = F.relu(self.t1(h5) + h4)
        d = F.relu(self.t2(d))
        d = F.relu(self.t3(d) + h2)
        d = F.relu(self.t4(d))
        return self.t5(d) + r0


class EDCNN(nn.Module):
    def __init__(self, ch=32, depth=8):
        super().__init__()
        sob = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]],
                            [[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]],
                            [[[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]]],
                            [[[2., 1., 0.], [1., 0., -1.], [0., -1., -2.]]]])
        self.register_buffer("sobel", sob)
        self.gain = nn.Parameter(torch.ones(4))
        body, c_in = [], 1 + 4
        for i in range(depth):
            body.append(nn.Sequential(nn.Conv2d(c_in, ch, 3, padding=1), nn.ReLU()))
            c_in += ch                                    # dense concatenation
        self.body = nn.ModuleList(body)
        self.tail = nn.Sequential(nn.Conv2d(c_in, ch, 1), nn.ReLU(),
                                  nn.Conv2d(ch, 1, 1))

    def forward(self, x):
        e = F.conv2d(x, self.sobel, padding=1) * self.gain.view(1, -1, 1, 1)
        h = torch.cat([x, e], 1)
        for blk in self.body:
            h = torch.cat([h, blk(h)], 1)
        return self.tail(h) + x


class UNetDenoiser(nn.Module):
    def __init__(self, base=48):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(o, o, 3, padding=1), nn.ReLU())
        self.e1, self.e2, self.e3 = blk(1, base), blk(base, base * 2), blk(base * 2, base * 4)
        self.b = blk(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = blk(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = blk(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = blk(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        b = self.b(F.avg_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1) + x


BASELINES = {"RED-CNN": REDCNN, "EDCNN": EDCNN, "UNet": UNetDenoiser}
