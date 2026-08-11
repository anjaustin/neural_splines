"""Atomic decomposition: what actually moves spline-MLP accuracy toward 98%?

Each factor is isolated so its contribution can be attributed, rather than
reported as one tuned number. Factors:

  grid2d      interpolate over (out, 28, 28) instead of (out, 784). The 784
              axis is RASTER order, so pixels one row apart sit 28 indices
              apart and the spline cannot express 2D locality. Prime suspect.
  multires    W = interp(coarse) + interp(fine): low and high frequency at once
  bias        uncompressed bias vs cp_bias = cp_h
  hidden      hidden width (cheap for a spline: cost grows with cp_h, not width)
  bn          BatchNorm after layer 1
  head        dense output layer (2,570 params) vs spline output layer
  train       epochs + cosine schedule
  aug         random small affine jitter

Run: python atomics.py [ladder|isolate]
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

D = ("/private/tmp/claude-501/-Users-aaronjosserand-austin-Projects-neural-splines/"
     "d8eefa75-2215-4320-8042-972f2a2367bf/scratchpad/data")
NORM = transforms.Normalize((0.1307,), (0.3081,))
PLAIN = transforms.Compose([transforms.ToTensor(), NORM])
AUG = transforms.Compose([
    transforms.RandomAffine(degrees=8, translate=(0.10, 0.10), scale=(0.92, 1.08)),
    transforms.ToTensor(), NORM])
TEST = DataLoader(datasets.MNIST(D, train=False, transform=PLAIN), batch_size=1024)


def loader(aug, bs=128):
    return DataLoader(datasets.MNIST(D, train=True, transform=AUG if aug else PLAIN),
                      batch_size=bs, shuffle=True)


class SplineW(nn.Module):
    """Generates one weight matrix from control points. 1D-raster or 2D-aware."""

    def __init__(self, out_f, in_f, grid, twod=False, multires=False):
        super().__init__()
        self.out_f, self.in_f, self.twod = out_f, in_f, twod
        self.side = int(round(math.sqrt(in_f))) if twod else None
        self.cp = nn.Parameter(torch.randn(*grid) * 0.02)
        self.cp_fine = None
        if multires:
            fine = tuple(min(g * 3, d) for g, d in
                         zip(grid, (out_f,) + ((self.side, self.side) if twod else (in_f,))))
            self.cp_fine = nn.Parameter(torch.randn(*fine) * 0.005)

    def _up(self, cp):
        if self.twod:
            w = F.interpolate(cp[None, None], size=(self.out_f, self.side, self.side),
                              mode="trilinear", align_corners=True)[0, 0]
            return w.reshape(self.out_f, self.in_f)
        return F.interpolate(cp[None, None], size=(self.out_f, self.in_f),
                             mode="bicubic", align_corners=True)[0, 0]

    def forward(self):
        w = self._up(self.cp)
        if self.cp_fine is not None:
            w = w + self._up(self.cp_fine)
        return w


class Net(nn.Module):
    def __init__(self, grid1, hidden=256, twod=False, multires=False, full_bias=True,
                 bn=False, dense_head=True, grid2=None):
        super().__init__()
        self.w1 = SplineW(hidden, 784, grid1, twod=twod, multires=multires)
        self.b1 = nn.Parameter(torch.zeros(hidden if full_bias else grid1[0]))
        self.full_bias = full_bias
        self.bn = nn.BatchNorm1d(hidden) if bn else None
        self.dense_head = dense_head
        if dense_head:
            self.fc2 = nn.Linear(hidden, 10)
        else:
            self.w2 = SplineW(10, hidden, grid2 or (10, 64))
            self.b2 = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        b = self.b1 if self.full_bias else F.interpolate(
            self.b1[None, None], size=self.w1.out_f, mode="linear", align_corners=True)[0, 0]
        h = F.linear(x, self.w1(), b)
        if self.bn is not None:
            h = self.bn(h)
        h = F.relu(h)
        return self.fc2(h) if self.dense_head else F.linear(h, self.w2(), self.b2)


def run(make, epochs, aug=False, seed=0, lr=2e-3):
    torch.manual_seed(seed)
    m = make()
    n = sum(p.numel() for p in m.parameters())
    tr = loader(aug)
    o = optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.OneCycleLR(o, max_lr=lr, epochs=epochs, steps_per_epoch=len(tr))
    for _ in range(epochs):
        m.train()
        for d, t in tr:
            o.zero_grad(); F.cross_entropy(m(d), t).backward(); o.step(); sch.step()
    m.eval(); c = 0
    with torch.no_grad():
        for d, t in TEST:
            c += m(d).argmax(1).eq(t).sum().item()
    return n, 100 * c / len(TEST.dataset)


def show(label, n, acc):
    print(f"{label:<52} {n:>8,}  {acc:>6.2f}%", flush=True)


def isolate():
    """One factor at a time, from a fixed reference point."""
    print("ISOLATION: change exactly one thing from the reference\n")
    print(f"{'configuration':<52} {'params':>8}  {'acc':>6}")
    ref = dict(grid1=(34, 60), hidden=256, twod=False, multires=False,
               full_bias=True, bn=False, dense_head=True)
    show("reference (1D raster grid 34x60, dense head, 15ep)", *run(lambda: Net(**ref), 15))

    # 2D-aware grid at MATCHED parameter count: 34x8x8 = 2176 vs 34x60 = 2040
    v = dict(ref, grid1=(34, 8, 8), twod=True)
    show("  + 2D grid over (out,28,28)  [34x8x8]", *run(lambda: Net(**v), 15))
    v = dict(ref, grid1=(24, 9, 9), twod=True)
    show("  + 2D grid, squarer          [24x9x9]", *run(lambda: Net(**v), 15))

    for label, over in [
        ("  + multi-resolution (coarse + fine)", dict(multires=True)),
        ("  + BatchNorm", dict(bn=True)),
        ("  + hidden 1024", dict(hidden=1024)),
        ("  + spline head (no dense fc2)", dict(dense_head=False)),
        ("  + compressed bias (cp_bias=cp_h)", dict(full_bias=False)),
    ]:
        show(label, *run(lambda o=over: Net(**dict(ref, **o)), 15))
    show("  + augmentation", *run(lambda: Net(**ref), 15, aug=True))


def ladder():
    """Stack the factors that helped, and push for 98%."""
    print("\nLADDER: accumulate the winners\n")
    print(f"{'configuration':<52} {'params':>8}  {'acc':>6}")
    steps = [
        ("2D grid 34x8x8, dense head, 15ep",
         dict(grid1=(34, 8, 8), twod=True, hidden=256), 15, False),
        ("+ hidden 1024",
         dict(grid1=(34, 8, 8), twod=True, hidden=1024), 15, False),
        ("+ BatchNorm",
         dict(grid1=(34, 8, 8), twod=True, hidden=1024, bn=True), 15, False),
        ("+ multi-resolution",
         dict(grid1=(34, 8, 8), twod=True, hidden=1024, bn=True, multires=True), 15, False),
        ("+ 40 epochs",
         dict(grid1=(34, 8, 8), twod=True, hidden=1024, bn=True, multires=True), 40, False),
        ("+ augmentation",
         dict(grid1=(34, 8, 8), twod=True, hidden=1024, bn=True, multires=True), 40, True),
    ]
    for label, cfg, ep, aug in steps:
        show(label, *run(lambda c=cfg: Net(**c), ep, aug=aug))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "isolate"
    print("MNIST, AdamW + OneCycle, seed 0. Dense 784-256-10 baseline: "
          "203,530 params / ~97.8%\n", flush=True)
    if what in ("isolate", "both"):
        isolate()
    if what in ("ladder", "both"):
        ladder()
