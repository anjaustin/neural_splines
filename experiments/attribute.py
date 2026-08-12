"""Is multi-resolution actually better, or is it just more parameters?

The isolation run showed multires at 98.10% -- but with 23,226 parameters
against the reference's 4,866. Almost 5x the budget. Every config here is
pinned to ~20,400 control points so the comparison is like-for-like:

  single 1D      one grid, raster input axis
  multires 1D    coarse + fine, raster input axis
  single 2D      one grid over (out, 28, 28)
  multires 2D    coarse + fine over (out, 28, 28)

If single 1D matches multires 1D, "multi-resolution" is a misattribution and
the finding is simply "spend more parameters".
"""

import math
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
TEST = DataLoader(datasets.MNIST(D, train=False, transform=PLAIN), batch_size=1024)
TRAIN = DataLoader(datasets.MNIST(D, train=True, transform=PLAIN), batch_size=128, shuffle=True)


class W(nn.Module):
    def __init__(self, out_f, in_f, coarse, fine=None, twod=False):
        super().__init__()
        self.out_f, self.in_f, self.twod = out_f, in_f, twod
        self.side = int(round(math.sqrt(in_f)))
        self.c = nn.Parameter(torch.randn(*coarse) * 0.02)
        self.f = nn.Parameter(torch.randn(*fine) * 0.005) if fine else None

    def _up(self, cp):
        if self.twod:
            return F.interpolate(cp[None, None], size=(self.out_f, self.side, self.side),
                                 mode="trilinear", align_corners=True)[0, 0].reshape(
                                     self.out_f, self.in_f)
        return F.interpolate(cp[None, None], size=(self.out_f, self.in_f),
                             mode="bicubic", align_corners=True)[0, 0]

    def forward(self):
        w = self._up(self.c)
        return w if self.f is None else w + self._up(self.f)


class Net(nn.Module):
    def __init__(self, coarse, fine=None, twod=False, hidden=256):
        super().__init__()
        self.w1 = W(hidden, 784, coarse, fine, twod)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(F.linear(x, self.w1(), self.b1)))


def run(make, epochs=15, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    m = make()
    n = sum(p.numel() for p in m.parameters())
    cp = sum(p.numel() for nm, p in m.named_parameters() if nm.startswith("w1."))
    o = optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.OneCycleLR(o, max_lr=lr, epochs=epochs, steps_per_epoch=len(TRAIN))
    for _ in range(epochs):
        m.train()
        for d, t in TRAIN:
            o.zero_grad(); F.cross_entropy(m(d), t).backward(); o.step(); sch.step()
    m.eval(); c = 0
    with torch.no_grad():
        for d, t in TEST:
            c += m(d).argmax(1).eq(t).sum().item()
    return cp, n, 100 * c / len(TEST.dataset)


CFGS = [
    ("reference: single 1D 34x60",        dict(coarse=(34, 60))),
    ("single 1D 108x188",                 dict(coarse=(108, 188))),
    ("multires 1D 34x60 + 102x180",       dict(coarse=(34, 60), fine=(102, 180))),
    ("single 2D 60x18x18",                dict(coarse=(60, 18, 18), twod=True)),
    ("multires 2D 24x8x8 + 72x16x16",     dict(coarse=(24, 8, 8), fine=(72, 16, 16), twod=True)),
]
print(f"{'configuration':<40} {'cp params':>10} {'total':>9} {'acc':>8}", flush=True)
for label, cfg in CFGS:
    cp, n, a = run(lambda c=cfg: Net(**c))
    print(f"{label:<40} {cp:>10,} {n:>9,} {a:>7.2f}%", flush=True)
