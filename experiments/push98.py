"""Final push: how close to 98%+ can a spline-parameterised layer get, and at
what parameter cost, under the SAME recipe the dense baselines got.

Honest reference points (AdamW + OneCycle, same data pipeline):
    784-256-10, 15ep, no aug     203,530 params   98.33%
    784-256-10 + BN, 40ep + aug  204,042 params   99.09%

Augmentation hurt at 15 epochs (-1.28) but that is the usual pattern -- it
needs a longer schedule to pay off, so it is only used with 40 here.
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
AUG = transforms.Compose([
    transforms.RandomAffine(degrees=8, translate=(0.10, 0.10), scale=(0.92, 1.08)),
    transforms.ToTensor(), NORM])
TEST = DataLoader(datasets.MNIST(D, train=False, transform=PLAIN), batch_size=1024)


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
    def __init__(self, coarse, fine=None, twod=False, hidden=256, bn=False):
        super().__init__()
        self.w1 = W(hidden, 784, coarse, fine, twod)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.bn = nn.BatchNorm1d(hidden) if bn else None
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = F.linear(x.view(x.size(0), -1), self.w1(), self.b1)
        if self.bn is not None:
            h = self.bn(h)
        return self.fc2(F.relu(h))


def run(make, epochs, aug, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    m = make()
    n = sum(p.numel() for p in m.parameters())
    tr = DataLoader(datasets.MNIST(D, train=True, transform=AUG if aug else PLAIN),
                    batch_size=128, shuffle=True)
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


CFGS = [
    ("single 1D 108x188, 40ep, NO aug",  dict(coarse=(108, 188)), 40, False),
    ("single 1D 108x188, 40ep +aug",     dict(coarse=(108, 188)), 40, True),
    ("single 1D 108x188 +BN, 40ep +aug", dict(coarse=(108, 188), bn=True), 40, True),
    ("single 1D 150x260 +BN, 40ep +aug", dict(coarse=(150, 260), bn=True), 40, True),
    ("single 1D 108x188 +BN h512, 40ep +aug",
     dict(coarse=(108, 188), bn=True, hidden=512), 40, True),
]
print(f"{'configuration':<42} {'params':>9} {'vs dense':>9} {'acc':>8}", flush=True)
for label, cfg, ep, aug in CFGS:
    n, a = run(lambda c=cfg: Net(**c), ep, aug)
    print(f"{label:<42} {n:>9,} {203530/n:>8.1f}x {a:>7.2f}%", flush=True)
