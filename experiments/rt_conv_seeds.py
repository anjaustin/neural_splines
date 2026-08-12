"""RED-TEAM: the conv comparison was single-seed, and the claims hinge on 0.1pp.

"spline 9x9 cp=5 reaches 99.27% against dense 9x9's 99.38% -- 0.11 points" and
"dense 3x3 99.14% vs spline cp=4 99.17%" are differences far smaller than
typical MNIST run-to-run spread. Three seeds each.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neural_splines import SplineConv2d

D = ("/private/tmp/claude-501/-Users-aaronjosserand-austin-Projects-neural-splines/"
     "d8eefa75-2215-4320-8042-972f2a2367bf/scratchpad/data")
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
TR = DataLoader(datasets.MNIST(D, train=True, transform=tf), batch_size=128, shuffle=True)
TE = DataLoader(datasets.MNIST(D, train=False, transform=tf), batch_size=1024)


def net(kind, k, cp=None, c1=16, c2=32):
    p = k // 2
    def conv(i, o):
        return nn.Conv2d(i, o, k, padding=p) if kind == "dense" \
            else SplineConv2d(i, o, k, cp_h=cp, padding=p)
    return nn.Sequential(conv(1, c1), nn.ReLU(), nn.MaxPool2d(2),
                         conv(c1, c2), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(c2 * 7 * 7, 10))


def run(make, seed, epochs=8):
    torch.manual_seed(seed)
    m = make()
    n = sum(p.numel() for p in m.parameters())
    o = optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.OneCycleLR(o, max_lr=2e-3, epochs=epochs, steps_per_epoch=len(TR))
    for _ in range(epochs):
        m.train()
        for d, t in TR:
            o.zero_grad(); F.cross_entropy(m(d), t).backward(); o.step(); sch.step()
    m.eval(); c = 0
    with torch.no_grad():
        for d, t in TE:
            c += m(d).argmax(1).eq(t).sum().item()
    return n, 100 * c / len(TE.dataset)


CFGS = [
    ("dense  3x3",      lambda: net("dense", 3)),
    ("dense  9x9",      lambda: net("dense", 9)),
    ("spline 9x9 cp=5", lambda: net("spline", 9, 5)),
]
print(f"{'configuration':<20} {'params':>8}  {'seed0':>7} {'seed1':>7} {'seed2':>7}"
      f"  {'mean':>7} {'spread':>7}", flush=True)
for label, mk in CFGS:
    accs = []
    for s in (0, 1, 2):
        n, a = run(mk, s)
        accs.append(a)
    print(f"{label:<20} {n:>8,}  " + " ".join(f"{a:>7.2f}" for a in accs) +
          f"  {sum(accs)/3:>7.2f} {max(accs)-min(accs):>7.2f}", flush=True)
