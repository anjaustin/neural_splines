"""Is a cheap large spline kernel actually better than a standard 3x3 conv?

The compression claim only matters if it beats the obvious alternative. A 3x3
dense conv is the thing practitioners would reach for, and it is already cheap,
so it is the baseline that counts -- not the 9x9 dense conv, which nobody would
choose.

Same architecture throughout: conv -> ReLU -> pool -> conv -> ReLU -> pool ->
linear head. Same recipe: AdamW + OneCycle, 8 epochs, no augmentation.
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
        if kind == "dense":
            return nn.Conv2d(i, o, k, padding=p)
        return SplineConv2d(i, o, k, cp_h=cp, padding=p)
    return nn.Sequential(
        conv(1, c1), nn.ReLU(), nn.MaxPool2d(2),
        conv(c1, c2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(c2 * 7 * 7, 10))


def run(make, epochs=8, seed=0):
    torch.manual_seed(seed)
    m = make()
    n = sum(p.numel() for p in m.parameters())
    conv_n = sum(mod.weight.numel() if isinstance(mod, nn.Conv2d) else mod.control_points.numel()
                 for mod in m.modules() if isinstance(mod, (nn.Conv2d, SplineConv2d)))
    o = optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.OneCycleLR(o, max_lr=2e-3, epochs=epochs, steps_per_epoch=len(TR))
    t0 = time.time()
    for _ in range(epochs):
        m.train()
        for d, t in TR:
            o.zero_grad(); F.cross_entropy(m(d), t).backward(); o.step(); sch.step()
    m.eval(); c = 0
    with torch.no_grad():
        for d, t in TE:
            c += m(d).argmax(1).eq(t).sum().item()
    return n, conv_n, 100 * c / len(TE.dataset), time.time() - t0


CFGS = [
    ("dense  3x3        (the real baseline)", lambda: net("dense", 3)),
    ("dense  9x9",                            lambda: net("dense", 9)),
    ("spline 9x9  cp=4",                      lambda: net("spline", 9, 4)),
    ("spline 9x9  cp=5",                      lambda: net("spline", 9, 5)),
    ("spline 13x13 cp=4",                     lambda: net("spline", 13, 4)),
    ("dense 13x13",                           lambda: net("dense", 13)),
]
print(f"{'configuration':<40} {'total':>8} {'conv wts':>9} {'acc':>8} {'train s':>8}",
      flush=True)
for label, mk in CFGS:
    n, cn, a, t = run(mk)
    print(f"{label:<40} {n:>8,} {cn:>9,} {a:>7.2f}% {t:>7.0f}s", flush=True)
