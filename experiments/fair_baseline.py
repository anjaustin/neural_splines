"""Dense baselines under the SAME recipe the spline models get.

The ~97.8% figure quoted throughout came from plain Adam, 5 epochs, no
augmentation. Comparing a spline model trained with AdamW + OneCycle + 40
epochs + augmentation against that number would credit the recipe to the
architecture. These are the honest reference points.
"""

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


def mlp(hidden, bn=False):
    layers = [nn.Flatten(), nn.Linear(784, hidden)]
    if bn:
        layers.append(nn.BatchNorm1d(hidden))
    layers += [nn.ReLU(), nn.Linear(hidden, 10)]
    return nn.Sequential(*layers)


print(f"{'dense baseline':<52} {'params':>8}  {'acc':>6}", flush=True)
for label, h, bn, ep, aug in [
    ("784-256-10, 15ep, no aug", 256, False, 15, False),
    ("784-256-10 + BN, 40ep + aug", 256, True, 40, True),
    ("784-1024-10 + BN, 40ep + aug", 1024, True, 40, True),
]:
    n, a = run(lambda: mlp(h, bn), ep, aug)
    print(f"{label:<52} {n:>8,}  {a:>6.2f}%", flush=True)
