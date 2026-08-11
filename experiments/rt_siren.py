"""RED-TEAM 1: was the SIREN result an artifact of my hyperparameters?

Two ways I may have rigged it against SIREN:

  (a) undertrained -- 3000 steps may simply not be enough on a HARD target.
      The smooth positive control does not rule this out: smooth targets are
      easy for SIREN, so it validates the harness only in the easy regime.

  (b) handicapped bandwidth -- omega_0 controls SIREN's frequency prior. At
      w0=30 the network carries a LOW-FREQUENCY bias, i.e. the very smoothness
      assumption I criticised the spline for. Fitting near-white noise may
      need a much higher w0.

If error is still falling at 20k steps, or a higher w0 does much better, my
"coordinate networks do not help" conclusion is unsafe.
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
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
tr = DataLoader(datasets.MNIST(D, train=True, transform=tf), batch_size=128, shuffle=True)
torch.manual_seed(0)
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
opt = optim.Adam(net.parameters(), lr=1e-3)
for _ in range(3):
    net.train()
    for d, t in tr:
        opt.zero_grad(); F.cross_entropy(net(d), t).backward(); opt.step()
W = net[1].weight.data.clone()
m, n = W.shape
print(f"target: trained {m}x{n} layer, budget 16384 (spline reached 0.8909, SVD 0.6924)\n",
      flush=True)


class Siren(nn.Module):
    def __init__(self, h, w0, layers=3):
        super().__init__()
        self.w0 = w0
        dims = [2] + [h] * (layers - 1) + [1]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        with torch.no_grad():
            for k, l in enumerate(self.lins):
                fan = l.weight.shape[1]
                if k == 0:
                    l.weight.uniform_(-1 / fan, 1 / fan)
                else:
                    b = math.sqrt(6 / fan) / w0
                    l.weight.uniform_(-b, b)
                l.bias.zero_()

    def forward(self, x):
        for l in self.lins[:-1]:
            x = torch.sin(self.w0 * l(x))
        return self.lins[-1](x).squeeze(-1)


def fit(w0, budget=16384, steps=20000, lr=1e-3):
    h = max(hh for hh in range(2, 400)
            if sum(p.numel() for p in Siren(hh, w0).parameters()) <= budget)
    torch.manual_seed(0)
    s = Siren(h, w0)
    nparams = sum(p.numel() for p in s.parameters())
    gi, gj = torch.meshgrid(torch.linspace(-1, 1, m), torch.linspace(-1, 1, n), indexing="ij")
    coords = torch.stack([gi.reshape(-1), gj.reshape(-1)], -1)
    tgt = W.reshape(-1)
    o = optim.Adam(s.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(o, T_max=steps)
    N = coords.shape[0]
    marks = {}
    for st in range(1, steps + 1):
        idx = torch.randint(0, N, (16384,))
        o.zero_grad(); F.mse_loss(s(coords[idx]), tgt[idx]).backward(); o.step(); sch.step()
        if st in (1000, 3000, 6000, 10000, 20000):
            with torch.no_grad():
                pred = torch.cat([s(coords[k:k + 65536]) for k in range(0, N, 65536)])
            marks[st] = (torch.norm(pred.view(m, n) - W) / torch.norm(W)).item()
    return h, nparams, marks


print(f"{'w0':>6} {'hidden':>7} {'params':>7} " + " ".join(f"{k:>8}" for k in
      (1000, 3000, 6000, 10000, 20000)), flush=True)
for w0 in (30.0, 100.0, 300.0):
    h, npar, marks = fit(w0)
    row = " ".join(f"{marks[k]:>8.4f}" for k in (1000, 3000, 6000, 10000, 20000))
    print(f"{w0:>6.0f} {h:>7} {npar:>7} {row}", flush=True)
print("\ncolumns are training steps; falling values at 20000 == undertrained", flush=True)
