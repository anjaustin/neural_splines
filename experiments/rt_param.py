"""RED-TEAM 4: is the README's "plateau in the low-to-mid 80s" real, or an
artifact of SplineMLP's square-grid parameterisation?

RT3 produced 94.76% from a spline-parameterised model at ~3k parameters,
against the ~85.7% ceiling the README reports for SplineMLP. Four things
differ at once, so they are separated here at matched budgets:

  A  repo default        square cp x cp grids, spline-interpolated biases
  B  A + full biases     biases stop being interpolated
  C  B + wide L1 grid    grid aspect follows the matrix (256x784 is not square)
  D  C + full-rank head  L2 grid gets one row PER CLASS (no smoothing across
                         the 10 classes, whose ordering is arbitrary)

If D >> A at equal parameter count, the README's plateau figure understates
what the representation can do and must be corrected.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

D_ = ("/private/tmp/claude-501/-Users-aaronjosserand-austin-Projects-neural-splines/"
      "d8eefa75-2215-4320-8042-972f2a2367bf/scratchpad/data")
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
tr = DataLoader(datasets.MNIST(D_, train=True, transform=tf), batch_size=128, shuffle=True)
te = DataLoader(datasets.MNIST(D_, train=False, transform=tf), batch_size=512)
IN, HID, OUT = 784, 256, 10


class Cfg(nn.Module):
    def __init__(self, g1, g2, full_bias):
        super().__init__()
        self.g1, self.g2 = g1, g2
        self.cp1 = nn.Parameter(torch.randn(*g1) * 0.02)
        self.cp2 = nn.Parameter(torch.randn(*g2) * 0.02)
        self.full_bias = full_bias
        self.b1 = nn.Parameter(torch.zeros(HID if full_bias else g1[0]))
        self.b2 = nn.Parameter(torch.zeros(OUT if full_bias else g2[0]))

    @staticmethod
    def _up(cp, size):
        return F.interpolate(cp[None, None], size=size, mode="bicubic",
                             align_corners=True)[0, 0]

    def _bias(self, b, out):
        if b.shape[0] == out:
            return b
        return F.interpolate(b[None, None], size=out, mode="linear",
                             align_corners=True)[0, 0]

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(F.linear(x, self._up(self.cp1, (HID, IN)), self._bias(self.b1, HID)))
        return F.linear(h, self._up(self.cp2, (OUT, HID)), self._bias(self.b2, OUT))


def run(cfg, epochs=5, seed=0):
    torch.manual_seed(seed)
    m = cfg()
    npar = sum(p.numel() for p in m.parameters())
    o = optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(epochs):
        m.train()
        for d, t in tr:
            o.zero_grad(); F.cross_entropy(m(d), t).backward(); o.step()
    m.eval(); c = 0
    with torch.no_grad():
        for d, t in te:
            c += m(d).argmax(1).eq(t).sum().item()
    return npar, 100 * c / len(te.dataset)


CFGS = [
    ("A repo default (32x32, 32x32, spline bias)", lambda: Cfg((32, 32), (32, 32), False)),
    ("B A + full biases",                          lambda: Cfg((32, 32), (32, 32), True)),
    ("C B + wide L1 grid (16x128)",                lambda: Cfg((16, 128), (32, 32), True)),
    ("D C + per-class head (10x64)",               lambda: Cfg((16, 128), (10, 64), True)),
    ("E D with square-ish L1 (32x64)",             lambda: Cfg((32, 64), (10, 64), True)),
]
print(f"{'config':>44} {'params':>8} {'acc (5 ep)':>11}", flush=True)
for name, c in CFGS:
    npar, acc = run(c)
    print(f"{name:>44} {npar:>8,} {acc:>10.2f}%", flush=True)
print("\ndense baseline is 203,530 params / ~97.8%", flush=True)
