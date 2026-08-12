"""On-device footprint of the configurations this repo actually recommends.

Four distinct things get conflated under "footprint", and they move in
different directions here:

  1. checkpoint    what you ship / download
  2. resident      parameter bytes held in RAM once loaded
  3. peak forward  transient allocation during one inference
  4. runtime       the interpreter and framework the model needs to run at all

(4) is usually the elephant and is measured last.
"""

import io
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from neural_splines import SplineConv2d, SplineLinear


def ckpt_bytes(m: nn.Module) -> int:
    b = io.BytesIO()
    torch.save(m.state_dict(), b)
    return b.tell()


def resident_bytes(m: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in m.parameters()) + \
           sum(b.numel() * b.element_size() for b in m.buffers())


def peak_forward(m: nn.Module, x: torch.Tensor) -> int:
    m.eval()
    with torch.no_grad():
        m(x)  # warm any lazy caches
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as p:
            m(x)
    return sum(e.cpu_memory_usage for e in p.key_averages() if e.cpu_memory_usage > 0)


def latency_us(m: nn.Module, x: torch.Tensor, n: int = 200) -> float:
    m.eval()
    with torch.no_grad():
        for _ in range(20):
            m(x)
        t = time.perf_counter()
        for _ in range(n):
            m(x)
    return (time.perf_counter() - t) / n * 1e6


class SplineNet(nn.Module):
    """The 98.56% MLP: SplineLinear(784,256,108,188) + dense 10-way head."""

    def __init__(self, separable: bool = False):
        super().__init__()
        self.s1 = SplineLinear(784, 256, 108, 188, cp_bias=256)
        self.s1.separable_forward = separable
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.s1(x.view(x.size(0), -1))))


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x.view(x.size(0), -1))))


def cnn(spline: bool):
    k, cp = (9, 5) if spline else (3, None)
    def c(i, o):
        return SplineConv2d(i, o, 9, cp_h=5, padding=4) if spline \
            else nn.Conv2d(i, o, 3, padding=1)
    return nn.Sequential(c(1, 16), nn.ReLU(), nn.MaxPool2d(2),
                         c(16, 32), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(32 * 7 * 7, 10))


torch.manual_seed(0)
xm = torch.randn(1, 784)
xc = torch.randn(1, 1, 28, 28)

sp = SplineNet(separable=False)
sp_sep = SplineNet(separable=True)
sp_sep.load_state_dict(sp.state_dict())
dn = DenseNet()
# densified spline: bake the interpolated weights into an nn.Linear
dn_from_sp = DenseNet()
with torch.no_grad():
    lin = sp.s1.to_dense_linear()
    dn_from_sp.fc1.weight.copy_(lin.weight); dn_from_sp.fc1.bias.copy_(lin.bias)
    dn_from_sp.fc2.load_state_dict(sp.fc2.state_dict())

scnn, dcnn = cnn(True), cnn(False)
scnn_dense = nn.Sequential(*[m.to_dense_conv() if isinstance(m, SplineConv2d) else m
                             for m in scnn])

ROWS = [
    ("MLP  spline 108x188  (98.6%)",            sp,          xm),
    ("MLP  spline, separable_forward",          sp_sep,      xm),
    ("MLP  spline, densified at load",          dn_from_sp,  xm),
    ("MLP  dense 784-256-10 (99.1%)",           dn,          xm),
    ("CNN  spline 9x9 cp=5 (99.3%)",            scnn,        xc),
    ("CNN  spline, densified at load",          scnn_dense,  xc),
    ("CNN  dense 3x3       (99.1%)",            dcnn,        xc),
]

print(f"{'configuration':<34} {'checkpoint':>11} {'resident':>10} "
      f"{'peak fwd':>10} {'TOTAL RAM':>11} {'latency':>9}")
print("-" * 90)
for label, m, x in ROWS:
    pk = peak_forward(m, x)          # run FIRST so lazy caches are populated
    res = resident_bytes(m)          # ...then measure what is actually held
    print(f"{label:<34} {ckpt_bytes(m):>10,}B {res:>9,}B "
          f"{pk:>9,}B {res+pk:>10,}B {latency_us(m, x):>8.1f}us", flush=True)

print("\n--- the runtime that has to be there too ---")
import torch as _t
tpath = os.path.dirname(_t.__file__)
total = sum(os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(tpath) for f in fs
            if os.path.exists(os.path.join(r, f)))
print(f"  installed torch package: {total/1e6:>10,.1f} MB   ({tpath})")
print(f"  smallest model above:    {min(ckpt_bytes(m) for _,m,_ in ROWS)/1e6:>10,.3f} MB")
