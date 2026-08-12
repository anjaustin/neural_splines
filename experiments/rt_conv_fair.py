"""RED-TEAM: was "the premise holds for conv but not for dense" apples-to-apples?

It may not have been. The conv figures were taken at 16/81 = 5x compression
(a 4x4 grid per 9x9 filter). The dense figure it was compared against, 0.99,
came from a 4,096-point grid on a 200,704-entry matrix -- 49x compression.
Different regimes entirely, and 5x is far easier than 49x.

Here both are swept across the SAME compression ratios, in closed form.
Also checked:
  - a random-filter control, the true null for the shuffle test
  - whether bicubic interpolation shrinks the effective init scale
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
TR = DataLoader(datasets.MNIST(D, train=True, transform=tf), batch_size=128, shuffle=True)
K = 9


def axis_op(n_cp, n_out):
    I = torch.eye(n_cp)
    return F.interpolate(I[None, None], size=(n_out, n_cp), mode="bicubic",
                         align_corners=True)[0, 0]


def fit_err(W2d, cp_h, cp_w):
    """Closed-form optimal control points for a single 2-D matrix."""
    m, n = W2d.shape
    A, B = axis_op(cp_h, m), axis_op(cp_w, n)
    C = torch.linalg.pinv(A) @ W2d @ torch.linalg.pinv(B).T
    return (torch.norm(A @ C @ B.T - W2d) / torch.norm(W2d)).item()


def fit_err_stack(W4d, cp):
    """Closed-form fit for a stack of square filters, each with its own grid."""
    n, _, k, _ = W4d.shape
    Wf = W4d.reshape(-1, k, k)
    A = axis_op(cp, k)
    pA = torch.linalg.pinv(A)
    C = torch.einsum("pk,nkl->npl", pA, Wf)
    C = torch.einsum("npl,ql->npq", C, pA)
    R = torch.einsum("kp,npq->nkq", A, C)
    R = torch.einsum("nkq,lq->nkl", R, A)
    return (torch.norm(R - Wf) / torch.norm(Wf)).item()


# ---- train both a dense MLP and a 9x9 CNN on the same data ----
torch.manual_seed(0)
mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
o = optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=1e-4)
s = optim.lr_scheduler.OneCycleLR(o, max_lr=2e-3, epochs=4, steps_per_epoch=len(TR))
for _ in range(4):
    mlp.train()
    for d, t in TR:
        o.zero_grad(); F.cross_entropy(mlp(d), t).backward(); o.step(); s.step()

torch.manual_seed(0)
cnn = nn.Sequential(nn.Conv2d(1, 16, K, padding=4), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, K, padding=4), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(), nn.Linear(32 * 7 * 7, 10))
o = optim.AdamW(cnn.parameters(), lr=2e-3, weight_decay=1e-4)
s = optim.lr_scheduler.OneCycleLR(o, max_lr=2e-3, epochs=4, steps_per_epoch=len(TR))
for _ in range(4):
    cnn.train()
    for d, t in TR:
        o.zero_grad(); F.cross_entropy(cnn(d), t).backward(); o.step(); s.step()
print("both models trained\n", flush=True)

Wmlp = mlp[1].weight.data
Wconv = cnn[3].weight.data   # (32,16,9,9)

print("Reconstruction error at MATCHED compression ratios (closed form)\n")
print(f"{'compression':>12} {'dense 256x784':>15} {'conv 9x9 stack':>16}")
for cp in (4, 5, 6, 7):
    ratio = (K * K) / (cp * cp)                    # conv compression at this grid
    conv_err = fit_err_stack(Wconv, cp)
    # dense grid giving the SAME compression on its 200,704 entries
    budget = int(Wmlp.numel() / ratio)
    r = math.sqrt(Wmlp.shape[0] / Wmlp.shape[1])
    ch = max(4, min(int(round(math.sqrt(budget * r))), Wmlp.shape[0]))
    cw = max(4, min(budget // ch, Wmlp.shape[1]))
    dense_err = fit_err(Wmlp, ch, cw)
    print(f"{ratio:>11.2f}x {dense_err:>15.4f} {conv_err:>16.4f}"
          f"   (dense grid {ch}x{cw})", flush=True)

print("\nControls for the shuffle test (conv, cp=4, 5.06x):")
print(f"  trained filters                    {fit_err_stack(Wconv,4):.4f}")
perm = torch.randperm(K * K)
Wp = Wconv.reshape(-1, K * K)[:, perm].reshape(Wconv.shape)
print(f"  same filters, spatial taps shuffled {fit_err_stack(Wp,4):.4f}")
torch.manual_seed(1)
Wr = torch.randn_like(Wconv) * Wconv.std()
print(f"  random filters, matched std         {fit_err_stack(Wr,4):.4f}   <- true null")

print("\nInit scale: does bicubic interpolation shrink the effective kernel?")
from neural_splines import SplineConv2d
torch.manual_seed(0)
sc = SplineConv2d(16, 32, K, cp_h=5, padding=4)
ref = nn.Conv2d(16, 32, K, padding=4)
print(f"  SplineConv2d interpolated kernel std {sc._interpolate_kernel().std().item():.5f}")
print(f"  nn.Conv2d default init std           {ref.weight.std().item():.5f}")
print(f"  ratio                                "
      f"{(sc._interpolate_kernel().std()/ref.weight.std()).item():.3f}")
