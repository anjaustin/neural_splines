"""Does the spline premise hold for CONVOLUTION, where it failed for dense layers?

Fitting a spline to a trained fully-connected weight matrix gives ~99% relative
error -- no better than predicting zeros -- because the axes it smooths across
(neuron indices) are arbitrarily ordered. Conv kernels are different: the
spatial axes are real image space, and trained filters are often smooth.

This fits control points to a TRAINED 9x9 filter bank, in closed form (the
interpolation is linear in the control points, so this is the provable optimum,
not a search). If the premise holds for conv, the error should be far below the
0.99 benchmark from the dense case.
"""

import time
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
TE = DataLoader(datasets.MNIST(D, train=False, transform=tf), batch_size=1024)
K = 9


def cnn(k=K):
    p = k // 2
    return nn.Sequential(
        nn.Conv2d(1, 16, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, k, padding=p), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 10))


def evaluate(m):
    m.eval(); c = 0
    with torch.no_grad():
        for d, t in TE:
            c += m(d).argmax(1).eq(t).sum().item()
    return 100 * c / len(TE.dataset)


t0 = time.time()
torch.manual_seed(0)
net = cnn()
o = optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sch = optim.lr_scheduler.OneCycleLR(o, max_lr=2e-3, epochs=6, steps_per_epoch=len(TR))
for ep in range(6):
    net.train()
    for d, t in TR:
        o.zero_grad(); F.cross_entropy(net(d), t).backward(); o.step(); sch.step()
    print(f"  epoch {ep+1}/6 done ({time.time()-t0:.0f}s)", flush=True)
acc = evaluate(net)
print(f"\ntrained dense 9x9 CNN: {acc:.2f}%  "
      f"({sum(p.numel() for p in net.parameters()):,} params, {time.time()-t0:.0f}s)\n", flush=True)


def axis_op(n_cp, n_out):
    I = torch.eye(n_cp)
    return F.interpolate(I[None, None], size=(n_out, n_cp), mode="bicubic",
                         align_corners=True)[0, 0]


def fit(W, cp):
    """Closed-form optimal control points for every filter at once."""
    n, _, k, _ = W.shape
    Wf = W.reshape(-1, k, k)
    A = axis_op(cp, k)
    pA = torch.linalg.pinv(A)
    C = torch.einsum("pk,nkl->npl", pA, Wf)
    C = torch.einsum("npl,ql->npq", C, pA)
    R = torch.einsum("kp,npq->nkq", A, C)
    R = torch.einsum("nkq,lq->nkl", R, A)
    return (torch.norm(R - Wf) / torch.norm(Wf)).item()


print("closed-form spline reconstruction of the TRAINED filters")
print("(dense-layer benchmark from earlier experiments: ~0.99 = no better than zeros)\n")
print(f"{'layer':<22} {'filters':>9} {'cp':>4} {'stored/dense':>13} {'rel error':>10}")
for name, layer in [("conv1 (1->16, 9x9)", net[0]), ("conv2 (16->32, 9x9)", net[3])]:
    W = layer.weight.data
    for cp in (4, 5, 6, 7):
        err = fit(W, cp)
        ratio = (cp * cp) / (K * K)
        print(f"{name:<22} {W.shape[0]*W.shape[1]:>9} {cp:>4} {ratio:>12.2f}x {err:>10.4f}",
              flush=True)

# Control: the same fit against random filters of the same shape and scale.
print()
torch.manual_seed(1)
for name, layer in [("conv2 SHUFFLED taps", net[3])]:
    W = layer.weight.data
    perm = torch.randperm(K * K)
    Wp = W.reshape(-1, K * K)[:, perm].reshape(W.shape)
    print(f"{name:<22} {'':>9} {4:>4} {16/81:>12.2f}x {fit(Wp,4):>10.4f}   "
          f"<- destroys spatial structure, keeps the value distribution")
