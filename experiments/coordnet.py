"""Can a coordinate network represent a trained weight matrix better than a spline?

Four representations are fitted to the same target matrices at matched
parameter budgets:

  spline  - fixed tensor-product cubic basis, linear readout (what the repo does)
  svd     - optimal adaptive LINEAR basis (Eckart-Young lower bound for its class)
  siren   - MLP with sinusoidal activations over (i, j) coordinates
  cfc     - CfC-style gated coordinate network (see caveat in the report)

Three targets, so that a null result is interpretable:

  trained - a real trained Linear(784, 256) weight        <- the actual question
  smooth  - a low-frequency analytic surface              <- POSITIVE control
  random  - i.i.d. Gaussian                               <- entropy ceiling

Without the positive control, "everything failed" cannot be distinguished
from "the harness is broken".
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEV = "cpu"
torch.manual_seed(0)
BUDGETS = [256, 1024, 4096, 16384]
STEPS = 3000
CHUNK = 16384


# ---------------------------------------------------------------- targets
def trained_weight():
    """Train a real MLP on MNIST and return its first-layer weight."""
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    D = ("/private/tmp/claude-501/-Users-aaronjosserand-austin-Projects-neural-splines/"
         "d8eefa75-2215-4320-8042-972f2a2367bf/scratchpad/data")
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    tr = DataLoader(datasets.MNIST(D, train=True, transform=tf), batch_size=128, shuffle=True)
    te = DataLoader(datasets.MNIST(D, train=False, transform=tf), batch_size=512)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    opt = optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(3):
        net.train()
        for d, t in tr:
            opt.zero_grad(); crit(net(d), t).backward(); opt.step()
    net.eval(); c = 0
    with torch.no_grad():
        for d, t in te:
            c += net(d).argmax(1).eq(t).sum().item()
    return net, tr, te, 100 * c / len(te.dataset)


def make_targets(W_trained):
    m, n = W_trained.shape
    ii = torch.linspace(-1, 1, m).view(-1, 1)
    jj = torch.linspace(-1, 1, n).view(1, -1)
    smooth = (torch.sin(2.5 * ii) * torch.cos(1.8 * jj)
              + 0.4 * torch.sin(4.0 * ii * jj))
    smooth = smooth * (W_trained.std() / smooth.std())
    torch.manual_seed(1)
    rand = torch.randn(m, n) * W_trained.std()
    return {"trained": W_trained, "smooth": smooth, "random": rand}


# ---------------------------------------------------------------- baselines
def fit_spline(W, budget):
    """Best-possible control points for a p x q grid, via LBFGS least squares."""
    m, n = W.shape
    p = max(4, int(round(math.sqrt(budget * m / n))))
    q = max(4, int(round(budget / max(p, 1))))
    p, q = min(p, m), min(q, n)
    cp = torch.zeros(p, q, requires_grad=True)
    with torch.no_grad():  # sensible init: subsample the target
        ri = torch.linspace(0, m - 1, p).long(); ci = torch.linspace(0, n - 1, q).long()
        cp.copy_(W[ri][:, ci])
    opt = optim.LBFGS([cp], lr=0.5, max_iter=400, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        R = F.interpolate(cp[None, None], size=(m, n), mode="bicubic", align_corners=True)[0, 0]
        loss = torch.norm(R - W); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        R = F.interpolate(cp[None, None], size=(m, n), mode="bicubic", align_corners=True)[0, 0]
    return R.detach(), p * q


def fit_svd(W, budget):
    """Optimal rank-r approximation. Eckart-Young: no linear method beats this."""
    m, n = W.shape
    r = max(1, budget // (m + n))
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r], r * (m + n)


# ---------------------------------------------------------------- coord nets
class Siren(nn.Module):
    def __init__(self, h, layers=3, w0=30.0):
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
        for k, l in enumerate(self.lins[:-1]):
            x = torch.sin(self.w0 * l(x)) if k == 0 else torch.sin(self.w0 * l(x))
        return self.lins[-1](x).squeeze(-1)


class CfCCoord(nn.Module):
    """CfC-style closed-form gating, adapted to coordinate regression.

    Mirrors the CfC closed-form shape
        out = sigmoid(-f(z) * t) * g(z) + (1 - sigmoid(-f(z) * t)) * h(z)
    with one coordinate playing the role of `t`. This is NOT a faithful CfC:
    a real CfC is a continuous-time sequence model with recurrent state, and
    there is no time axis in a static weight matrix. It is included to test
    the *gated closed-form interpolation* inductive bias, which is the part
    that could plausibly transfer.
    """

    def __init__(self, h):
        super().__init__()
        self.emb = nn.Linear(2, h)
        self.f = nn.Linear(h, h); self.g = nn.Linear(h, h); self.h = nn.Linear(h, h)
        self.out = nn.Linear(h, 1)

    def forward(self, x):
        t = x[:, :1]
        z = torch.sin(30.0 * self.emb(x))
        gate = torch.sigmoid(-self.f(z) * t)
        y = gate * torch.tanh(self.g(z)) + (1 - gate) * torch.tanh(self.h(z))
        return self.out(y).squeeze(-1)


def size_to_budget(ctor, budget):
    """Largest hidden width whose parameter count fits the budget."""
    best = None
    for h in range(2, 400):
        p = sum(q.numel() for q in ctor(h).parameters())
        if p <= budget:
            best = (h, p)
        else:
            break
    return best


def fit_coordnet(W, budget, kind, steps=STEPS, seed=0):
    m, n = W.shape
    ctor = (lambda h: Siren(h)) if kind == "siren" else (lambda h: CfCCoord(h))
    sized = size_to_budget(ctor, budget)
    if sized is None:
        return None, 0
    h, nparams = sized
    torch.manual_seed(seed)
    net = ctor(h)
    gi, gj = torch.meshgrid(torch.linspace(-1, 1, m), torch.linspace(-1, 1, n), indexing="ij")
    coords = torch.stack([gi.reshape(-1), gj.reshape(-1)], -1)
    target = W.reshape(-1)
    opt = optim.Adam(net.parameters(), lr=3e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    N = coords.shape[0]
    for s in range(steps):
        idx = torch.randint(0, N, (min(CHUNK, N),))
        opt.zero_grad()
        loss = F.mse_loss(net(coords[idx]), target[idx])
        loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        pred = torch.cat([net(coords[k:k + 65536]) for k in range(0, N, 65536)])
    return pred.view(m, n), nparams


# ---------------------------------------------------------------- run
def rel(R, W):
    return (torch.norm(R - W) / torch.norm(W)).item()


def main():
    t0 = time.time()
    net, tr, te, acc = trained_weight()
    W = net[1].weight.data.clone()
    print(f"trained dense baseline: {acc:.2f}%  (target = its 256x784 first layer)\n", flush=True)
    targets = make_targets(W)

    results = {}
    for tname, T in targets.items():
        print(f"=== target: {tname} ===", flush=True)
        print(f"{'budget':>7} {'spline':>9} {'svd':>9} {'siren':>9} {'cfc':>9}", flush=True)
        for b in BUDGETS:
            row = {}
            R, _ = fit_spline(T, b);  row["spline"] = rel(R, T)
            R2, _ = fit_svd(T, b);    row["svd"] = rel(R2, T)
            for kind in ("siren", "cfc"):
                Rk, np_ = fit_coordnet(T, b, kind)
                row[kind] = rel(Rk, T) if Rk is not None else float("nan")
                if tname == "trained" and kind == "siren":
                    results[(b, "siren_recon")] = Rk
            print(f"{b:>7} {row['spline']:>9.4f} {row['svd']:>9.4f} "
                  f"{row['siren']:>9.4f} {row['cfc']:>9.4f}", flush=True)
            results[(tname, b)] = row
        print(flush=True)

    # Functional metric: does reconstruction error even predict accuracy?
    print("=== functional check: substitute the reconstruction back into the net ===",
          flush=True)
    print(f"{'budget':>7} {'method':>8} {'rel_err':>9} {'MNIST acc':>10}", flush=True)

    def acc_with(Wsub):
        orig = net[1].weight.data.clone()
        net[1].weight.data = Wsub
        net.eval(); c = 0
        with torch.no_grad():
            for d, t in te:
                c += net(d).argmax(1).eq(t).sum().item()
        net[1].weight.data = orig
        return 100 * c / len(te.dataset)

    for b in BUDGETS:
        for kind, fn in (("spline", fit_spline), ("svd", fit_svd)):
            R, _ = fn(W, b)
            print(f"{b:>7} {kind:>8} {rel(R,W):>9.4f} {acc_with(R):>9.2f}%", flush=True)
        Rk = results.get((b, "siren_recon"))
        if Rk is not None:
            print(f"{b:>7} {'siren':>8} {rel(Rk,W):>9.4f} {acc_with(Rk):>9.2f}%", flush=True)
    print(f"\n(random-chance accuracy is ~9.8%; elapsed {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
