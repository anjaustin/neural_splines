"""Is the converter's problem the BASIS, or the OBJECTIVE?

Same representation (a spline control-point grid), same parameter budget,
same target network. Only the fitting objective changes:

  A. frobenius  - minimize ||W_hat - W||_F            <- what the converter does
  B. distill    - match the teacher's LOGITS on data  <- what compression normally does
  C. labels     - train the control points on labels directly

If A collapses and B/C do not, the representation was never the binding
constraint and the converter is solving the wrong problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

D = ("/private/tmp/claude-501/-Users-aaronjosserand-austin-Projects-neural-splines/"
     "d8eefa75-2215-4320-8042-972f2a2367bf/scratchpad/data")
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
tr = DataLoader(datasets.MNIST(D, train=True, transform=tf), batch_size=256, shuffle=True)
te = DataLoader(datasets.MNIST(D, train=False, transform=tf), batch_size=512)

# ---- teacher ----
torch.manual_seed(0)
teacher = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
opt = optim.Adam(teacher.parameters(), lr=1e-3)
for _ in range(3):
    teacher.train()
    for d, t in tr:
        opt.zero_grad(); F.cross_entropy(teacher(d), t).backward(); opt.step()
teacher.eval()


def evaluate(model):
    model.eval(); c = 0
    with torch.no_grad():
        for d, t in te:
            c += model(d).argmax(1).eq(t).sum().item()
    return 100 * c / len(te.dataset)


print(f"teacher accuracy: {evaluate(teacher):.2f}%\n", flush=True)
W = teacher[1].weight.data.clone()
b1 = teacher[1].bias.data.clone()
m, n = W.shape


class SplineStudent(nn.Module):
    """Teacher architecture with layer 1's weight generated from control points."""

    def __init__(self, p, q):
        super().__init__()
        self.cp = nn.Parameter(torch.zeros(p, q))
        with torch.no_grad():
            ri = torch.linspace(0, m - 1, p).long(); ci = torch.linspace(0, n - 1, q).long()
            self.cp.copy_(W[ri][:, ci])
        self.b1 = nn.Parameter(b1.clone())
        self.fc2 = nn.Linear(256, 10)
        self.fc2.load_state_dict(teacher[3].state_dict())

    def weight(self):
        return F.interpolate(self.cp[None, None], size=(m, n),
                             mode="bicubic", align_corners=True)[0, 0]

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(F.linear(x, self.weight(), self.b1)))


def rel_err(model):
    with torch.no_grad():
        return (torch.norm(model.weight() - W) / torch.norm(W)).item()


def run(p, q, objective, epochs=2):
    torch.manual_seed(0)
    s = SplineStudent(p, q)
    if objective == "frobenius":
        o = optim.LBFGS([s.cp], lr=0.5, max_iter=400, line_search_fn="strong_wolfe")

        def closure():
            o.zero_grad(); l = torch.norm(s.weight() - W); l.backward(); return l
        o.step(closure)
        return s
    params = [s.cp, s.b1] + list(s.fc2.parameters())
    o = optim.Adam(params, lr=3e-3)
    for _ in range(epochs):
        s.train()
        for d, t in tr:
            o.zero_grad()
            if objective == "distill":
                with torch.no_grad():
                    tgt = teacher(d)
                loss = F.kl_div(F.log_softmax(s(d), -1), F.softmax(tgt, -1), reduction="batchmean")
            else:
                loss = F.cross_entropy(s(d), t)
            loss.backward(); o.step()
    return s


print(f"{'budget':>7} {'grid':>9} {'objective':>10} {'rel_err(W)':>11} {'accuracy':>9}", flush=True)
for (p, q) in [(4, 4), (16, 16), (32, 64)]:
    for obj in ("frobenius", "distill", "labels"):
        s = run(p, q, obj)
        print(f"{p*q:>7} {f'{p}x{q}':>9} {obj:>10} {rel_err(s):>11.4f} {evaluate(s):>8.2f}%", flush=True)
print("\n(random chance ~9.8%)", flush=True)
