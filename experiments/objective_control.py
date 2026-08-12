"""Control for the objective experiment: how much of the gain is the SPLINE,
and how much is just retraining the downstream layer?

In objective.py the distill/labels runs optimized cp AND b1 AND fc2, while the
frobenius run optimized cp alone. That confounds the comparison. Here every
arm is labelled by exactly which tensors are trainable:

  frob_only      cp fitted to W, nothing else trained        (the converter)
  frob_then_head cp fitted to W and FROZEN, head retrained   <- isolates the head
  distill_cp     cp trained by distillation, head FROZEN     <- isolates the spline
  distill_both   cp and head both trained                    (objective.py's arm)

If frob_then_head already recovers most of the accuracy, the spline grid is
not what is doing the work and objective.py overstated its case.
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

torch.manual_seed(0)
teacher = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
opt = optim.Adam(teacher.parameters(), lr=1e-3)
for _ in range(3):
    teacher.train()
    for d, t in tr:
        opt.zero_grad(); F.cross_entropy(teacher(d), t).backward(); opt.step()
teacher.eval()
W = teacher[1].weight.data.clone(); b1 = teacher[1].bias.data.clone()
m, n = W.shape


def evaluate(model):
    model.eval(); c = 0
    with torch.no_grad():
        for d, t in te:
            c += model(d).argmax(1).eq(t).sum().item()
    return 100 * c / len(te.dataset)


print(f"teacher: {evaluate(teacher):.2f}%\n", flush=True)


class Student(nn.Module):
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


def frob_fit(s):
    o = optim.LBFGS([s.cp], lr=0.5, max_iter=400, line_search_fn="strong_wolfe")

    def closure():
        o.zero_grad(); l = torch.norm(s.weight() - W); l.backward(); return l
    o.step(closure)


def distill(s, params, epochs=2):
    if not params:
        return
    o = optim.Adam(params, lr=3e-3)
    for _ in range(epochs):
        s.train()
        for d, t in tr:
            with torch.no_grad():
                tgt = teacher(d)
            o.zero_grad()
            F.kl_div(F.log_softmax(s(d), -1), F.softmax(tgt, -1),
                     reduction="batchmean").backward()
            o.step()


HEAD_PARAMS = 256 + 256 * 10 + 10   # b1 + fc2

print(f"{'grid':>8} {'cp':>6} {'+head':>7} {'arm':>15} {'trainable':>22} {'acc':>8}", flush=True)
for (p, q) in [(4, 4), (16, 16), (32, 64)]:
    for arm in ("frob_only", "frob_then_head", "distill_cp", "distill_both"):
        torch.manual_seed(0)
        s = Student(p, q)
        if arm.startswith("frob"):
            frob_fit(s)
        if arm == "frob_then_head":
            distill(s, [s.b1] + list(s.fc2.parameters()))
            trainable = "head only (cp frozen)"
        elif arm == "distill_cp":
            distill(s, [s.cp])
            trainable = "cp only (head frozen)"
        elif arm == "distill_both":
            distill(s, [s.cp, s.b1] + list(s.fc2.parameters()))
            trainable = "cp + head"
        else:
            trainable = "none (cp fitted to W)"
        print(f"{f'{p}x{q}':>8} {p*q:>6} {p*q+HEAD_PARAMS:>7} {arm:>15} "
              f"{trainable:>22} {evaluate(s):>7.2f}%", flush=True)
    print(flush=True)
print("(teacher layer-1 alone is 200,704 params; random chance ~9.8%)", flush=True)
