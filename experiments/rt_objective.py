"""RED-TEAM 2 and 3 on the objective experiment.

RT2. Was `frob_only` just under-optimised?
     W_hat = A C B^T is LINEAR in C, so ||A C B^T - W||_F is a convex
     quadratic with a unique global minimum and a closed form
     C* = pinv(A) W pinv(B)^T. If LBFGS matches the closed form, the
     Frobenius arm is provably at its ceiling and cannot be blamed on
     optimiser effort.

RT3. Does the TEACHER contribute anything at all?
     The distilled student initialises its control points from the teacher's
     weights, and objective.py's `labels` arm scored the same as `distill`.
     If a randomly initialised student trained on labels alone matches it,
     then this is not "compressing the teacher" -- it is just training a
     small model, and the converter has no reason to exist.

     Also tested: spline-L1 + DENSE-L2 versus spline-L1 + SPLINE-L2, because
     the 92-94% here conflicts with the ~85% ceiling measured earlier for
     SplineMLP, and the architectures differ in exactly that way.
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
o = optim.Adam(teacher.parameters(), lr=1e-3)
for _ in range(3):
    teacher.train()
    for d, t in tr:
        o.zero_grad(); F.cross_entropy(teacher(d), t).backward(); o.step()
teacher.eval()
W = teacher[1].weight.data.clone(); B1 = teacher[1].bias.data.clone()
m, n = W.shape


def ev(model):
    model.eval(); c = 0
    with torch.no_grad():
        for d, t in te:
            c += model(d).argmax(1).eq(t).sum().item()
    return 100 * c / len(te.dataset)


print(f"teacher: {ev(teacher):.2f}%\n", flush=True)

# ------------------------------------------------------------------ RT2
def axis_op(n_cp, n_out):
    I = torch.eye(n_cp)
    return F.interpolate(I[None, None], size=(n_out, n_cp), mode="bicubic",
                         align_corners=True)[0, 0]


print("RT2: is the Frobenius fit at its provable global optimum?")
print(f"{'grid':>8} {'LBFGS err':>11} {'closed-form':>12} {'gap':>10}", flush=True)
for (p, q) in [(4, 4), (16, 16), (32, 64)]:
    A, Bm = axis_op(p, m), axis_op(q, n)
    C_star = torch.linalg.pinv(A) @ W @ torch.linalg.pinv(Bm).T
    err_cf = (torch.norm(A @ C_star @ Bm.T - W) / torch.norm(W)).item()

    cp = torch.zeros(p, q, requires_grad=True)
    with torch.no_grad():
        ri = torch.linspace(0, m - 1, p).long(); ci = torch.linspace(0, n - 1, q).long()
        cp.copy_(W[ri][:, ci])
    opt = optim.LBFGS([cp], lr=0.5, max_iter=400, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        R = F.interpolate(cp[None, None], size=(m, n), mode="bicubic", align_corners=True)[0, 0]
        l = torch.norm(R - W); l.backward(); return l
    opt.step(cl)
    with torch.no_grad():
        R = F.interpolate(cp[None, None], size=(m, n), mode="bicubic", align_corners=True)[0, 0]
    err_lb = (torch.norm(R - W) / torch.norm(W)).item()
    print(f"{f'{p}x{q}':>8} {err_lb:>11.4f} {err_cf:>12.4f} {err_lb-err_cf:>+10.5f}", flush=True)

# ------------------------------------------------------------------ RT3
class Student(nn.Module):
    def __init__(self, p, q, spline_head=False, from_teacher=True):
        super().__init__()
        self.cp = nn.Parameter(torch.zeros(p, q))
        with torch.no_grad():
            if from_teacher:
                ri = torch.linspace(0, m - 1, p).long(); ci = torch.linspace(0, n - 1, q).long()
                self.cp.copy_(W[ri][:, ci])
            else:
                self.cp.normal_(0, 0.02)
        self.b1 = nn.Parameter(B1.clone() if from_teacher else torch.zeros(m))
        self.spline_head = spline_head
        if spline_head:
            self.cp2 = nn.Parameter(torch.randn(q if q < 10 else 10, q) * 0.02)
            self.b2 = nn.Parameter(torch.zeros(10))
        else:
            self.fc2 = nn.Linear(256, 10)
            if from_teacher:
                self.fc2.load_state_dict(teacher[3].state_dict())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        w1 = F.interpolate(self.cp[None, None], size=(m, n), mode="bicubic",
                           align_corners=True)[0, 0]
        h = F.relu(F.linear(x, w1, self.b1))
        if self.spline_head:
            w2 = F.interpolate(self.cp2[None, None], size=(10, 256), mode="bicubic",
                               align_corners=True)[0, 0]
            return F.linear(h, w2, self.b2)
        return self.fc2(h)


def train(s, mode, epochs=2):
    op = optim.Adam(s.parameters(), lr=3e-3)
    for _ in range(epochs):
        s.train()
        for d, t in tr:
            op.zero_grad()
            if mode == "distill":
                with torch.no_grad():
                    tg = teacher(d)
                F.kl_div(F.log_softmax(s(d), -1), F.softmax(tg, -1),
                         reduction="batchmean").backward()
            else:
                F.cross_entropy(s(d), t).backward()
            op.step()
    return s


print("\nRT3: does the teacher contribute anything?")
print(f"{'grid':>8} {'head':>7} {'init':>10} {'objective':>10} {'accuracy':>9}", flush=True)
for (p, q) in [(16, 16), (32, 64)]:
    for head, init, mode in [
        ("dense", "teacher", "distill"),
        ("dense", "teacher", "labels"),
        ("dense", "random", "labels"),
        ("spline", "random", "labels"),
    ]:
        torch.manual_seed(0)
        s = Student(p, q, spline_head=(head == "spline"), from_teacher=(init == "teacher"))
        train(s, mode)
        print(f"{f'{p}x{q}':>8} {head:>7} {init:>10} {mode:>10} {ev(s):>8.2f}%", flush=True)

print("\nRT4: honest parameter accounting for the 32x64 student", flush=True)
head = 256 * 10 + 10
print(f"  teacher total          : {784*256+256+head:>8,}")
print(f"  student cp only        : {32*64:>8,}   (layer-1 ratio {784*256/(32*64):.0f}x)")
print(f"  student total w/ head  : {32*64+256+head:>8,}   (overall ratio "
      f"{(784*256+256+head)/(32*64+256+head):.1f}x)")
