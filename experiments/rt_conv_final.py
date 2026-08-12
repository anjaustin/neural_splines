"""Final fair comparison: corrected init, 3 seeds, and a PARAM-MATCHED 3x3.

The earlier 3x3 baseline had 20,490 parameters against the spline's 28,938.
Comparing accuracy across a 41% parameter gap favours the larger model, so a
wider 3x3 (c2=48) is included at a matched budget.
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from neural_splines import SplineConv2d
D=("/private/tmp/claude-501/-Users-aaronjosserand-austin-Projects-neural-splines/"
   "d8eefa75-2215-4320-8042-972f2a2367bf/scratchpad/data")
tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
TR=DataLoader(datasets.MNIST(D,train=True,transform=tf),batch_size=128,shuffle=True)
TE=DataLoader(datasets.MNIST(D,train=False,transform=tf),batch_size=1024)
def net(kind,k,cp=None,c1=16,c2=32):
    p=k//2
    def cv(i,o): return nn.Conv2d(i,o,k,padding=p) if kind=="dense" else SplineConv2d(i,o,k,cp_h=cp,padding=p)
    return nn.Sequential(cv(1,c1),nn.ReLU(),nn.MaxPool2d(2),cv(c1,c2),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Flatten(),nn.Linear(c2*7*7,10))
def run(mk,seed,ep=8):
    torch.manual_seed(seed); m=mk(); n=sum(p.numel() for p in m.parameters())
    o=optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4)
    s=optim.lr_scheduler.OneCycleLR(o,max_lr=2e-3,epochs=ep,steps_per_epoch=len(TR))
    for _ in range(ep):
        m.train()
        for d,t in TR: o.zero_grad(); F.cross_entropy(m(d),t).backward(); o.step(); s.step()
    m.eval(); c=0
    with torch.no_grad():
        for d,t in TE: c+=m(d).argmax(1).eq(t).sum().item()
    return n,100*c/len(TE.dataset)
CFGS=[("dense  3x3  c2=48 (param-matched)", lambda: net("dense",3,c2=48)),
      ("spline 9x9  cp=5 (corrected init)", lambda: net("spline",9,5))]
print(f"{'configuration':<36} {'params':>8}  {'s0':>6} {'s1':>6} {'s2':>6}  {'mean':>6} {'spread':>6}",flush=True)
for lab,mk in CFGS:
    a=[]
    for s in (0,1,2):
        n,x=run(mk,s); a.append(x)
    print(f"{lab:<36} {n:>8,}  "+" ".join(f"{v:>6.2f}" for v in a)+
          f"  {sum(a)/3:>6.2f} {max(a)-min(a):>6.2f}",flush=True)
