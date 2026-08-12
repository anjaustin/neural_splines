"""Is it RANK or INPUT-AXIS RESOLUTION that binds?

All grids below hold cp_h * cp_w = 2048 exactly, so the parameter budget is
identical. Rank is bounded by min(cp_h, cp_w). If RANK binds, 45x45-ish wins.
If INPUT RESOLUTION binds (cp_w spread over the 784 input pixels), the wide
grids win and it is not about rank at all.
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
D_=("/private/tmp/claude-501/-Users-aaronjosserand-austin-Projects-neural-splines/"
    "d8eefa75-2215-4320-8042-972f2a2367bf/scratchpad/data")
tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
tr=DataLoader(datasets.MNIST(D_,train=True,transform=tf),batch_size=128,shuffle=True)
te=DataLoader(datasets.MNIST(D_,train=False,transform=tf),batch_size=512)
IN,HID,OUT=784,256,10
class M(nn.Module):
    def __init__(s,g1):
        super().__init__(); s.cp1=nn.Parameter(torch.randn(*g1)*0.02)
        s.cp2=nn.Parameter(torch.randn(10,64)*0.02)
        s.b1=nn.Parameter(torch.zeros(HID)); s.b2=nn.Parameter(torch.zeros(OUT))
    def forward(s,x):
        x=x.view(x.size(0),-1)
        w1=F.interpolate(s.cp1[None,None],size=(HID,IN),mode='bicubic',align_corners=True)[0,0]
        h=F.relu(F.linear(x,w1,s.b1))
        w2=F.interpolate(s.cp2[None,None],size=(OUT,HID),mode='bicubic',align_corners=True)[0,0]
        return F.linear(h,w2,s.b2)
print(f"{'L1 grid':>10} {'rank<=':>7} {'cp per 784 in':>14} {'params':>8} {'acc':>8}",flush=True)
for g in [(8,256),(16,128),(32,64),(45,45),(64,32),(128,16),(256,8)]:
    torch.manual_seed(0); m=M(g); npar=sum(p.numel() for p in m.parameters())
    o=optim.Adam(m.parameters(),lr=1e-3)
    for _ in range(5):
        m.train()
        for d,t in tr: o.zero_grad(); F.cross_entropy(m(d),t).backward(); o.step()
    m.eval(); c=0
    with torch.no_grad():
        for d,t in te: c+=m(d).argmax(1).eq(t).sum().item()
    print(f"{f'{g[0]}x{g[1]}':>10} {min(g):>7} {g[1]:>14} {npar:>8,} {100*c/len(te.dataset):>7.2f}%",flush=True)
