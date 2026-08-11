"""Verify the C6 fix on real MNIST, at MATCHED parameter counts."""
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from neural_splines import SplineMLP
D=("/private/tmp/claude-501/-Users-aaronjosserand-austin-Projects-neural-splines/"
   "d8eefa75-2215-4320-8042-972f2a2367bf/scratchpad/data")
tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
tr=DataLoader(datasets.MNIST(D,train=True,transform=tf),batch_size=128,shuffle=True)
te=DataLoader(datasets.MNIST(D,train=False,transform=tf),batch_size=512)
def run(make,label,seeds=(0,1)):
    accs=[]
    for s in seeds:
        torch.manual_seed(s); m=make(); n=sum(p.numel() for p in m.parameters())
        o=optim.Adam(m.parameters(),lr=1e-3)
        for _ in range(5):
            m.train()
            for d,t in tr: o.zero_grad(); F.cross_entropy(m(d),t).backward(); o.step()
        m.eval(); c=0
        with torch.no_grad():
            for d,t in te: c+=m(d).argmax(1).eq(t).sum().item()
        accs.append(100*c/len(te.dataset))
    print(f"{label:<46} {n:>7,} params   {sum(accs)/len(accs):>6.2f}%  (seeds {['%.2f'%a for a in accs]})",flush=True)
print("5 epochs, Adam 1e-3, batch 128; dense baseline 203,530 params / ~97.8%\n",flush=True)
run(lambda: SplineMLP(784,256,10,32,32),      "OLD square grid  --cp 32")
run(lambda: SplineMLP(784,256,10,36,36),      "OLD square grid  --cp 36 (param-matched)")
run(lambda: SplineMLP.with_budget(784,256,10,2048), "NEW aspect grid  --budget 2048")
