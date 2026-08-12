import importlib.util, sys, torch
spec=importlib.util.spec_from_file_location("p","push98.py")
# reuse Net/run without executing the CFGS loop
src=open("push98.py").read().split("CFGS = [")[0]
g={}; exec(compile(src,"p","exec"),g)
Net, run = g["Net"], g["run"]
print("winner: single 1D 108x188, 40ep +aug -- seed stability", flush=True)
accs=[]
for s in (1,2):
    n,a = run(lambda: Net(coarse=(108,188)), 40, True, seed=s)
    accs.append(a); print(f"  seed {s}: {n:,} params  {a:.2f}%", flush=True)
print(f"  seed 0 was 98.67%; mean over 3 = {(98.67+sum(accs))/3:.2f}%, "
      f"spread = {max(accs+[98.67])-min(accs+[98.67]):.2f}pp", flush=True)
