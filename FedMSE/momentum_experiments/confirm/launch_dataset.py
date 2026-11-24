import subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parent; src=root.parent.parent/'src'; logs=root/'logs'
W=int(sys.argv[1]); R=int(sys.argv[2])
jobs=[]
for ds,seeds,runs in (('ton',(1234,2024),range(10,20)),('cic_km',(1234,2024),range(10,20)),('cic_iid',(1234,),range(10,15))):
    for s in seeds:
        for r in runs:
            for k,v in (('D0','mse'),('D1','mse_m')): jobs.append((f'{k}_{ds}_s{s}',v,ds,s,r))
running=[]
while jobs or running:
    for p,j in list(running):
        if p.poll() is not None: print('done',j,'rc',p.returncode,flush=True); running.remove((p,j))
    while jobs and len(running)<W:
        c,v,ds,s,r=jobs.pop(0)
        cmd=['nice','-n','10',sys.executable,str(root/'run_confirm.py'),'--data_seed',str(s),'--runs',str(r),'--rounds',str(R),'--exp','DATASET20261122',
             '--cell',c,'--dataset',ds,'--patience','3','--threads','1','--topology','parallel','--variant',v]
        running.append((subprocess.Popen(cmd,cwd=src,stdout=open(logs/f'dataset_{c}_run{r}.log','w'),stderr=subprocess.STDOUT),(c,v,ds,s,r)))
    time.sleep(5)
print('ALL DONE',flush=True)
