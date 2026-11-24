import subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parent; src=root.parent.parent/'src'; logs=root/'logs'
NONIID='Configuration/repro-noniid-10clients.json'
W=int(sys.argv[1]); R=int(sys.argv[2]); SUF=sys.argv[3] if len(sys.argv)>3 else ''; SMOKE=SUF=='_smoke'
seeds=[2024] if SMOKE else [2024,4321,7777]
jobs=[]
for s in seeds:
    for lrtag,lr,runs in (('1e-5','1e-5',range(10,20)),('5e-5','5e-5',range(10,15))):
        if SMOKE and lrtag!='1e-5': continue
        for r in (runs if not SMOKE else [10]):
            for k,v in (('D0','mse'),('D1','mse_m')): jobs.append((f'{k}_s{s}_lr{lrtag}',v,s,lr,r))
running=[]
while jobs or running:
    for p,j in list(running):
        if p.poll() is not None: print('done',j,'rc',p.returncode,flush=True); running.remove((p,j))
    while jobs and len(running)<W:
        c,v,s,lr,r=jobs.pop(0)
        cmd=['nice','-n','10',sys.executable,str(root/'run_confirm.py'),'--data_seed',str(s),'--runs',str(r),'--rounds',str(R),'--exp','SPLIT20261122',
             '--cell',c+SUF,'--config',NONIID,'--patience','3','--threads','1','--topology','parallel','--lr',lr,'--variant',v]
        running.append((subprocess.Popen(cmd,cwd=src,stdout=open(logs/f'split_{c}{SUF}_run{r}.log','w'),stderr=subprocess.STDOUT),(c,v,s,lr,r)))
    time.sleep(5)
print('ALL DONE',flush=True)
