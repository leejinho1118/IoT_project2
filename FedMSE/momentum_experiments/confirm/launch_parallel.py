import subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parent; src=root.parent.parent/'src'; logs=root/'logs'; logs.mkdir(exist_ok=True)
NONIID='Configuration/repro-noniid-10clients.json'; IID='Configuration/repro-noniid-10clients-iidified.json'
jobs=[]
for m in ['mse','fedavg','fedavgm','eta10']:
    for r in range(10): jobs.append(('noniid',NONIID,m,r,20))
for m in ['mse','fedavg','fedavgm','eta10']:
    for r in range(5): jobs.append(('iid',IID,m,r,20))
for r in range(10): jobs.append(('noniid',NONIID,'mse',r,40))
W=int(sys.argv[1]) if len(sys.argv)>1 else 10
running=[]
def start(j):
    cell,cfg,m,r,rounds=j
    cmd=['nice','-n','10',sys.executable,str(root/'run_confirm.py'),'--data_seed','1234','--runs',str(r),'--rounds',str(rounds),
         '--exp','CONFIRM20261121','--cell',cell,'--config',cfg,'--patience','3','--threads','1','--variant',m]
    if m=='eta10': cmd+=['--server_eta','10']
    f=open(logs/f'{cell}_{m}_{rounds}_run{r}.log','w')
    return subprocess.Popen(cmd,cwd=src,stdout=f,stderr=subprocess.STDOUT),j
while jobs or running:
    for p,j in list(running):
        if p.poll() is not None:
            print('done',j,'rc',p.returncode,flush=True); running.remove((p,j))
    while jobs and len(running)<W: running.append(start(jobs.pop(0)))
    time.sleep(5)
print('ALL DONE',flush=True)
