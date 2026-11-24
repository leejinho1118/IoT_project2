import subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parent; src=root.parent.parent/'src'; logs=root/'logs'
NONIID='Configuration/repro-noniid-10clients.json'
R=int(sys.argv[2]) if len(sys.argv)>2 else 40; RUNS=list(range(10,20)) if len(sys.argv)<=3 else list(range(10,10+int(sys.argv[3])))
SUF=sys.argv[4] if len(sys.argv)>4 else ''; W=int(sys.argv[1]) if len(sys.argv)>1 else 10
jobs=[(c,v,r) for r in RUNS for c,v in (('D0_mse_par','mse'),('D1_msem_par','mse_m'))]; running=[]
while jobs or running:
    for p,j in list(running):
        if p.poll() is not None: print('done',j,'rc',p.returncode,flush=True); running.remove((p,j))
    while jobs and len(running)<W:
        c,v,r=jobs.pop(0)
        cmd=['nice','-n','10',sys.executable,str(root/'run_confirm.py'),'--data_seed','1234','--runs',str(r),'--rounds',str(R),'--exp','DIRECT20261122',
             '--cell',c+SUF,'--config',NONIID,'--patience','3','--threads','1','--topology','parallel','--variant',v]
        running.append((subprocess.Popen(cmd,cwd=src,stdout=open(logs/f'direct_{c}{SUF}_run{r}.log','w'),stderr=subprocess.STDOUT),(c,v,r)))
    time.sleep(5)
print('ALL DONE',flush=True)
