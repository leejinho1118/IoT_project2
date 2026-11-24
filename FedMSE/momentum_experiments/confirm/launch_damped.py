import subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parent; src=root.parent.parent/'src'; logs=root/'logs'
NONIID='Configuration/repro-noniid-10clients.json'
W=int(sys.argv[1]); R=int(sys.argv[2]) if len(sys.argv)>2 else 40
RUNS=range(10,20) if len(sys.argv)<=3 else range(10,10+int(sys.argv[3]))
EXP='DAMPED20261123'
# (cell, variant, data_seed, dataset, config)
cells=[('D2_mse_par','mse_dm',1234,'nbaiot')]
cells+=[(f'D2_s{s}_lr1e-5','mse_dm',s,'nbaiot') for s in (2024,4321,7777)]
cells+=[(f'D2_ton_s{s}','mse_dm',s,'ton') for s in (1234,2024)]
cells+=[('D2_cic_km_s1234','mse_dm',1234,'cic_km')]
for ds in ('ton','cic_km'):
    cells+=[(f'D0_{ds}_s4321','mse',4321,ds),(f'D2_{ds}_s4321','mse_dm',4321,ds)]
if len(sys.argv)>4 and sys.argv[4]=='A': cells=cells[:7]
jobs=[(c,v,s,ds,r) for (c,v,s,ds) in cells for r in RUNS]
running=[]
while jobs or running:
    for p,j in list(running):
        if p.poll() is not None: print('done',j,'rc',p.returncode,flush=True); running.remove((p,j))
    while jobs and len(running)<W:
        c,v,s,ds,r=jobs.pop(0)
        cmd=['nice','-n','10',sys.executable,str(root/'run_confirm.py'),'--data_seed',str(s),'--runs',str(r),'--rounds',str(R),'--exp',EXP,
             '--cell',c,'--patience','3','--threads','1','--topology','parallel','--lr','1e-5','--variant',v]
        cmd+=['--config',NONIID] if ds=='nbaiot' else ['--dataset',ds]
        running.append((subprocess.Popen(cmd,cwd=src,stdout=open(logs/f'damped_{c}_run{r}.log','w'),stderr=subprocess.STDOUT),(c,v,s,ds,r)))
    time.sleep(5)
print('ALL DONE',flush=True)
