import subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parent; src=root.parent.parent/'src'; logs=root/'logs'
NONIID='Configuration/repro-noniid-10clients.json'
cells=[('NAG_mse_par',1234,'nbaiot'),('NAG_ton_s1234',1234,'ton'),('NAG_cic_km_s1234',1234,'cic_km')]
ps=[]
for c,s,ds in cells:
    for r in range(10,15):
        cmd=['nice','-n','10',sys.executable,str(root/'run_confirm.py'),'--data_seed',str(s),'--runs',str(r),'--rounds','40','--exp','NAG20261124','--cell',c,'--patience','3','--threads','1','--topology','parallel','--lr','1e-5','--variant','mse_nag']
        cmd+=['--config',NONIID] if ds=='nbaiot' else ['--dataset',ds]
        ps.append(subprocess.Popen(cmd,cwd=src,stdout=open(logs/f'nag_{c}_run{r}.log','w'),stderr=subprocess.STDOUT))
for p in ps: p.wait()
print('ALL DONE',[p.returncode for p in ps],flush=True)
