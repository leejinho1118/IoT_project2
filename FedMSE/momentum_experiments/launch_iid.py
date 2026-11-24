import subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parent
src=root.parent/'src'
logs=root/'logs'; logs.mkdir(exist_ok=True)
procs=[]
for variant in ['mse','fedavgm']:
    f=open(logs/f'iid_{variant}.log','w')
    cmd=[sys.executable,str(root/'run_variants.py'),'--data_seed','1234','--runs',*map(str,range(5)),
         '--rounds','20','--exp','AGG20261121_IID','--cell','iid','--config',
         'Configuration/repro-noniid-10clients-iidified.json','--patience','3','--threads','1','--variant',variant]
    procs.append((variant,subprocess.Popen(cmd,cwd=src,stdout=f,stderr=subprocess.STDOUT),f))
while any(p.poll() is None for _,p,_ in procs):
    for name,p,_ in procs:
        if p.poll() is not None and not getattr(p,'_reported',False):
            print(f'{name}: exit {p.returncode}',flush=True); p._reported=True
    time.sleep(10)
for _,_,f in procs: f.close()
if any(p.returncode for _,p,_ in procs): sys.exit(1)
