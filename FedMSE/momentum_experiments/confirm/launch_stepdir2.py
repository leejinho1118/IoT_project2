import subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parent; src=root.parent.parent/'src'; logs=root/'logs'
jobs=[]
for ds in ('ton','cic_km'):
    jobs+=[(f'D0L_{ds}_s1234','mse',ds,None),(f'D1L_{ds}_s1234','mse_m',ds,None),(f'D2L_{ds}_s1234','mse_dm',ds,None),
           (f'D0M_{ds}_s1234','mse',ds,f'runs/mse_m/D1_{ds}_s1234')]
ps=[]
for c,v,ds,tgt in jobs:
    cmd=['nice','-n','10',sys.executable,str(root/'run_confirm.py'),'--data_seed','1234','--runs','10','--rounds','40','--exp','STEPDIR20261124','--cell',c,'--patience','3','--threads','1','--topology','parallel','--lr','1e-5','--variant',v,'--dataset',ds,'--dir_log']
    if tgt: cmd+=['--disp_target_cell',str(root/tgt)]
    ps.append(subprocess.Popen(cmd,cwd=src,stdout=open(logs/f'stepdir2_{c}_run10.log','w'),stderr=subprocess.STDOUT))
for p in ps: p.wait()
print('ALL DONE',[p.returncode for p in ps],flush=True)
