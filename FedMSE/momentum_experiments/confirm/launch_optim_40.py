# 추가 탐색 6셀 (DEVIATIONS_optim.md 2026-11-23). 사용: python3 launch_optim_40.py W
import subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parent; src=root.parent.parent/'src'; logs=root/'logs'
W=int(sys.argv[1]); R=40
ARMS={'mom_b0.9_e0.03':('mse_m',['--momentum','0.9','--server_eta','0.03']),'mom_b0.9_e0.1':('mse_m',['--momentum','0.9','--server_eta','0.1'])}
jobs=[(f'O1y_{a}_ton_s1234',v,ex) for a,(v,ex) in ARMS.items()]
running=[]
while jobs or running:
    for p,j in list(running):
        if p.poll() is not None: print('done',j,'rc',p.returncode,flush=True); running.remove((p,j))
    while jobs and len(running)<W:
        c,v,ex=jobs.pop(0)
        cmd=['nice','-n','10',sys.executable,str(root/'run_confirm.py'),'--data_seed','1234','--runs','10','--rounds',str(R),'--exp','OPTIM20261123','--cell',c,
             '--dataset','ton','--patience','3','--threads','1','--topology','parallel','--variant',v]+ex
        running.append((subprocess.Popen(cmd,cwd=src,stdout=open(logs/f'optim40_{c}_run10.log','w'),stderr=subprocess.STDOUT),(c,v)))
    time.sleep(5)
print('ALL DONE',flush=True)
