# 사용: python3 launch_optim.py W stage   (stage=1: ToN s1234 8셀, stage=2: 선택 결과 JSON(optim_selected.json)로 확인 실행)
import subprocess, sys, json, time
from pathlib import Path
root=Path(__file__).resolve().parent; src=root.parent.parent/'src'; logs=root/'logs'
NONIID='Configuration/repro-noniid-10clients.json'
W=int(sys.argv[1]); stage=int(sys.argv[2]); R=20; RUNS=range(10,11) if stage==1 else range(10,15)  # 1단계 1런 축소: DEVIATIONS_optim.md
ARMS={}  # 이름 -> (variant, extra args)
for e in ('0.003','0.01','0.03'):
    ARMS[f'adam_e{e}']=('mse_adam',['--server_eta',e]); ARMS[f'yogi_e{e}']=('mse_yogi',['--server_eta',e])
ARMS['mom_b0.5_e1']=('mse_m',['--momentum','0.5','--server_eta','1']); ARMS['mom_b0.9_e0.1']=('mse_m',['--momentum','0.9','--server_eta','0.1'])
jobs=[]
if stage==1:
    for a,(v,ex) in ARMS.items():
        for r in RUNS: jobs.append((f'O1_{a}_ton_s1234',v,ex,'ton',1234,r))
else:
    sel=json.load(open(root/'optim_selected.json'))   # {"adam":"adam_e0.01","yogi":..., "mom":...}
    for a in sel.values():
        v,ex=ARMS[a]
        for ds,s in (('ton',2024),('nbaiot',1234)):
            for r in RUNS: jobs.append((f'O2_{a}_{ds}_s{s}',v,ex,ds,s,r))
running=[]
while jobs or running:
    for p,j in list(running):
        if p.poll() is not None: print('done',j,'rc',p.returncode,flush=True); running.remove((p,j))
    while jobs and len(running)<W:
        c,v,ex,ds,s,r=jobs.pop(0)
        cmd=['nice','-n','10',sys.executable,str(root/'run_confirm.py'),'--data_seed',str(s),'--runs',str(r),'--rounds',str(R),'--exp','OPTIM20261123','--cell',c,
             '--dataset',ds,'--patience','3','--threads','1','--topology','parallel','--variant',v]+ex+(['--config',NONIID] if ds=='nbaiot' else [])
        running.append((subprocess.Popen(cmd,cwd=src,stdout=open(logs/f'optim_{c}_run{r}.log','w'),stderr=subprocess.STDOUT),(c,v,ds,s,r)))
    time.sleep(5)
print('ALL DONE',flush=True)
