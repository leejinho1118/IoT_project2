import subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parent; src=root.parent.parent/'src'; logs=root/'logs'
NONIID='Configuration/repro-noniid-10clients.json'
# 11셀 x run10. D2_mse_par·D2_s2024는 이미 실행 중이라 제외.
jobs=[('D2_s4321_lr1e-5','mse_dm',4321,'nbaiot'),('D2_s7777_lr1e-5','mse_dm',7777,'nbaiot'),
 ('D2_ton_s1234','mse_dm',1234,'ton'),('D2_ton_s2024','mse_dm',2024,'ton'),('D2_cic_km_s1234','mse_dm',1234,'cic_km'),
 ('D0_ton_s4321','mse',4321,'ton'),('D2_ton_s4321','mse_dm',4321,'ton'),('D0_cic_km_s4321','mse',4321,'cic_km'),('D2_cic_km_s4321','mse_dm',4321,'cic_km')]
ps=[]
for c,v,s,ds in jobs:
    cmd=['nice','-n','10',sys.executable,str(root/'run_confirm.py'),'--data_seed',str(s),'--runs','10','--rounds','40','--exp','DAMPED20261123','--cell',c,'--patience','3','--threads','1','--topology','parallel','--lr','1e-5','--variant',v]
    cmd+=['--config',NONIID] if ds=='nbaiot' else ['--dataset',ds]
    ps.append(subprocess.Popen(cmd,cwd=src,stdout=open(logs/f'damped_{c}_run10.log','w'),stderr=subprocess.STDOUT))
for p in ps: p.wait()
print('ALL DONE',[p.returncode for p in ps],flush=True)
