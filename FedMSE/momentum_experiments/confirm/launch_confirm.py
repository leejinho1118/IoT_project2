import subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parent
src=root.parent.parent/'src'
logs=root/'logs'; logs.mkdir(exist_ok=True)
def phase(cell, config, runs, methods, rounds=20):
    for method in methods:
        f=open(logs/f'{cell}_{method}_{rounds}.log','w')
        cmd=[sys.executable,str(root/'run_confirm.py'),'--data_seed','1234','--runs',*map(str,runs),
             '--rounds',str(rounds),'--exp','CONFIRM20261121','--cell',cell,'--config',config,
             '--patience','3','--threads','1','--variant',method]
        if method=='eta10': cmd += ['--server_eta','10']
        print(f'start {cell} {method} rounds={rounds}',flush=True)
        rc=subprocess.call(['nice','-n','10',*cmd],cwd=src,stdout=f,stderr=subprocess.STDOUT)
        f.close()
        print(f'{cell} {method} {rounds}: exit {rc}',flush=True)
        if rc: raise SystemExit(f'failed {cell} {method}: {rc}')
phase('noniid','Configuration/repro-noniid-10clients.json',range(10),['mse','fedavg','fedavgm','eta10'])
phase('noniid','Configuration/repro-noniid-10clients.json',range(10),['mse'],40)
phase('iid','Configuration/repro-noniid-10clients-iidified.json',range(5),['mse','fedavg','fedavgm','eta10'])
