import subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parent; src=root.parent.parent/'src'; logs=root/'logs'; logs.mkdir(exist_ok=True)
NONIID='Configuration/repro-noniid-10clients.json'
R=int(sys.argv[2]) if len(sys.argv)>2 else 40
RUNS=int(sys.argv[3]) if len(sys.argv)>3 else 10
SUF=sys.argv[4] if len(sys.argv)>4 else ''
# (cell, variant, extra args)
def cells(lr,tag):
    c=[(f'A_{tag}','fedavg',['--lr',lr]),(f'B_{tag}','fedavgm',['--lr',lr,'--momentum','0.9'])]
    if tag=='lr1e-5':
        c+=[(f'C3_{tag}','eta10',['--lr',lr,'--server_eta','3']),(f'C10_{tag}','eta10',['--lr',lr,'--server_eta','10']),
            (f'X1_{tag}','fedavgm',['--lr',lr,'--momentum','0.5']),(f'X2_{tag}','fedavgm',['--lr',lr,'--momentum','0.99'])]
    return c
p1=cells('1e-5','lr1e-5')+cells('5e-5','lr5e-5')
p2=[(f'E_{t}','fedavg',['--lr',l,'--disp_target_cell',str(root/'runs'/'fedavgm'/f'B_{t}{SUF}')]) for l,t in (('1e-5','lr1e-5'),('5e-5','lr5e-5'))]
W=int(sys.argv[1]) if len(sys.argv)>1 else 10
def run_phase(cs):
    jobs=[(c,v,a,r) for c,v,a in cs for r in range(RUNS)]; running=[]
    while jobs or running:
        for p,j in list(running):
            if p.poll() is not None: print('done',j[0],j[3],'rc',p.returncode,flush=True); running.remove((p,j))
        while jobs and len(running)<W:
            c,v,a,r=jobs.pop(0)
            cmd=['nice','-n','10',sys.executable,str(root/'run_confirm.py'),'--data_seed','1234','--runs',str(r),'--rounds',str(R),
                 '--exp','EXPAND20261121','--cell',c+SUF,'--config',NONIID,'--patience','3','--threads','1','--variant',v]+a
            running.append((subprocess.Popen(cmd,cwd=src,stdout=open(logs/f'expand_{c}{SUF}_run{r}.log','w'),stderr=subprocess.STDOUT),(c,v,a,r)))
        time.sleep(5)
run_phase(p1); run_phase(p2); print('ALL DONE',flush=True)
