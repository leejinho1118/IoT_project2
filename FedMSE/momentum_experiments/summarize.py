import csv, json, math
from pathlib import Path
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'src/Checkpoint/Results/EXP/H12/final_mse_avg'
VARIANTS={'A1-p2':'p2','A1-p4':'p4','A2-FedAvgM':'fedavgm','A3-trimmed':'trimmed'}
def read(path):
    return [list(json.loads(line).values())[0] for line in path.open()]
def final(path):
    d=read(path)
    if len(d)!=20: raise ValueError(f'incomplete {path}: {len(d)} rounds')
    return d[-1],d
base={r:(read(BASE/f'Run_{r}.jsonl')) for r in range(10)}
bvals=np.array([np.mean(base[r][-1]['global_auc']) for r in range(10)])
rows=[]; summary=[]
for label,v in VARIANTS.items():
    vals=[]; diffs=[]; join_ok=True
    for r in range(10):
        last,data=final(ROOT/f'runs/{v}/noniid/Run_{r}.jsonl')
        vals.append(np.mean(last['global_auc']))
        join_ok &= all(data[i]['join_clients']==base[r][i]['join_clients'] for i in range(20))
        for rnd,x in enumerate(data,1):
            rows.append({'variant':label,'run':r,'round':rnd,'mean_client_auc':float(np.mean(x['global_auc'])),'global_auc_clients':json.dumps(x['global_auc']),'join_clients':json.dumps(x['join_clients'])})
    vals=np.array(vals); diffs=(vals-bvals)*100
    mean=float(np.mean(vals)); md=float(np.mean(diffs)); sd=float(np.std(diffs,ddof=1)); half=float(t.ppf(.975,9)*sd/math.sqrt(10))
    summary.append({'variant':label,'runs':10,'mean_final_auc':mean,'mean_final_auc_pct':mean*100,'paired_diff_pp':md,'ci95_low_pp':md-half,'ci95_high_pp':md+half,'df':9,'join_sequences_match':join_ok})
with (ROOT/'RESULTS.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
with (ROOT/'SUMMARY.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
# MSE replay comparison uses all rounds, device AUC values and participant sequence.
a=read(BASE/'Run_0.jsonl'); b=read(ROOT/'runs/mse/mse_repro/Run_0.jsonl')
repro={'rounds_old':len(a),'rounds_new':len(b),'joins_match':len(a)==len(b) and all(x['join_clients']==y['join_clients'] for x,y in zip(a,b)),
       'max_abs_auc_diff':max(float(np.max(np.abs(np.asarray(x['global_auc'])-np.asarray(y['global_auc'])))) for x,y in zip(a,b)) if len(a)==len(b) else None}
(ROOT/'REPRO_CHECK.json').write_text(json.dumps(repro,indent=2)+'\n')
print(json.dumps({'summary':summary,'repro':repro},indent=2))
