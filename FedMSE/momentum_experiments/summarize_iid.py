import csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parent

def read(p): return [list(json.loads(x).values())[0] for x in p.open()]
def run(v,r):
 d=read(ROOT/f'runs/{v}/iid/Run_{r}.jsonl')
 if len(d)!=20: raise ValueError(f'incomplete {v} run {r}: {len(d)}')
 return d
base=[]; treatment=[]; rows=[]; join=True
for r in range(5):
 a=run('mse',r); b=run('fedavgm',r)
 join &= all(x['join_clients']==y['join_clients'] for x,y in zip(a,b))
 base.append(float(np.mean(a[-1]['global_auc']))); treatment.append(float(np.mean(b[-1]['global_auc'])))
 for method,data in [('MSEAvg',a),('FedAvgM',b)]:
  for i,d in enumerate(data,1): rows.append({'method':method,'run':r,'round':i,'mean_client_auc':float(np.mean(d['global_auc'])),'global_auc_clients':json.dumps(d['global_auc']),'join_clients':json.dumps(d['join_clients'])})
diff=(np.array(treatment)-np.array(base))*100
m=float(diff.mean()); h=float(t.ppf(.975,4)*diff.std(ddof=1)/math.sqrt(5))
with (ROOT/'IID_RESULTS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
res={'runs':5,'df':4,'mseavg_mean_final_auc':float(np.mean(base)),'fedavgm_mean_final_auc':float(np.mean(treatment)),
     'paired_diff_pp':m,'ci95_low_pp':m-h,'ci95_high_pp':m+h,'join_sequences_match':join,
     'pass_threshold_diff_gt_minus_0_3pp':bool(m>-.3)}
(ROOT/'IID_SUMMARY.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res,indent=2))
