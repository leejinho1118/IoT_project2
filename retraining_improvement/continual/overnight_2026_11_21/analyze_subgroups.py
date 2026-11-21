"""Summarize device/pattern/attack-type failures without multiplying repeated test rows."""
from pathlib import Path
import json,csv,collections
import numpy as np
H=Path(__file__).resolve().parent;A=json.loads((H/'analysis_v1/analysis.json').read_text());R=A['selection']['selected'][1];groups=collections.defaultdict(list)
for run in A['runs']:
 if run['mode']!=R['mode'] or run['loss']!=R['loss']:continue
 folder=next((H/root/run['run'] for root in ['runs','normal_runs','control_runs','objective_runs'] if (H/root/run['run']).exists()),None);records=json.loads((folder/'final_evaluation_v1/results.json').read_text())
 for r in records:
  if r['stage']!=3:continue
  for domain in ['old','new']:
   m=r['metrics'][domain]
   for typ,field in [('attack','attack_type_tpr'),('normal','normal_pattern_fpr')]:
    for label,v in m[field].items():groups[(run['data'],run['kind'],r['method'],domain,typ,label)].append((r['seed'],v['total'],v['tpr'] if typ=='attack' else v['fpr']))
   for c in m['clients']:
    for typ,field,nfield in [('device_FPR','fpr','normal'),('device_TPR','tpr','attack')]:
     if c[field] is not None:groups[(run['data'],run['kind'],r['method'],domain,typ,str(c['device']))].append((r['seed'],c[nfield],c[field]))
rows=[]
for k,v in groups.items():
 assert len({x[0] for x in v})==len(v);assert len({x[1] for x in v})==1
 rows.append(dict(zip(['data','kind','method','domain','type','group'],k))|dict(n_seeds=len(v),test_rows_per_seed=v[0][1],mean_rate=float(np.mean([x[2] for x in v])),sd_rate=float(np.std([x[2] for x in v],ddof=1)) if len(v)>1 else None,min_rate=min(x[2] for x in v),max_rate=max(x[2] for x in v)))
p=H/'analysis_v1/subgroup_summary.json';assert not p.exists();p.write_text(json.dumps(rows,indent=2),encoding='utf-8')
with p.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Subgroup summaries:',len(rows))
