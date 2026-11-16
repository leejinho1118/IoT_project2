"""정상 보정만으로 고정한 q별 오탐/탐지 절충. 보류 정답으로 임계값을 정하지 않는다."""
import sys,json,gzip,csv,hashlib,time,statistics
from pathlib import Path
import torch,numpy as np,pandas as pd
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from metric_models import METHODS
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'continual/drift_v2'));import backend as B
D=Path(__file__).resolve().parent;O=D/'output/metrics_v1';T=D/'output/operating_metrics_v1'
if T.exists():raise FileExistsError('Preserve old results')
T.mkdir();torch.set_num_threads(1);limits=threadpool_limits(limits=1)
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
with gzip.open(O/'partition_manifest.json.gz','rt',encoding='utf-8') as f:groups=json.load(f)
seeds=sorted({r['seed'] for r in load(O/'development_results.json')});methods=[m for m in METHODS if m!='centroid_original'];qs=[.95,.975,.99,.995,1.0];thresholds={};counts=[];cost=[];cache={}
# 正常 보정 행만 읽고 모든 임계값을 먼저 동결한다.
for seed in seeds:
 bundles={m:torch.load(O/'models'/f'seed_{seed}'/f'{m}.pt',map_location='cpu',weights_only=False) for m in methods}
 b=bundles[methods[0]];model=B.Shrink_Autoencoder(**b['architecture']);model.load_state_dict(b['model_state']);model.eval()
 for cid in b['clients']:
  scaler=b['clients'][cid]['input_scaler'];views=[]
  for role in ['calib_old','calib_new']:
   rows=groups[f'{cid}/{role}'];assert all(r['y']==0 for r in rows);raw=[]
   for r in rows:
    k=(cid,r['source'])
    if k not in cache:cache[k]=pd.read_csv(ROOT/'Data/N-BaIoT/IID-50-Client_Data'/f'Client-{cid}'/r['source']/'data.csv',header=None).to_numpy(dtype=float)
    raw.append(cache[k][int(r['id'].split('/')[-1])])
   raw=np.asarray(raw);scaled=scaler.transform(raw)
   with torch.no_grad():z=model(torch.tensor(scaled,dtype=torch.float32))[0].numpy()
   views.append((scaled,z));counts.append(dict(seed=seed,client=cid,role=role,n=len(raw),nominal_rank_resolution=1/(len(raw)+1)))
  for m,bun in bundles.items():
   h=bun['clients'][cid]['head'];v=[np.concatenate([s,z],axis=1) if m.startswith('residual_') else s if m.startswith('input_') else z for s,z in views]
   start=time.perf_counter();scores=[h.score(x) for x in v];elapsed=time.perf_counter()-start
   cost.append(dict(seed=seed,client=cid,method=m,query_rows=sum(map(len,v)),score_seconds=elapsed))
   for q in qs:thresholds[f'{seed}/{cid}/{m}/{q}']=max(float(np.quantile(s,q)) for s in scores)
(T/'thresholds_frozen.json').write_text(json.dumps(dict(quantiles=qs,thresholds=thresholds,role='normal calibration only, no finite-sample guarantee')))
allrows=[]
for phase in ['development','final']:
 rows=[]
 for seed in seeds:
  for m in methods:
   with gzip.open(O/phase/f'seed_{seed}'/m/'predictions.csv.gz','rt',encoding='utf-8') as f:pred=list(csv.DictReader(f))
   for q in qs:
    metrics={}
    for domain in ['old','new']:
     ps=[r for r in pred if r['domain']==domain];y=np.array([int(r['label']) for r in ps]);s=np.array([float(r['score']) for r in ps]);th=np.array([thresholds[f'{seed}/{r["client"]}/{m}/{q}'] for r in ps]);p=s>th
     cm=confusion_matrix(y,p,labels=[0,1]);tn,fp,fn,tp=cm.ravel();metrics[domain]=dict(fpr=float(fp/(tn+fp)),tpr=float(tp/(tp+fn)),confusion_matrix=cm.tolist())
    rows.append(dict(seed=seed,method=m,quantile=q,metrics=metrics))
 summary=[]
 for m in methods:
  for q in qs:
   rr=[r for r in rows if r['method']==m and r['quantile']==q]
   v=dict(method=m,quantile=q,metrics={d:{k:statistics.mean(r['metrics'][d][k] for r in rr) for k in ['fpr','tpr']} for d in ['old','new']},worst_seed_fpr=max(r['metrics'][d]['fpr'] for r in rr for d in ['old','new']))
   v['minimum_tpr']=min(v['metrics'][d]['tpr'] for d in ['old','new']);v['meets_low_fpr_screen']=max(v['metrics'][d]['fpr'] for d in ['old','new'])<=.01 and v['worst_seed_fpr']<=.02
   summary.append(v)
 (T/f'{phase}_results.json').write_text(json.dumps(rows,indent=2));(T/f'{phase}_summary.json').write_text(json.dumps(summary,indent=2));allrows+=rows
 if phase=='development':
  feasible=[r for r in summary if r['meets_low_fpr_screen']];chosen=max(feasible,key=lambda r:r['minimum_tpr']) if feasible else None
  (T/'selection_frozen.json').write_text(json.dumps(dict(criterion='mean old/new FPR<=1%, worst-seed FPR<=2%, maximize minimum TPR; not a statistical guarantee',selected=chosen,scope='exploratory reused data, candidate is an operating point not deployment certification'),indent=2))
 else:assert (T/'selection_frozen.json').exists()
(T/'calibration_counts.json').write_text(json.dumps(counts,indent=2));(T/'head_query_cost.json').write_text(json.dumps(cost,indent=2))
(T/'complete.json').write_text(json.dumps(dict(state='complete',records=len(allrows),thresholds=len(thresholds),calibration_n_min=min(r['n'] for r in counts),calibration_n_max=max(r['n'] for r in counts),rank_resolution_max=max(r['nominal_rank_resolution'] for r in counts)),indent=2))
print((T/'complete.json').read_text());print((T/'selection_frozen.json').read_text())
