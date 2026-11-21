"""Independent final CSV verification against immutable role labels."""
from pathlib import Path
import argparse,json,gzip,csv,hashlib
import numpy as np
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score
P=argparse.ArgumentParser();P.add_argument('runs',nargs='+');a=P.parse_args();H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def check(name):
 out=H/'objective_runs'/name;target=out/'final_evaluation_v1';complete=json.loads((target/'complete.json').read_text());assert complete['state']=='complete';cfg=json.loads((out/'protocol_frozen.json').read_text())['config'];data=ROOT/cfg['data']
 with gzip.open(data/'roles.json.gz','rt',encoding='utf-8') as f:roles=json.load(f)
 byid={r['id']:(key,r) for key,rows in roles.items() for r in rows};records=json.loads((target/'results.json').read_text());baseline={};checks=0;retained=set()
 for r in records:
  path=target/r['prediction_file']
  with gzip.open(path,'rt',encoding='utf-8') as f:pred=list(csv.DictReader(f))
  if r['method']=='no_update':
   sig=hashlib.sha256(path.read_bytes()).hexdigest() # gzip header varies; semantic comparison below
   baseline.setdefault(r['seed'],pred);assert pred==baseline[r['seed']]
  for domain in ['old','new']:
   rr=[x for x in pred if x['domain']==domain];m=r['metrics'][domain];assert len({x['id'] for x in rr})==len(rr)
   expected={v['id'] for k,rows in roles.items() if k.split('/')[1] in [f'final_{domain}_normal',f'final_{domain}_attack'] for v in rows};assert {x['id'] for x in rr}==expected
   for x in rr:
    key,meta=byid[x['id']];assert int(x['label'])==meta['y'];assert key.startswith(str(x['client'])+'/');assert bool(x['label_conflict']=='True')==bool(meta.get('label_conflict',False))
    if meta.get('label_conflict'):retained.add(x['id'])
   y=np.array([int(x['label']) for x in rr]);s=np.array([float(x['score']) for x in rr]);t=np.array([float(x['threshold']) for x in rr]);p=(s>t).astype(int);assert np.isfinite(s).all();assert np.array_equal(p,[int(x['prediction']) for x in rr]);cm=confusion_matrix(y,p,labels=[0,1]);assert cm.tolist()==m['confusion_matrix'];tn,fp,fn,tp=cm.ravel();assert abs(fp/(tn+fp)-m['fpr'])<1e-12;assert abs(tp/(tp+fn)-m['tpr'])<1e-12
   auc=[];ap=[]
   for cid in sorted({x['client'] for x in rr}):
    cc=[x for x in rr if x['client']==cid];yy=[int(x['label']) for x in cc];ss=[float(x['score']) for x in cc]
    if len(set(yy))==2:auc.append(roc_auc_score(yy,ss));ap.append(average_precision_score(yy,ss))
   assert len(auc)==m['auc_clients'];assert abs(np.mean(auc)-m['mean_client_auc'])<1e-12;assert abs(np.mean(ap)-m['mean_client_pr_auc'])<1e-12;checks+=len(auc)
 assert len(retained)==complete['final_conflicting_label_rows_retained']
 result=dict(state='passed',run=name,records=len(records),client_aucs_and_pr_aucs_recomputed=checks,all_expected_final_IDs_and_ground_truth=True,all_conflicting_label_rows_retained=len(retained),frozen_baseline_metrics_and_predictions_invariant=True)
 path=target/'independent_validation.json'
 if path.exists():assert json.loads(path.read_text())==result
 else:path.write_text(json.dumps(result,indent=2),encoding='utf-8')
 return result
print(json.dumps([check(n) for n in a.runs],indent=2))
