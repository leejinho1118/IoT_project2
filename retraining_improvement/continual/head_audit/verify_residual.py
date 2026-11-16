"""탐지기 비교의 예측·정상 fit ID·보정 임계값·고정 인코더를 독립 확인."""
import argparse,json,csv,gzip,hashlib,math,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
THREAD_LIMITS=threadpool_limits(limits=1)
from residual_models import METHODS,DIAGNOSTICS
from head_models import updated_threshold
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'continual/drift_v2'));import backend as B
P=argparse.ArgumentParser();P.add_argument('--run-id',required=True);a=P.parse_args();O=Path(__file__).resolve().parent/'output'/a.run_id
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
status=load(O/'status.json');assert status['state']=='complete'
for path,h in load(O/'protected_artifacts.json').items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
for name,h in load(O/'runtime_hashes.json').items():assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==h
for path,h in load(O/'data_audit.json')['source_csv_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
with gzip.open(O/'partition_manifest.json.gz','rt',encoding='utf-8') as f:groups=json.load(f)
by={};fps=set()
for role,rows in groups.items():
 for r in rows:assert r['id'] not in by and r['fingerprint'] not in fps;by[r['id']]=(role,r);fps.add(r['fingerprint'])
audit=load(O/'fit_audit.json')
for r in audit:
 assert all(by[i][1]['y']==0 and (by[i][0].endswith('/train') or '/candidate_' in by[i][0]) for i in r['reference_ids'])
 assert set(r['reference_ids']).isdisjoint(r['calibration_ids'])
 assert all(by[i][1]['y']==0 and by[i][0].endswith(('/calib_old','/calib_new')) for i in r['calibration_ids'])
selection=O/'selection_frozen.json';sha=hashlib.sha256(selection.read_bytes()).hexdigest();assert load(selection)['selected_method'] not in DIAGNOSTICS
count=0;client_auc=0
for phase in ['development','final']:
 records=load(O/f'{phase}_results.json');seeds={r['seed'] for r in records};assert len(records)==len(seeds)*len(METHODS)
 for r in records:
  if phase=='final':assert r['selection_sha256']==sha
  with gzip.open(O/phase/f'seed_{r["seed"]}'/r['method']/'predictions.csv.gz','rt',encoding='utf-8') as f:pred=list(csv.DictReader(f))
  for domain in ['old','new']:
   q=[p for p in pred if p['domain']==domain];assert len({x['id'] for x in q})==len(q);cm=[[0,0],[0,0]]
   normal=('decision_old' if domain=='old' else 'decision_new') if phase=='development' else ('old_normal' if domain=='old' else 'new_normal')
   attack='decision_attack' if phase=='development' else ('old_attack' if domain=='old' else 'new_attack')
   for p in q:
    role,row=by[p['id']];assert role.endswith('/'+(normal if p['label']=='0' else attack)) and row['y']==int(p['label']);assert math.isfinite(float(p['score'])) and math.isfinite(float(p['threshold']))
    assert int(p['prediction'])==int(float(p['score'])>float(p['threshold']));cm[int(p['label'])][int(p['prediction'])]+=1
   m=r['metrics'][domain];assert cm==m['confusion_matrix'];assert abs(m['fpr']-cm[0][1]/sum(cm[0]))<1e-12 and abs(m['tpr']-cm[1][1]/sum(cm[1]))<1e-12
   aucs=[]
   for cid in sorted({int(x['client']) for x in q}):
    p=[x for x in q if int(x['client'])==cid];aucs.append(roc_auc_score([int(x['label']) for x in p],[float(x['score']) for x in p]));client_auc+=1
   assert abs(np.mean(aucs)-m['mean_client_auc'])<1e-12
  count+=1
 # 임계값만 바꾼 비교에서는 점수 자체가 같아야 한다.
 for seed in seeds:
  with gzip.open(O/phase/f'seed_{seed}/centroid_original/predictions.csv.gz','rt',encoding='utf-8') as f:base=list(csv.DictReader(f))
  with gzip.open(O/phase/f'seed_{seed}/threshold_updated/predictions.csv.gz','rt',encoding='utf-8') as f:changed=list(csv.DictReader(f))
  assert [(r['id'],r['domain'],r['score']) for r in base]==[(r['id'],r['domain'],r['score']) for r in changed]
# 보정은 최종 정답과 무관한 정상 두 풀에서 계산됐는지 번들과 원본으로 다시 계산한다.
torch.set_num_threads(1);cache={};threshold_checks=0
for seed in seeds:
 bundles={m:torch.load(O/'models'/f'seed_{seed}'/f'{m}.pt',map_location='cpu',weights_only=False) for m in METHODS}
 original=bundles['centroid_original'];model=B.Shrink_Autoencoder(**original['architecture']);model.load_state_dict(original['model_state']);model.eval()
 for m,b in bundles.items():assert b['source_model_sha256']==original['source_model_sha256']
 for cid in original['clients']:
  scaler=original['clients'][cid]['input_scaler'];views={}
  for domain,key in [('old','calib_old'),('new','calib_new')]:
   xs=[]
   for r in groups[f'{cid}/{key}']:
    k=(cid,r['source'])
    if k not in cache:cache[k]=pd.read_csv(ROOT/'Data/N-BaIoT/IID-50-Client_Data'/f'Client-{cid}'/r['source']/'data.csv',header=None).to_numpy(dtype=np.float64)
    xs.append(cache[k][int(r['id'].split('/')[-1])])
   raw=np.asarray(xs);scaled=scaler.transform(raw)
   with torch.no_grad():z=model(torch.tensor(scaled,dtype=torch.float32))[0].numpy()
   views[domain]=(raw,scaled,z)
  for method,b in bundles.items():
   if method=='centroid_original':continue
   h=b['clients'][cid]['head'];scores=[]
   for domain in ['old','new']:
    raw,scaled,z=views[domain];scores.append(h.score(np.concatenate([scaled,z],axis=1) if method.startswith('residual_') else raw if method=='log_input_knn5' else scaled if method=='input_knn5' else z))
   expected=updated_threshold(*scores);assert math.isclose(expected,h.threshold,rel_tol=2e-4,abs_tol=2e-5),(seed,cid,method,expected,h.threshold);threshold_checks+=1
report=dict(validation='passed',evaluation_records=count,independent_client_auc_checks=client_auc,normal_reference_and_calibration_audits=len(audit),threshold_checks=threshold_checks,checks=['protected source/results and CSV hashes','global ID and exact feature disjointness','normal-only reference/calibration roles','all stored confusion matrices and client AUCs','threshold-only score invariance','same frozen source encoder per seed','normal calibration thresholds recomputed from raw data','development choice frozen before exploratory heldout evaluation; raw diagnostics excluded'])
(O/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
