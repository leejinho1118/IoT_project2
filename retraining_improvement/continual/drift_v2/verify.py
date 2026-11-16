"""분할·학습 ID·예측·AUC·보관 예산을 독립 재검산한다. 재학습하지 않는다."""
import argparse,json,gzip,csv,math,hashlib,itertools,statistics,pickle,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from scenario import uncertainty_verdict
P=argparse.ArgumentParser();P.add_argument('--run-id',required=True);args=P.parse_args()
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'output'/args.run_id
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
cfg=load(OUT/'config.json');s=load(OUT/'status.json');records=load(OUT/'results.json')
assert s['state']=='complete'
for name,h in s['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,('source changed',name)
arms=['no_update']+[f'{"new_only" if cap==0 else "replay"+str(cap)}_u{int(rate*100):02d}' for rate in cfg['uncertainty_rates'] for cap in [0]+cfg['capacities']]
expected=set(itertools.product(cfg['seeds'],arms,range(1,cfg['stages']+1)))
assert len(records)==len(expected) and {(r['seed'],r['arm'],r['stage']) for r in records}==expected
with gzip.open(OUT/'split_manifest.json.gz','rt',encoding='utf-8') as f:groups=json.load(f)
by={};fingerprints=set()
for role,rows in groups.items():
 for row in rows:
  assert row['id'] not in by and row['fingerprint'] not in fingerprints
  by[row['id']]=(role,row);fingerprints.add(row['fingerprint'])
with (OUT/'scenario_mapper.pkl').open('rb') as f:mapper=pickle.load(f)
assert set(mapper['source_ids'])==set(load(OUT/'split_audit.json')['scenario_fit_ids'])
assert all(by[i][0].endswith('/scenario') and by[i][1]['y']==0 for i in mapper['source_ids'])
# 재검산용 원본 자료는 fit에 전달하지 않는다. mapper가 초기 A와 후보 B를 구분하는지 직접 확인한다.
source_cache={}
def raw_value(row):
 key=(row['client'],row['source'])
 if key not in source_cache:
  import pandas as pd
  source_cache[key]=pd.read_csv(ROOT/'Data/N-BaIoT/IID-50-Client_Data'/f'Client-{key[0]}'/key[1]/'data.csv',header=None).to_numpy(dtype=np.float64)
 return source_cache[key][int(row['id'].split('/')[-1])]
normal_roles={role:rows for role,rows in groups.items() if not role.endswith(('/old_attack','/new_attack'))}
for role,rows in normal_roles.items():
 normals=[r for r in rows if r['y']==0]
 if not normals:continue
 x=np.asarray([raw_value(r) for r in normals]);z=mapper['pca'].transform(mapper['scaler'].transform(np.sign(x)*np.log1p(np.abs(x)))).ravel()
 assert all(hashlib.sha256(np.where(xi==0,0,xi).astype('<f8').tobytes()).hexdigest()==r['fingerprint'] for xi,r in zip(x,normals))
 if role.endswith(('/train','/valid','/dev','/old_normal')):assert np.all(z<=mapper['cut'])
 if '/candidate_' in role or role.endswith(('/new_valid','/new_normal')):assert np.all(z>mapper['cut'])
for r in records:
 folder=OUT/f'seed_{r["seed"]}'/r['arm']/f'stage_{r["stage"]}'
 with gzip.open(folder/'predictions.csv.gz','rt',encoding='utf-8') as f:pred=list(csv.DictReader(f))
 assert len({p['id'] for p in pred})==len(pred)
 for p in pred:
  role,row=by[p['id']];assert role.endswith(('/old_normal','/old_attack')) if p['domain']=='old' else role.endswith(('/new_normal','/new_attack'))
  assert row['y']==int(p['label']) and math.isfinite(float(p['score']))
  assert int(p['prediction'])==int(float(p['score'])>float(p['threshold']))
 for domain in ['old','new']:
  prs=[p for p in pred if p['domain']==domain];cm=[[0,0],[0,0]]
  for p in prs:cm[int(p['label'])][int(p['prediction'])]+=1
  m=r['metrics'][domain];assert m['confusion_matrix']==cm
  assert abs(m['fpr']-cm[0][1]/sum(cm[0]))<1e-12 and abs(m['tpr']-cm[1][1]/sum(cm[1]))<1e-12
  aucs=[]
  for cid in range(1,cfg['clients']+1):
   q=[p for p in prs if int(p['client'])==cid];aucs.append(roc_auc_score([int(p['label']) for p in q],[float(p['score']) for p in q]))
  assert abs(statistics.mean(aucs)-m['mean_client_auc'])<1e-12
  assert math.isfinite(r['reconstruction_mse'][domain])
 inputs=load(folder/'training_input_ids.json');routing=load(OUT/f'seed_{r["seed"]}/verified_u{int(r["uncertainty_rate"]*100):02d}_stage_{r["stage"]}/routing.json')['items']
 for item in routing:assert uncertainty_verdict(by[item['id']][1],r['uncertainty_rate'])==item['verified']
 allowed={item['id'] for item in routing if item['verified']=='normal'};replay_count=0
 for cid,ids in inputs.items():
  assert len(set(ids))==len(ids)
  if r['capacity']>0:bank={row['id'] for row in groups[f'{cid}/train'][:r['capacity']]}
  else:bank=set()
  assert set(ids)<=allowed|bank
  assert all(by[i][1]['y']==0 for i in ids)
  replay_count+=len(set(ids)&bank)
 assert replay_count==r['replay_samples']
 if r['arm']=='no_update':assert not inputs and r['optimizer_steps']==0
 elif not r['update_skipped']:
  training=load(folder/'training/training.json');assert len(training['rounds'])<=cfg['update_rounds']
  steps=0
  for rr in training['rounds']:
   assert len(set(rr['participants']))==int(.5*cfg['clients']) and math.isfinite(rr['global_loss'])
   for loc in rr['local']:
    assert loc['rows']==len(inputs[str(loc['client'])]);assert loc['optimizer_steps']==loc['epochs']*math.ceil(loc['rows']/cfg['batch_size']);assert 1<=loc['epochs']<=cfg['local_epochs'];steps+=loc['optimizer_steps']
  assert steps==r['optimizer_steps']
for seed in cfg['seeds']:
 base=load(OUT/f'seed_{seed}/initial_evaluation/metrics.json');no=[r for r in records if r['seed']==seed and r['arm']=='no_update']
 assert len({r['state_sha256'] for r in no})==1 and all(r['metrics']==base for r in no)
 for rate in cfg['uncertainty_rates']:
  for stage in range(1,cfg['stages']+1):
   rs=[r for r in records if (r['seed'],r['stage'],r['uncertainty_rate'])==(seed,stage,rate) and r['arm']!='no_update']
   budget=[];hist=[]
   for r in rs:
    folder=OUT/f'seed_{seed}'/r['arm']/f'stage_{stage}';inputs=load(folder/'training_input_ids.json');budget.append({k:len(v) for k,v in inputs.items()})
    if not r['update_skipped']:hist.append(load(folder/'training/training.json')['rounds'])
   assert all(b==budget[0] for b in budget)
   for a,b in zip(hist,hist[1:]):assert all(x['participants']==y['participants'] for x,y in zip(a,b))
# 저장 번들의 기준 자료·입력 스케일러·무갱신 가중치를 별도로 검사한다.
sys.path.insert(0,str(ROOT/'src'))
import torch
bundle_checks=0
for seed in cfg['seeds']:
 initial=torch.load(OUT/f'seed_{seed}/initial_evaluation/detector_bundle.pt',map_location='cpu',weights_only=False)
 initial_hash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial['model_state'].values())).hexdigest()
 thresholds={cid:c['threshold'] for cid,c in initial['clients'].items()}
 for arm in arms:
  previous={cid:c['reference_ids'] for cid,c in initial['clients'].items()}
  for stage in range(1,cfg['stages']+1):
   folder=OUT/f'seed_{seed}/{arm}/stage_{stage}';r=next(x for x in records if (x['seed'],x['arm'],x['stage'])==(seed,arm,stage))
   bundle=torch.load(folder/'detector_bundle.pt',map_location='cpu',weights_only=False)
   digest=hashlib.sha256(b''.join(v.numpy().tobytes() for v in bundle['model_state'].values())).hexdigest();assert digest==r['state_sha256']
   if arm=='no_update':assert digest==initial_hash
   inputs=load(folder/'training_input_ids.json')
   if inputs:previous={int(cid):ids for cid,ids in inputs.items()}
   for cid,c in bundle['clients'].items():
    assert c['reference_ids']==previous[cid]
    assert set(c['threshold_source_ids'])=={x['id'] for x in groups[f'{cid}/valid']}
    assert all(by[i][1]['y']==0 for i in c['reference_ids'])
    assert np.array_equal(c['input_scaler'].mean_,initial['clients'][cid]['input_scaler'].mean_)
    assert np.array_equal(c['input_scaler'].scale_,initial['clients'][cid]['input_scaler'].scale_)
    assert math.isfinite(c['threshold'])
   with gzip.open(folder/'predictions.csv.gz','rt',encoding='utf-8') as f:prs=list(csv.DictReader(f))
   for domain in ['old','new']:
    q=[x for x in prs if x['domain']==domain];cm=[[0,0],[0,0]]
    for x in q:cm[int(x['label'])][int(float(x['score'])>thresholds[int(x['client'])])]+=1
    assert abs(r['frozen_initial_threshold'][domain]['fpr']-cm[0][1]/sum(cm[0]))<1e-12
    assert abs(r['frozen_initial_threshold'][domain]['tpr']-cm[1][1]/sum(cm[1]))<1e-12
   bundle_checks+=1
summary=[]
for arm in arms:
 rs=[r for r in records if r['arm']==arm and r['stage']==cfg['stages']];item=dict(arm=arm,seeds=len(rs))
 for domain in ['old','new']:
  item[domain]={k:dict(mean=statistics.mean(r['metrics'][domain][k] for r in rs),std=statistics.stdev(r['metrics'][domain][k] for r in rs) if len(rs)>1 else None) for k in ['mean_client_auc','fpr','tpr']}
 for k in ['old_auc_change','old_fpr_change','old_reconstruction_change','cumulative_additional_seconds','cumulative_optimizer_steps','replay_feature_bytes']:
  item[k]=dict(mean=statistics.mean(r[k] for r in rs),std=statistics.stdev(r[k] for r in rs) if len(rs)>1 else None)
 summary.append(item)
report=dict(bundle_checks=bundle_checks,validation='passed',records=len(records),unique_partition_rows=len(by),profile=cfg['profile'],checks=['source hashes','global ID/fingerprint disjointness','scenario PCA fit IDs only development','A/B assignments recomputed from raw CSV','normal-only verified/replay training IDs','uncertainty routing','prediction labels, thresholds, confusion, client AUC','no-update invariance','participation prefix/sample budgets/steps'],summary=summary)
(OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(validation='passed',records=len(records),profile=cfg['profile']),ensure_ascii=False))
