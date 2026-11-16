from pathlib import Path
import json,gzip,csv,math,hashlib,statistics,itertools
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parents[1]; O=R/'continual/output/full_v1'
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
config=load(O/'config.json'); results=load(O/'results.json'); status=load(O/'status.json')
assert status['state']=='complete' and len(results)==45
assert len({(x['seed'],x['arm'],x['stage']) for x in results})==45
assert {(x['seed'],x['arm'],x['stage']) for x in results}==set(itertools.product(config['seeds'],['no_update','new_only','replay'],[1,2,3]))
for group,prefix in [('source_hashes',''),('extension_source_hashes','continual/')]:
 for name,h in status[group].items():assert hashlib.sha256((R/(prefix+name)).read_bytes()).hexdigest()==h
with gzip.open(O/'split_manifest.json.gz','rt',encoding='utf-8') as f:groups=json.load(f)
ids=set(); fps=set(); by={}
for name,rows in groups.items():
 for row in rows:
  assert row['id'] not in ids and row['fingerprint'] not in fps
  ids.add(row['id']);fps.add(row['fingerprint']);by[row['id']]=(name,row)
for r in results:
 p=O/f'seed_{r["seed"]}'/r['arm']/f'stage_{r["stage"]}'
 with gzip.open(p/'predictions.csv.gz','rt',encoding='utf-8') as f:prs=list(csv.DictReader(f))
 assert len({x['id'] for x in prs})==len(prs)
 for x in prs:
  role,row=by[x['id']];assert role.endswith(('/old_normal','/old_attack')) if x['domain']=='old' else role.endswith(('/new_normal','/new_attack'))
  assert int(x['label'])==row['y'] and math.isfinite(float(x['score']))
 for d in ['old','new']:
  q=[x for x in prs if x['domain']==d]; tn=sum(x['label']=='0' and x['prediction']=='0' for x in q);fp=sum(x['label']=='0' and x['prediction']=='1' for x in q);fn=sum(x['label']=='1' and x['prediction']=='0' for x in q);tp=sum(x['label']=='1' and x['prediction']=='1' for x in q)
  m=r['metrics'][d];assert m['confusion_matrix']==[[tn,fp],[fn,tp]];assert abs(m['fpr']-fp/(tn+fp))<1e-12 and abs(m['tpr']-tp/(tp+fn))<1e-12
  clients=sorted({x['client'] for x in q});assert len(clients)==50
  aucs=[roc_auc_score([int(x['label']) for x in q if x['client']==c],[float(x['score']) for x in q if x['client']==c]) for c in clients]
  assert abs(statistics.mean(aucs)-m['mean_client_auc'])<1e-12
 for x in load(p/'detections_before_update.json')['items']:assert '/candidate_' in by[x['id']][0] and math.isfinite(x['score'])
for seed in config['seeds']:
 base=load(O/f'seed_{seed}/initial_evaluation/metrics.json')
 no=[x for x in results if x['seed']==seed and x['arm']=='no_update'];assert len({x['state_sha256'] for x in no})==1 and all(x['metrics']==base for x in no)
 for stage in [1,2,3]:
  routing=load(O/f'seed_{seed}/verified_stage_{stage}/routing.json')['items']
  for x in routing:
   role,row=by[x['id']];assert '/candidate_' in role;assert (x['verified']=='normal')==(row['y']==0)
  a,b=[load(O/f'seed_{seed}/{arm}/stage_{stage}/training/training.json') for arm in ['new_only','replay']]
  for x,y in zip(a['rounds'],b['rounds']):
   assert x['participants']==y['participants'] and len(set(x['participants']))==25
   assert [(v['client'],v['rows']) for v in x['local']]==[(v['client'],v['rows']) for v in y['local']]
  for t in [a,b]:
   assert len(t['rounds'])<=20
   for rr in t['rounds']:
    assert math.isfinite(rr['global_loss'])
    for v in rr['local']:assert 1<=v['epochs']<=100 and v['optimizer_steps']==v['epochs']*math.ceil(v['rows']/12)
  for arm in ['new_only','replay']:
   x=next(x for x in results if (x['seed'],x['arm'],x['stage'])==(seed,arm,stage));assert not x['update_skipped'] and x['state_sha256']!=no[0]['state_sha256']
   if arm=='replay':assert x['replay_samples_this_stage']>0
summary=[]
for arm in ['no_update','new_only','replay']:
 rows=[x for x in results if x['arm']==arm and x['stage']==3]; item={'arm':arm}
 for domain in ['old','new']:
  item[domain]={k:{'mean':statistics.mean(x['metrics'][domain][k] for x in rows),'std':statistics.stdev(x['metrics'][domain][k] for x in rows)} for k in ['mean_client_auc','fpr','tpr']}
 for k in ['old_auc_change','old_fpr_change','cumulative_additional_seconds','initial_training_seconds','replay_bank_feature_bytes']:
  vals=[x[k] for x in rows];item[k]={'mean':statistics.mean(vals),'std':statistics.stdev(vals)}
 steps=[]
 for seed in config['seeds']:
  steps.append(0 if arm=='no_update' else sum(v['optimizer_steps'] for stage in [1,2,3] for rr in load(O/f'seed_{seed}/{arm}/stage_{stage}/training/training.json')['rounds'] for v in rr['local']))
 item['update_optimizer_steps']={'mean':statistics.mean(steps),'std':statistics.stdev(steps)};summary.append(item)
output={'validation':'passed','records':45,'unique_partition_rows':len(ids),'checks':['45 unique complete evaluations','Source and extension hashes unchanged','Global row-ID and feature-fingerprint disjointness','Prediction labels/roles and confusion matrices independently checked','All 4500 client AUCs recomputed','No-update metrics and weights invariant','Verified normal routing only','Common participation prefix and per-client sample budgets; actual early-stop steps recorded'],'summary_stage_3':summary,'initial_cost_total_seconds':sum(next(x['initial_training_seconds'] for x in results if x['seed']==s) for s in config['seeds'])}
(O/'validation.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(output,ensure_ascii=False,indent=2))
