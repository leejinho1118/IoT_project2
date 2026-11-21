"""Independent evidence checks; no final feature arrays are opened."""
from pathlib import Path
import argparse,json,gzip,csv,hashlib,collections
from approval_stress import approval_view
import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix
P=argparse.ArgumentParser();P.add_argument('runs',nargs='+');a=P.parse_args();H=Path(__file__).resolve().parent;ROOT=H.parents[1]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(run):
 out=H/'runs'/run;protocol=json.loads((out/'protocol_frozen.json').read_text());cfg=protocol['config'];data=ROOT/cfg['data']
 assert json.loads((out/'status.json').read_text())['state']=='complete'
 for name,value in json.loads((out/'runtime_hashes.json').read_text()).items():
  assert digest(out/'source_snapshot'/name)==value
  if name in ['run_study.py','federated_core.py','study_data.py','study_metrics.py','transforms.py']:assert digest(H/name)==value
 for name,value in json.loads((out/'upstream_hashes.json').read_text()).items():assert digest(ROOT/name)==value
 assert digest(data/'development.npz')==protocol['development_npz_sha256'];assert digest(data/'final.npz')==protocol['final_npz_sha256']
 with gzip.open(data/'roles.json.gz','rt',encoding='utf-8') as f:roles=json.load(f)
 byid={r['id']:(key,r) for key,rows in roles.items() for r in rows};assert len(byid)==sum(map(len,roles.values()))
 records=json.loads((out/'development_results.json').read_text());expected={(p['name'],s,t) for p in protocol['methods'] for s in cfg['seeds'] for t in [1,2,3]};assert {(r['method'],r['seed'],r['stage']) for r in records}==expected;assert len(records)==len(expected)
 checks=0;prefixes={};initial={}
 for rec in records:
  method,seed,stage=rec['method'],rec['seed'],rec['stage'];sf=out/method/f'seed_{seed}'/f'stage_{stage}';start=json.loads((sf.parent/'start.json').read_text());initial.setdefault(seed,start['initial_weights_sha256']);assert initial[seed]==start['initial_weights_sha256']
  if method in ['no_update','calibration_only','reference_only']:assert rec['weights_sha256']==initial[seed]
  routing=json.loads((sf/'routing_and_input_ids.json').read_text())
  for r in routing:
   cid=r['client'];normal=r['normal_ids'];train=r['training_ids'];view,audit=approval_view(roles[f'{cid}/candidate_{stage-1}'],cfg['approval_stress'],seed,stage,cid);assert audit==r['approval_audit'];approved={v['id'] for v in view if v['y']==0};assert set(normal)<=approved;assert set(train)<=(approved|{v['id'] for v in roles[f'{cid}/train']});assert set(r['training_actual_attack_ids'])=={i for i in train if byid[i][1]['y']!=0};assert len(normal)<=cfg['update_cap'];assert len(train)<=cfg['update_cap'];assert all(byid[i][1]['y']==0 or i in r['training_actual_attack_ids'] or i in normal for i in normal+train)
   assert all(byid[i][0]==f'{cid}/candidate_{stage-1}' for i in normal)
   assert all(byid[i][0] in [f'{cid}/candidate_{stage-1}',f'{cid}/train'] for i in train)
   assert all(byid[i][1]['y']==1 for i in r['attack_ids_excluded']);assert not set(train)&set(r['attack_ids_excluded'])
   k=(seed,stage,cid);prefixes.setdefault(k,normal);assert prefixes[k]==normal
  with gzip.open(sf/'development_predictions.csv.gz','rt',encoding='utf-8') as f:pred=list(csv.DictReader(f))
  for domain in ['old','new']:
   rr=[r for r in pred if r['domain']==domain];y=np.array([int(r['label']) for r in rr]);s=np.array([float(r['score']) for r in rr]);t=np.array([float(r['threshold']) for r in rr]);p=(s>t).astype(int);assert np.isfinite(s).all();assert np.array_equal(p,[int(r['prediction']) for r in rr]);m=rec['metrics'][domain];cm=confusion_matrix(y,p,labels=[0,1]);assert cm.tolist()==m['confusion_matrix'];tn,fp,fn,tp=cm.ravel();assert abs(fp/(tn+fp)-m['fpr'])<1e-12;assert abs(tp/(tp+fn)-m['tpr'])<1e-12
   auc=[]
   for cid in sorted({r['client'] for r in rr}):
    cc=[r for r in rr if r['client']==cid];yy=[int(r['label']) for r in cc];ss=[float(r['score']) for r in cc];assert len(set(yy))==2;auc.append(roc_auc_score(yy,ss))
   assert abs(np.mean(auc)-m['mean_client_auc'])<1e-12;checks+=len(auc)
   for r in rr:
    key,_=byid[r['id']];assert key.split('/')[1] in ['decision_old' if domain=='old' else f'decision_new_{stage-1}','decision_attack']
  if rec['weight_training_performed']:
   training=json.loads((sf/'training.json').read_text())['rounds'];steps=0
   for rnd in training:
    assert len(rnd['participants'])==max(1,int(.5*cfg['clients']));assert len(set(rnd['participants']))==len(rnd['participants'])
    for local in rnd['local']:
     assert local['rows']<=cfg['update_cap'];assert local['epochs']<=cfg['local_epochs'];assert local['optimizer_steps']==local['epochs']*int(np.ceil(local['rows']/cfg['batch_size']));steps+=local['optimizer_steps']
   if rec['checkpoint']['enabled'] and rec['checkpoint']['accepted']:
    cp=rec['checkpoint'];assert cp['selected_round']>0;selected=[x for x in cp['candidate_checks'] if x['round']==cp['selected_round']][0];assert selected['guard_pass'];assert selected['state_sha256']==rec['weights_sha256']
 result=dict(state='passed',run=run,records=len(records),client_aucs_independently_recomputed=checks,data_and_upstream_hashes=True,normal_only_training=cfg['approval_stress']=='query25',declared_approval_perturbation_verified=True,current_stage_release_ids=True,matched_initial_and_candidate_inputs=True,optimizer_budget_and_checkpoint_hashes=True,final_features_opened=False)
 target=out/'independent_approval_validation.json'
 if target.exists():assert json.loads(target.read_text())==result
 else:target.write_text(json.dumps(result,indent=2),encoding='utf-8')
 return result
print(json.dumps([check(r) for r in a.runs],indent=2))
