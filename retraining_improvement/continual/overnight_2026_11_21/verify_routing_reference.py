"""Verify pre-update detection, approval routing and deployed reference lineage."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import argparse,json
import torch,numpy as np
import federated_core as C
from study_data import load_development,activate
from approval_stress import approval_view
P=argparse.ArgumentParser();P.add_argument('runs',nargs='+');P.add_argument('--root',default='runs');a=P.parse_args();H=Path(__file__).resolve().parent;ROOT=H.parents[1];torch.set_num_threads(1)
for name in a.runs:
 out=H/a.root/name;protocol=json.loads((out/'protocol_frozen.json').read_text());cfg=protocol['config'];clients,_=load_development(ROOT/cfg['data']);clients=clients[:3] if cfg['profile']=='smoke' else clients;C.configure(cfg,'cpu');sources=json.loads((out/'initial_sources.json').read_text());policies={p['name']:p for p in protocol['methods']};records=json.loads((out/'development_results.json').read_text());detections=0;refs=0;false_approved=0
 for rec in records:
  seed,stage,method=rec['seed'],rec['stage'],rec['method'];sf=out/method/f'seed_{seed}'/f'stage_{stage}';prevpath=ROOT/sources[str(seed)]['path']/'detector_bundle.pt' if stage==1 else sf.parent/f'stage_{stage-1}'/'detector_bundle.pt';prev=torch.load(prevpath,map_location='cpu',weights_only=False);current=torch.load(sf/'detector_bundle.pt',map_location='cpu',weights_only=False);model=C.StudySAE(**prev['architecture']);model.load_state_dict(prev['model_state']);model.eval();activate(clients,stage);routing={r['client']:r for r in json.loads((sf/'routing_and_input_ids.json').read_text())};fp=0;total=0;policy=policies[method]
  for c in clients:
   cid=c['id'];row=routing[cid];items=c['candidate_stages'][stage-1];d=prev['clients'][cid];c['scaler']=d['input_scaler'];saved={r['id']:r for r in row['detections']};assert set(saved)=={r['id'] for r in items};score=d['cen'].get_density(C.B.encode(model,c,items));assert np.isfinite(score).all();assert np.allclose(score,[saved[r['id']]['score'] for r in items],rtol=2e-4,atol=2e-6)
   for item in saved.values():assert item['anomaly_flag']==(item['score']>d['threshold'])
   if cfg.get('approval_stress'):
    view,audit=approval_view(items,cfg['approval_stress'],seed,stage,cid);assert audit==row['approval_audit'];queried=set(audit['query_ids']);assert all(x.get('queried')==(x['id'] in queried) for x in row['detections']);false_approved+=audit['false_normal_approvals']
   else:view=items
   pool=[r for r in view if r['y']==0];fp+=sum(saved[r['id']]['anomaly_flag'] for r in pool);total+=len(pool)
   if len(pool)>cfg['update_cap']:
    ix=np.random.default_rng(4422026+stage*100+cid).choice(len(pool),cfg['update_cap'],replace=False);pool=[pool[int(i)] for i in ix]
   assert row['normal_ids']==[r['id'] for r in pool]
   replace=(policy.get('no_train') and not policy.get('keep_reference')) or (rec['weight_training_performed'] and rec['checkpoint']['accepted'])
   expected=row['training_ids'] if replace else prev['clients'][cid]['reference_ids'];assert current['clients'][cid]['reference_ids']==expected
   if policy.get('frozen'):assert current['clients'][cid]['threshold']==prev['clients'][cid]['threshold']
   refs+=1;detections+=len(items)
  assert abs(fp/total-rec['normal_candidate_fpr'])<1e-12
 result=dict(state='passed',run=name,records=len(records),candidate_scores_checked=detections,client_reference_lineages_checked=refs,detection_used_previous_published_model=True,every_candidate_scored_before_approval=True,normal_selection_exact_cap_and_IDs=True,reference_comes_from_accepted_training_or_previous_published_bundle=True,declared_false_normal_approvals_across_arms_stages=false_approved,final_features_opened=False)
 p=out/'independent_routing_and_reference_validation.json'
 if p.exists():assert json.loads(p.read_text())==result
 else:p.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result))
