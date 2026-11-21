"""Independent replay of guarded round selection using current-stage clean development."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import argparse,json,copy
import torch,numpy as np
import normal_gate_core as C
from study_data import load_development,activate
from normal_gate_core import evaluate
P=argparse.ArgumentParser();P.add_argument('runs',nargs='+');a=P.parse_args();H=Path(__file__).resolve().parent;ROOT=H.parents[1];torch.set_num_threads(1)
def score(m):return -.5*sum(np.log(m[d]['normal_reconstruction_mse']+1e-12) for d in ['old','new'])
def eligible(m,p,i):
 return m['old']['normal_reconstruction_mse']<=i['old']['normal_reconstruction_mse']*1.05 and all(m[d]['fpr']<=p[d]['fpr']+.02 for d in ['old','new'])
def check(name):
 out=H/'runs'/name;cfg=json.loads((out/'protocol_frozen.json').read_text())['config'];clients,_=load_development(ROOT/cfg['data']);clients=clients[:3] if cfg['profile']=='smoke' else clients;C.configure(cfg,'cpu');sources=json.loads((out/'initial_sources.json').read_text());records=json.loads((out/'development_results.json').read_text());count=0;rollbacks=0
 for rec in records:
  cp=rec['checkpoint']
  if not cp.get('enabled') or cp.get('trigger_skipped'):continue
  sf=out/rec['method']/f"seed_{rec['seed']}"/f"stage_{rec['stage']}";activate(clients,rec['stage']);start=json.loads((sf.parent/'start.json').read_text());initial=start['initial_old_only']
  prevpath=ROOT/sources[str(rec['seed'])]['path']/'detector_bundle.pt' if rec['stage']==1 else sf.parent/f"stage_{rec['stage']-1}"/'detector_bundle.pt'
  prev=torch.load(prevpath,map_location='cpu',weights_only=False);model=C.StudySAE(**prev['architecture']);model.load_state_dict(prev['model_state']);model.eval();det=copy.deepcopy(prev['clients'])
  for c in clients:
   cid=c['id'];c['scaler']=det[cid]['input_scaler'];cen=det[cid]['cen'];old=np.quantile(cen.get_density(C.B.encode(model,c,c['calib_old'])),.95);new=np.quantile(cen.get_density(C.B.encode(model,c,c['calib_new'])),.95);det[cid]['threshold']=max(old,new)
  previous=evaluate(model,clients,det);best=score(previous);chosen=0;history=json.loads((sf/'training.json').read_text())['rounds']
  for row,audit in zip(history,cp['candidate_checks']):
   valid=eligible(row['development'],previous,initial);value=score(row['development']);assert abs(value-audit['score'])<1e-12;assert valid==audit['guard_pass']
   if valid and value>best+1e-4:best=value;chosen=row['round']
  assert chosen==cp['selected_round'],(name,rec['method'],rec['stage'],chosen,cp['selected_round']);current=torch.load(sf/'detector_bundle.pt',map_location='cpu',weights_only=False)
  if chosen:
   roundobj=torch.load(sf/'global_rounds'/f'round_{chosen}.pt',map_location='cpu',weights_only=False);assert roundobj['weights_sha256']==rec['weights_sha256'];routing=json.loads((sf/'routing_and_input_ids.json').read_text());expected={r['client']:r['training_ids'] for r in routing}
   for c in clients:assert current['clients'][c['id']]['reference_ids']==expected[c['id']]
  else:
   rollbacks+=1;assert current['weights_sha256']==prev['weights_sha256']
   for c in clients:assert current['clients'][c['id']]['reference_ids']==prev['clients'][c['id']]['reference_ids']
  count+=1
 result=dict(state='passed',run=name,guarded_updates_replayed=count,rollbacks=rollbacks,weights_and_reference_restored_together=True,current_stage_recalibration_of_previous_model=True,final_features_opened=False)
 p=out/'independent_normal_guard_validation.json'
 if p.exists():assert json.loads(p.read_text())==result
 else:p.write_text(json.dumps(result,indent=2),encoding='utf-8')
 return result
print(json.dumps([check(n) for n in a.runs],indent=2))

