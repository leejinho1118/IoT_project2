"""Locked final evaluation after family freeze. No fitting or model selection."""
from pathlib import Path
import argparse,os,json,hashlib,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch
import numpy as np
import federated_core as C
from study_data import load_development,load_final
from study_metrics import evaluate
P=argparse.ArgumentParser();P.add_argument('runs',nargs='+');a=P.parse_args();H=Path(__file__).resolve().parent;ROOT=H.parents[1];FREEZE=H/'family_selection_frozen.json'
assert FREEZE.exists();F=json.loads(FREEZE.read_text());assert F['state']=='frozen';torch.set_num_threads(1)
def evaluate_run(name):
 out=H/'normal_runs'/name;assert json.loads((out/'status.json').read_text())['state']=='complete';assert (out/'independent_development_validation.json').exists() or (out/'independent_approval_validation.json').exists()
 protocol=json.loads((out/'protocol_frozen.json').read_text());cfg=protocol['config'];assert any(r['mode']==cfg['mode'] and r['loss']==cfg['shrink_kind'] for r in F['selected']);assert cfg['profile']=='study'
 target=out/'final_evaluation_v1';assert not target.exists(),'Existing final evaluation preserved';target.mkdir()
 data=ROOT/cfg['data'];assert hashlib.sha256((data/'final.npz').read_bytes()).hexdigest()==protocol['final_npz_sha256']
 clients,roles=load_development(data);final=load_final(data,FREEZE,target/'final_access.json');C.configure(cfg,'cpu');rows=[];start=time.perf_counter()
 records=json.loads((out/'development_results.json').read_text())
 for rec in records:
  path=out/rec['bundle'];obj=torch.load(path,map_location='cpu',weights_only=False);model=C.StudySAE(**obj['architecture']);model.load_state_dict(obj['model_state']);model.eval();assert C.whash(model)==rec['weights_sha256']
  det={}
  for c in clients:
   item=obj['clients'][c['id']];c['scaler']=item['input_scaler'];det[c['id']]=item
  pred=target/f"{rec['method']}_seed{rec['seed']}_stage{rec['stage']}.csv.gz"
  metrics=evaluate(model,clients,det,final=final,save=pred)
  rows.append(dict(method=rec['method'],seed=rec['seed'],stage=rec['stage'],metrics=metrics,weights_sha256=rec['weights_sha256'],bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cumulative_training_seconds=rec['cumulative_training_seconds'],cumulative_optimizer_steps=rec['cumulative_optimizer_steps'],checkpoint=rec['checkpoint'],prediction_file=pred.name))
  (target/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 proof=dict(state='complete',run=name,records=len(rows),selection_sha256=hashlib.sha256(FREEZE.read_bytes()).hexdigest(),fitting_performed=False,thresholds_changed=False,final_npz_sha256=protocol['final_npz_sha256'],seconds=time.perf_counter()-start,initial_cost_counted_once_per_seed=True,final_conflicting_label_rows_retained=sum(r.get('label_conflict',False) for rr in final.values() for r in rr))
 (target/'complete.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');return proof
print(json.dumps([evaluate_run(name) for name in a.runs],indent=2))
