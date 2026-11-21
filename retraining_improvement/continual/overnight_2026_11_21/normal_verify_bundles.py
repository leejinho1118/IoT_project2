"""Recompute normal calibration and CEN references from saved bundles on CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
import federated_core as C
from transforms import InputTransform
from study_data import load_development,activate
P=argparse.ArgumentParser();P.add_argument('runs',nargs='+');a=P.parse_args();H=Path(__file__).resolve().parent;ROOT=H.parents[1];torch.set_num_threads(1)
def check(name):
 out=H/'normal_runs'/name;pr=json.loads((out/'protocol_frozen.json').read_text());cfg=pr['config'];data=ROOT/cfg['data'];clients,roles=load_development(data);C.configure(cfg,'cpu');records=json.loads((out/'development_results.json').read_text());checks=0
 if cfg['profile']=='smoke':clients=clients[:3]
 for rec in records:
  activate(clients,rec['stage']);obj=torch.load(out/rec['bundle'],map_location='cpu',weights_only=False);model=C.StudySAE(**obj['architecture']);model.load_state_dict(obj['model_state']);model.eval();assert C.whash(model)==rec['weights_sha256']
  for c in clients:
   cid=c['id'];d=obj['clients'][cid];c['scaler']=d['input_scaler'];fitted=InputTransform(cfg['mode']).fit(C.payload(c['train']));assert np.array_equal(fitted.scaler.mean_,c['scaler'].scaler.mean_);assert np.array_equal(fitted.scaler.scale_,c['scaler'].scaler.scale_)
   byid={r['id']:r for k in ['train','calib_old','calib_new'] for r in c[k]}
   # A rollback can retain a reference from any earlier released stage, never a future stage.
   for st in range(1,rec['stage']+1):
    activate([c],st)
    byid.update({r['id']:r for r in c['candidate_stages'][st-1]})
   activate([c],rec['stage'])
   reference=[byid[i] for i in d['reference_ids']];assert all(r['y']==0 for r in reference);fresh=C.B.CentroidBasedOneClassClassifier();fresh.fit(C.B.encode(model,c,reference))
   old=C.B.encode(model,c,c['calib_old']);qold=float(np.quantile(fresh.get_density(old),.95));assert np.isclose(qold,d['old_threshold'],rtol=2e-4,atol=2e-6), (name,rec['method'],rec['stage'],cid,qold,d['old_threshold']);assert np.allclose(fresh.get_density(old),d['cen'].get_density(old),rtol=2e-4,atol=2e-6)
   assert d['threshold_old_ids']==[r['id'] for r in c['calib_old']]
   if rec['method']=='no_update':assert not d['threshold_new_ids'];expected=qold
   else:
    assert d['threshold_new_ids']==[r['id'] for r in c['calib_new']];qnew=float(np.quantile(fresh.get_density(C.B.encode(model,c,c['calib_new'])),.95));assert np.isclose(qnew,d['new_threshold'],rtol=2e-4,atol=2e-6);expected=max(qold,qnew)
   assert np.isclose(expected,d['threshold'],rtol=2e-4,atol=2e-6);checks+=1
 result=dict(state='passed',run=name,bundles=len(records),client_detectors_recomputed=checks,scaler_exact_initial_only=True,CEN_refit_matches=True,thresholds_recomputed=True,no_future_or_final_features_in_detector=True,final_npz_opened=False)
 p=out/'independent_bundle_validation.json'
 if p.exists():assert json.loads(p.read_text())==result
 else:p.write_text(json.dumps(result,indent=2),encoding='utf-8')
 return result
print(json.dumps([check(n) for n in a.runs],indent=2))

