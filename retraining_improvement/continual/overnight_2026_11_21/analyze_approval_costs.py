"""Quantify actual query and false-normal fractions; do not assume target 5% is exact."""
from pathlib import Path
import json,gzip,csv,collections,os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch
import federated_core
H=Path(__file__).resolve().parent;A=json.loads((H/'analysis_v1/analysis.json').read_text());ROOT=H.parents[1];records=[];role_cache={}
for run in A['runs']:
 if run['kind'] not in ['query25','poison05']:continue
 out=H/'runs'/run['run'];cfg=json.loads((out/'protocol_frozen.json').read_text())['config'];data=ROOT/cfg['data']
 if str(data) not in role_cache:
  with gzip.open(data/'roles.json.gz','rt',encoding='utf-8') as f:roles=json.load(f)
  role_cache[str(data)]={r['id']:r['y'] for rows in roles.values() for r in rows}
 truth=role_cache[str(data)]
 for rec in json.loads((out/'development_results.json').read_text()):
  sf=out/rec['method']/f"seed_{rec['seed']}"/f"stage_{rec['stage']}";routing=json.loads((sf/'routing_and_input_ids.json').read_text());alln=sum(r['approval_audit']['all_candidate_rows'] for r in routing);query=sum(r['approval_audit']['queried_rows'] for r in routing);approved=sum(r['approval_audit']['approved_rows'] for r in routing);bad=sum(r['approval_audit']['false_normal_approvals'] for r in routing);train=[i for r in routing for i in r['training_ids']];trainbad=sum(truth[i]!=0 for i in train);obj=torch.load(sf/'detector_bundle.pt',map_location='cpu',weights_only=False);ref=[i for d in obj['clients'].values() for i in d['reference_ids']];refbad=sum(truth[i]!=0 for i in ref)
  records.append(dict(data=run['data'],setting=run['kind'],method=rec['method'],seed=rec['seed'],stage=rec['stage'],candidate_rows=alln,queried_rows=query,query_fraction=query/alln,approved_rows=approved,false_normal_approvals=bad,approval_contamination=bad/approved,training_rows=len(train),actual_attacks_in_training=trainbad,training_contamination=trainbad/len(train),CEN_reference_rows=len(ref),actual_attacks_in_CEN_reference=refbad,CEN_reference_contamination=refbad/len(ref),weight_training_performed=rec['weight_training_performed']))
summary=[];groups=collections.defaultdict(list)
for r in records:groups[(r['data'],r['setting'],r['method'])].append(r)
for k,rr in groups.items():
 entry=dict(zip(['data','setting','method'],k));entry['seed_stage_observations']=len(rr)
 for metric in ['query_fraction','approval_contamination','training_contamination','CEN_reference_contamination']:entry[metric+'_mean']=sum(r[metric] for r in rr)/len(rr);entry[metric+'_min']=min(r[metric] for r in rr);entry[metric+'_max']=max(r[metric] for r in rr)
 summary.append(entry)
obj=dict(records=records,summary=summary,interpretation='Ratios average seed-stage observations, not extra independent traffic acquisitions. Candidate labels are shared among methods. Protected new validation/calibration/development normals are additionally required. When updates are rejected/skipped, actual CEN reference contamination may differ from proposed training contamination.')
p=H/'analysis_v1/approval_cost_analysis.json';assert not p.exists();p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
if records:
 with p.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
print('Approval records independently counted:',len(records))

