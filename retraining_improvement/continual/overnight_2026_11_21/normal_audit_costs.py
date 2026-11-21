"""Count explicit oracle-label assumptions and feature payload, not process RSS."""
from pathlib import Path
import argparse,json,gzip,collections
P=argparse.ArgumentParser();P.add_argument('runs',nargs='+');a=P.parse_args();H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def audit(name):
 out=H/'normal_runs'/name;cfg=json.loads((out/'protocol_frozen.json').read_text())['config'];data=ROOT/cfg['data']
 with gzip.open(data/'roles.json.gz','rt',encoding='utf-8') as f:roles=json.load(f)
 counts={key:len(rows) for key,rows in roles.items()};stages=[]
 for stage in [1,2,3]:
  labels=collections.Counter()
  for key,n in counts.items():
   role=key.split('/')[1]
   if role in [f'candidate_{stage-1}',f'new_valid_{stage-1}',f'calib_new_{stage-1}',f'decision_new_{stage-1}']:labels[role]+=n
  stages.append(dict(stage=stage,shared_available_labelled_rows=dict(labels),protected_new_normal_rows_beyond_candidate=sum(n for k,n in labels.items() if not k.startswith('candidate'))))
 records=json.loads((out/'development_results.json').read_text());details=[]
 for rec in records:
  sf=out/rec['method']/f"seed_{rec['seed']}"/f"stage_{rec['stage']}";routing=json.loads((sf/'routing_and_input_ids.json').read_text());rrows=sum(r['replay_rows'] for r in routing)
  detail=dict(method=rec['method'],seed=rec['seed'],stage=rec['stage'],real_training_rows=sum(len(r['training_ids']) for r in routing),old_rows_in_training=rrows,replay_feature_bytes_float64=rrows*cfg['input_dim']*8,bank_feature_bytes_float64=cfg['clients']*cfg['capacity']*cfg['input_dim']*8,weight_training_performed=rec['weight_training_performed'],cumulative_optimizer_steps=rec['cumulative_optimizer_steps'],cumulative_training_seconds=rec['cumulative_training_seconds'])
  if routing[0].get('approval_audit'):detail['candidate_queries']=sum(r['approval_audit']['queried_rows'] for r in routing);detail['false_normal_approvals']=sum(r['approval_audit']['false_normal_approvals'] for r in routing);detail['actual_attacks_in_training']=sum(len(r['training_actual_attack_ids']) for r in routing)
  else:detail['candidate_queries_in_instrumented_runner']=sum(len(r['detections']) for r in routing)
  details.append(detail)
 result=dict(run=name,stages=stages,methods=details,cost_interpretation='Do not multiply shared verified materials by number of arms. Frozen baseline does not need candidate oracle labels to deploy; runner calculates routing for matched bookkeeping. Query25 is candidate-only saving: clean new validation, calibration, decision labels remain additionally required. Replay payload is logical feature content, NOT RSS or measured network traffic. The implementation materializes initial banks for common instrumentation even for no replay policies.')
 p=out/'label_and_payload_audit.json'
 if not p.exists():p.write_text(json.dumps(result,indent=2),encoding='utf-8')
 return dict(run=name,rows=len(details),stages=stages)
print(json.dumps([audit(n) for n in a.runs],indent=2))
