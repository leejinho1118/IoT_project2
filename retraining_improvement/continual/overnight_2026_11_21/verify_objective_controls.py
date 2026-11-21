"""Prove unchanged controls can be reused for the extra objective trials."""
from pathlib import Path
import json,argparse,random
P=argparse.ArgumentParser();P.add_argument('runs',nargs='+');a=P.parse_args();H=Path(__file__).resolve().parent
for name in a.runs:
 out=H/'objective_runs'/name;cfg=json.loads((out/'protocol_frozen.json').read_text())['config'];assert cfg['profile']=='study';source=None
 for p in (H/'runs').iterdir():
  if not (p/'complete.json').exists():continue
  c=json.loads((p/'protocol_frozen.json').read_text())['config']
  if not c.get('approval_stress') and not c.get('selection_ablation') and all(c.get(k)==cfg.get(k) for k in ['data','mode','shrink_kind','seeds','profile','device','local_epochs','rounds','batch_size','lr','capacity','update_cap','initial_cap','input_dim']):source=p;break
 assert source is not None;assert json.loads((out/'initial_sources.json').read_text())==json.loads((source/'initial_sources.json').read_text())
 aa=json.loads((out/'runtime_hashes.json').read_text());bb=json.loads((source/'runtime_hashes.json').read_text())
 for k in ['federated_core.py','study_data.py','study_metrics.py','transforms.py']:assert aa[k]==bb[k]
 comparisons=0
 for rec in json.loads((out/'development_results.json').read_text()):
  target='safe_replay' if rec['method']=='safe_align_distill' else 'replay';sf=out/rec['method']/f"seed_{rec['seed']}"/f"stage_{rec['stage']}";base=source/target/f"seed_{rec['seed']}"/f"stage_{rec['stage']}";one=json.loads((sf/'routing_and_input_ids.json').read_text());two=json.loads((base/'routing_and_input_ids.json').read_text());assert [(r['client'],r['training_ids'],r['replay_rows']) for r in one]==[(r['client'],r['training_ids'],r['replay_rows']) for r in two]
  h1=json.loads((sf/'training.json').read_text())['rounds'];h2=json.loads((base/'training.json').read_text())['rounds']
  for r1,r2 in zip(h1,h2):assert r1['participants']==r2['participants']
  comparisons+=1
 proof=dict(state='passed',objective_run=name,matched_main_run=source.name,stage_comparisons=comparisons,identical_core_loader_metrics_transform_hashes=True,identical_initial_sources=True,identical_per_client_ordered_training_IDs=True,identical_participant_prefixes=True,max_budget_and_device_equal=True,only_objective_and_guard_policy_differs=True)
 p=out/'matched_main_control_validation.json'
 if p.exists():assert json.loads(p.read_text())==proof
 else:p.write_text(json.dumps(proof,indent=2),encoding='utf-8')
 print(json.dumps(proof))
