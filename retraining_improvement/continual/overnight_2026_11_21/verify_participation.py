"""Replay deterministic participant prefixes and actual per-client row budgets."""
from pathlib import Path
import argparse,json,random,collections
P=argparse.ArgumentParser();P.add_argument('runs',nargs='+');P.add_argument('--root',default='runs');a=P.parse_args();H=Path(__file__).resolve().parent
for name in a.runs:
 out=H/a.root/name;cfg=json.loads((out/'protocol_frozen.json').read_text())['config'];ids=json.loads((out/'client_ids.json').read_text());records=json.loads((out/'development_results.json').read_text());rounds=0;counts={};cumulative={}
 for r in records:
  sf=out/r['method']/f"seed_{r['seed']}"/f"stage_{r['stage']}";routing=json.loads((sf/'routing_and_input_ids.json').read_text());k=(r['seed'],r['stage']);rows={x['client']:len(x['training_ids']) for x in routing};counts.setdefault(k,rows);assert counts[k]==rows
  ck=(r['method'],r['seed']);cumulative.setdefault(ck,0)
  if not r['weight_training_performed']:
   assert cumulative[ck]==r['cumulative_optimizer_steps'];continue
  history=json.loads((sf/'training.json').read_text())['rounds'];rng=random.Random(r['seed']+100000+r['stage'])
  stage_steps=sum(local['optimizer_steps'] for rd in history for local in rd['local']);cumulative[ck]+=stage_steps;assert cumulative[ck]==r['cumulative_optimizer_steps']
  for rd in history:
   expected=[ids[i] for i in rng.sample(range(len(ids)),max(1,int(.5*len(ids))))];assert rd['participants']==expected
   assert [x['client'] for x in rd['local']]==expected;assert all(x['rows']==rows[x['client']] for x in rd['local']);rounds+=1
 result=dict(state='passed',run=name,participant_rounds_replayed=rounds,same_ordered_participant_prefix=True,same_actual_row_budget_per_client_and_stage=True,early_stop_and_trigger_may_change_executed_round_count=True)
 path=out/'independent_participation_validation.json'
 if path.exists():assert json.loads(path.read_text())==result
 else:path.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result))
