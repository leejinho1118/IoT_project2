"""CPU normal-reference controls reuse only completed main initial models."""
from pathlib import Path
import json,subprocess,sys,datetime,time,os
H=Path(__file__).resolve().parent;ROOT=H.parents[1];S=H/'reference_control_status_v1.json'
if S.exists():raise RuntimeError('Preserve existing control worker')
def now():return datetime.datetime.now(datetime.timezone.utc)
def save():S.write_text(json.dumps(state,indent=2),encoding='utf-8')
def call(script,name,extra=None):
 cmd=[sys.executable,'-X','utf8','-B',str(H/script)]+([name] if extra is None else extra)
 with (H/(name+'_control_worker.log')).open('a',encoding='utf-8') as log:p=subprocess.run(cmd,cwd=H,stdout=log,stderr=log)
 if p.returncode:raise RuntimeError(f'{script} failed for {name}: {p.returncode}')
state=dict(state='running',pid=os.getpid(),trained=[],final_evaluated=[]);save();F=json.loads((H/'family_selection_frozen.json').read_text());recipe=F['selected'][1]
try:
 while now()<datetime.datetime(2026,11,20,22,15,tzinfo=datetime.timezone.utc):
  main=json.loads((H/'replication_status_v1.json').read_text());final_allowed=main['state'] in ['complete','time_budget_stop'];eligible=[]
  for folder in sorted((H/'runs').iterdir()):
   if not (folder/'complete.json').exists():continue
   cfg=json.loads((folder/'protocol_frozen.json').read_text())['config']
   if cfg['profile']!='study' or cfg['mode']!=recipe['mode'] or cfg['shrink_kind']!=recipe['loss'] or cfg.get('approval_stress') or cfg['local_epochs']!=20:continue
   if len(cfg['seeds'])!=1:raise RuntimeError('Expected singleton seed')
   seed=cfg['seeds'][0];data=Path(cfg['data']).name;tag={'raw_blocks_v2':'nb','rt_thing_to_mqtt_v1':'rt_tm','rt_mqtt_to_thing_v1':'rt_mt'}[data];name=f'{tag}_reference_only_new_seed{seed}';eligible.append(name);out=H/'control_runs'/name
   if name not in state['trained']:
    state.update(current=name,operation='CPU_reference_control');save()
    if not (out/'complete.json').exists():call('run_reference_control.py',name,['--data',str(H/data),'--run-id',name,'--mode',recipe['mode'],'--loss',recipe['loss'],'--device','cpu','--seeds',str(seed)])
    for script in ['control_verify_study.py','control_verify_bundles.py','control_audit_costs.py']:call(script,name)
    for stage in [1,2,3]:
     control=out/'reference_only_new'/f'seed_{seed}'/f'stage_{stage}';baseline=folder/'new_only'/f'seed_{seed}'/f'stage_{stage}';assert json.loads((control/'routing_and_input_ids.json').read_text())==json.loads((baseline/'routing_and_input_ids.json').read_text()) or [r['training_ids'] for r in json.loads((control/'routing_and_input_ids.json').read_text())]==[r['training_ids'] for r in json.loads((baseline/'routing_and_input_ids.json').read_text())]
    assert json.loads((out/'initial_sources.json').read_text())[str(seed)]['weights_sha256']==json.loads((folder/'initial_sources.json').read_text())[str(seed)]['weights_sha256'];state['trained'].append(name);save()
   if final_allowed and name not in state['final_evaluated']:
    state.update(current=name,operation='locked_final_evaluation');save();call('control_evaluate_study.py',name);call('control_verify_final.py',name);state['final_evaluated'].append(name);save()
  if final_allowed and len(state['final_evaluated'])==len(eligible):state.update(state='complete');save();break
  if main['state']=='failed':state.update(state='blocked_prior_failure');save();break
  time.sleep(15)
 else:state.update(state='deadline_stop');save()
except BaseException as e:state.update(state='failed',error=repr(e));save();raise
