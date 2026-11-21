"""CPU checks of completed runs; final evaluation waits for main replication completion."""
from pathlib import Path
import subprocess,sys,json,time,datetime,os
H=Path(__file__).resolve().parent;S=H/'validation_worker_status_v1.json'
if S.exists():raise RuntimeError('Existing validation worker preserved')
state=dict(state='running',pid=os.getpid(),validated=[],final_evaluated=[],failures=[])
def save():S.write_text(json.dumps(state,indent=2),encoding='utf-8')
def call(script,name):
 with (H/f'{name}_{script}.log').open('a',encoding='utf-8') as f:p=subprocess.run([sys.executable,'-X','utf8','-B',str(H/script),name],cwd=H,stdout=f,stderr=f)
 if p.returncode:raise RuntimeError(f'{script} failed for {name}; exit {p.returncode}')
save()
try:
 while datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,11,20,22,20,tzinfo=datetime.timezone.utc):
  selection=H/'family_selection_frozen.json'
  if selection.exists():F=json.loads(selection.read_text());selected={(r['mode'],r['loss']) for r in F['selected']}
  else:time.sleep(10);continue
  main=json.loads((H/'replication_status_v1.json').read_text());final_allowed=main['state'] in ['complete','time_budget_stop']
  for folder in sorted((H/'runs').iterdir()):
   if not (folder/'complete.json').exists():continue
   pr=json.loads((folder/'protocol_frozen.json').read_text());cfg=pr['config']
   if cfg['profile']!='study' or (cfg['mode'],cfg['shrink_kind']) not in selected:continue
   n=folder.name;stress=bool(cfg.get('approval_stress'))
   if n not in state['validated']:
    state.update(current=n,operation='development_validation');save();call('verify_approval.py' if stress else 'verify_study.py',n);call('verify_approval_bundles.py' if stress else 'verify_bundles.py',n);call('verify_guard.py',n);call('audit_costs.py',n);state['validated'].append(n);save()
   if final_allowed and n not in state['final_evaluated']:
    state.update(current=n,operation='locked_final_evaluation');save();call('evaluate_study.py',n);call('verify_final.py',n);state['final_evaluated'].append(n);save()
  states=[json.loads((H/p).read_text())['state'] for p in ['replication_status_v1.json','budget_status_v1.json','approval_status_v1.json']]
  if all(x in ['complete','time_budget_stop'] for x in states):state.update(state='complete');save();break
  time.sleep(15)
 else:state.update(state='deadline_stop');save()
except BaseException as e:state['failures'].append(repr(e));state.update(state='failed');save();raise
