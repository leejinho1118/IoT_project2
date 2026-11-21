"""Normal-only selection ablation, isolated from main worker and sequential GPU queues."""
from pathlib import Path
import sys,subprocess,json,datetime,time,os,hashlib
H=Path(__file__).resolve().parent;ROOT=H.parents[1];S=H/'normal_gate_status_v1.json'
if S.exists():raise RuntimeError('Preserve existing normal-only queue')
def now():return datetime.datetime.now(datetime.timezone.utc)
def save():S.write_text(json.dumps(state,indent=2),encoding='utf-8')
state=dict(state='waiting_for_approval_queue',pid=os.getpid(),completed=[]);save()
while True:
 r=json.loads((H/'approval_status_v1.json').read_text())
 if r['state'] in ['complete','time_budget_stop']:break
 if r['state'] in ['failed','blocked_prior_failure']:state.update(state='blocked_prior_failure');save();raise SystemExit(1)
 time.sleep(15)
recipe=json.loads((H/'family_selection_frozen.json').read_text())['selected'][1];jobs=[]
for seed in [0,10000,20000]:
 for data,tag in [('raw_blocks_v2','nb'),('rt_thing_to_mqtt_v1','rt_tm'),('rt_mqtt_to_thing_v1','rt_mt')]:jobs.append(dict(data=data,run=f"{tag}_normal_gate_{recipe['mode']}_{recipe['loss']}_seed{seed}",seed=seed,mode=recipe['mode'],loss=recipe['loss']))
state.update(state='running',jobs=jobs);save()
for j in jobs:
 if now()>=datetime.datetime(2026,11,20,21,58,tzinfo=datetime.timezone.utc):state.update(state='time_budget_stop');save();break
 cmd=[sys.executable,'-X','utf8','-B',str(H/'run_normal_gate.py'),'--data',str(H/j['data']),'--run-id',j['run'],'--mode',j['mode'],'--loss',j['loss'],'--seeds',str(j['seed'])]
 state.update(current=j,started_current=now().isoformat());save()
 with (H/(j['run']+'.log')).open('x',encoding='utf-8') as log:p=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=log)
 if p.returncode:state.update(state='failed',exit_code=p.returncode);save();raise SystemExit(p.returncode)
 for script in ['normal_verify_study.py','normal_verify_bundles.py','normal_verify_guard.py','normal_audit_costs.py','normal_evaluate_study.py','normal_verify_final.py']:
  with (H/(j['run']+'_validation.log')).open('a',encoding='utf-8') as log:p=subprocess.run([sys.executable,'-X','utf8','-B',str(H/script),j['run']],cwd=H,stdout=log,stderr=log)
  if p.returncode:state.update(state='failed',validation_tool=script,exit_code=p.returncode);save();raise SystemExit(p.returncode)
 state['completed'].append(dict(**j,exit_code=0,finished=now().isoformat()));save()
else:state.update(state='complete',finished=now().isoformat());save()
