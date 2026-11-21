"""Sensitivity to original-sized epoch/round ceilings, after matched main cohorts."""
from pathlib import Path
import sys,subprocess,json,datetime,time,os
H=Path(__file__).resolve().parent;ROOT=H.parents[1];S=H/'budget_status_v1.json'
if S.exists():raise RuntimeError('Preserve existing budget study')
def now():return datetime.datetime.now(datetime.timezone.utc)
def save():S.write_text(json.dumps(state,indent=2),encoding='utf-8')
state=dict(state='waiting_for_replications',pid=os.getpid(),completed=[]);save()
while True:
 r=json.loads((H/'replication_status_v1.json').read_text())
 if r['state'] in ['complete','time_budget_stop']:break
 if r['state']=='failed':state.update(state='blocked_prior_failure');save();raise SystemExit(1)
 time.sleep(15)
F=json.loads((H/'family_selection_frozen.json').read_text());recipe=F['selected'][1];jobs=[]
for seed in [0,10000,20000]:
 for data,tag in [('raw_blocks_v2','nb'),('rt_thing_to_mqtt_v1','rt_tm'),('rt_mqtt_to_thing_v1','rt_mt')]:jobs.append(dict(data=data,run=f"{tag}_{recipe['mode']}_{recipe['loss']}_budget{seed}",seed=seed,mode=recipe['mode'],loss=recipe['loss']))
state.update(state='running',jobs=jobs);save()
for j in jobs:
 if now()>=datetime.datetime(2026,11,20,21,20,tzinfo=datetime.timezone.utc):state.update(state='time_budget_stop');save();break
 cmd=[sys.executable,'-X','utf8','-B',str(H/'run_budget_study.py'),'--data',str(H/j['data']),'--run-id',j['run'],'--mode',j['mode'],'--loss',j['loss'],'--seeds',str(j['seed'])]
 state.update(current=j,started_current=now().isoformat());save()
 with (H/(j['run']+'.log')).open('x',encoding='utf-8') as log:p=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=log)
 state['completed'].append(dict(**j,exit_code=p.returncode,finished=now().isoformat()));save()
 if p.returncode:state.update(state='failed');save();raise SystemExit(p.returncode)
else:state.update(state='complete',finished=now().isoformat());save()
