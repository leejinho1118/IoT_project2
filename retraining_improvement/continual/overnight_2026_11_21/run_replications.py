"""Freeze family selection from development only, then sequential matched replications."""
from pathlib import Path
import json,subprocess,sys,datetime,time,hashlib
H=Path(__file__).resolve().parent;ROOT=H.parents[1];S=H/'replication_status_v1.json'
if S.exists():raise RuntimeError('Existing replication preserved')
def now():return datetime.datetime.now(datetime.timezone.utc)
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
state=dict(state='waiting_for_grid',pid=__import__('os').getpid(),completed=[],created=now().isoformat());write(S,state)
while json.loads((H/'grid_status_v1.json').read_text())['state']=='running':time.sleep(10)
grid=json.loads((H/'grid_status_v1.json').read_text());assert grid['state']=='complete'
recipes=[]
for mode,loss in [('standard','norm'),('standard','squared'),('signed_log','norm'),('signed_log','squared')]:
 run=f'nb_{mode}_{loss}_dev0';rows=[r for r in json.loads((H/'runs'/run/'development_results.json').read_text()) if r['method']=='safe_new'];assert len(rows)==3
 recipes.append(dict(mode=mode,loss=loss,run=run,score=sum(r['development_score'] for r in rows)/3,all_stage_fpr_under_10pct=all(max(r['metrics'][d]['fpr'] for d in ['old','new'])<=.10 for r in rows)))
ranked=sorted(recipes,key=lambda r:(r['all_stage_fpr_under_10pct'],r['score']),reverse=True);best=ranked[0]
if best['mode']=='standard' and best['loss']=='norm':best=next(r for r in ranked if r['mode']=='signed_log')
selected=[recipes[0],best]
freeze=dict(state='frozen',created=now().isoformat(),selection_source='N-BaIoT development only; final feature arrays not opened',primary_method='safe_new',recipes=recipes,selected=selected,planned_seeds=[0,10000,20000,30000,40000],primary_stage=3,main_budget=dict(initial_epochs=30,initial_rounds=8,local_epochs=20,rounds=6),deadline_policy='Time-limited complete cohorts, never performance-dependent omission',external_dataset='RT-IoT2022 two reciprocal scenarios; no external final tuning',protocol_sha256=hashlib.sha256((H/'overnight_protocol_frozen.json').read_bytes()).hexdigest())
F=H/'family_selection_frozen.json';assert not F.exists();write(F,freeze)
jobs=[]
# Complete seed cohorts across source data and both external directions before adding more seeds.
for seed in [0,10000,20000,30000,40000]:
 for dataset,tag in [('raw_blocks_v2','nb'),('rt_thing_to_mqtt_v1','rt_tm'),('rt_mqtt_to_thing_v1','rt_mt')]:
  for r in selected:
   if dataset=='raw_blocks_v2' and seed==0:continue
   jobs.append(dict(data=dataset,run=f"{tag}_{r['mode']}_{r['loss']}_rep{seed}",mode=r['mode'],loss=r['loss'],seed=seed))
state.update(state='running',jobs=jobs,selection_sha256=hashlib.sha256(F.read_bytes()).hexdigest());write(S,state)
for j in jobs:
 if now()>=datetime.datetime(2026,11,20,21,35,tzinfo=datetime.timezone.utc):state.update(state='time_budget_stop');write(S,state);break
 cmd=[sys.executable,'-X','utf8','-B',str(H/'run_study.py'),'--data',str(H/j['data']),'--run-id',j['run'],'--mode',j['mode'],'--loss',j['loss'],'--seeds',str(j['seed'])]
 state.update(current=j,started_current=now().isoformat());write(S,state)
 with (H/(j['run']+'.log')).open('x',encoding='utf-8') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=log)
 state['completed'].append(dict(**j,exit_code=result.returncode,finished=now().isoformat()));write(S,state)
 if result.returncode:state.update(state='failed');write(S,state);raise SystemExit(result.returncode)
else:state.update(state='complete',finished=now().isoformat());write(S,state)
