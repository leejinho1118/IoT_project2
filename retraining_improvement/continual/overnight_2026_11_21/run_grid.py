"""Sequential GPU development grid. No final evaluation and no concurrent GPU fits."""
from pathlib import Path
import subprocess,sys,json,datetime,hashlib
H=Path(__file__).resolve().parent;ROOT=H.parents[1];statefile=H/'grid_status_v1.json'
if statefile.exists():raise SystemExit('Existing grid status preserved; do not duplicate')
jobs=[dict(run='smoke_rt_v1',data='rt_thing_to_mqtt_v1',mode='standard',loss='norm',profile='smoke')]
for mode,loss in [('standard','norm'),('standard','squared'),('signed_log','norm'),('signed_log','squared')]:jobs.append(dict(run=f'nb_{mode}_{loss}_dev0',data='raw_blocks_v2',mode=mode,loss=loss,profile='study'))
state=dict(state='running',created=datetime.datetime.now().astimezone().isoformat(),jobs=jobs,completed=[],runner_sha256=hashlib.sha256((H/'run_study.py').read_bytes()).hexdigest(),final_evaluation_permitted=False)
def save():statefile.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
for job in jobs:
 if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,11,20,22,10,tzinfo=datetime.timezone.utc):state.update(state='deadline_stop');save();raise SystemExit(0)
 cmd=[sys.executable,'-X','utf8','-B',str(H/'run_study.py'),'--data',str(H/job['data']),'--run-id',job['run'],'--mode',job['mode'],'--loss',job['loss'],'--profile',job['profile'],'--seeds','0']
 state.update(current=job,started_current=datetime.datetime.now().astimezone().isoformat());save()
 with (H/(job['run']+'.log')).open('w',encoding='utf-8') as f:result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=f)
 state['completed'].append(dict(**job,exit_code=result.returncode,finished=datetime.datetime.now().astimezone().isoformat()));save()
 if result.returncode:state.update(state='failed');save();raise SystemExit(result.returncode)
state.update(state='complete',finished=datetime.datetime.now().astimezone().isoformat());save();print(json.dumps(state,indent=2))
