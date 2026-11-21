"""Coverage-first sequential GPU coordinator; existing training and outputs preserved."""
from pathlib import Path
import json,subprocess,sys,datetime,time,os,hashlib
H=Path(__file__).resolve().parent;ROOT=H.parents[1];S=H/'followup_coordinator_v2.json'
if S.exists():raise RuntimeError('Existing coordinator preserved')
def now():return datetime.datetime.now(datetime.timezone.utc)
def atomic(p,obj):
 tmp=p.with_suffix('.writing');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
 for _ in range(100):
  try:os.replace(tmp,p);return
  except PermissionError:time.sleep(.03)
 raise RuntimeError('Cannot update coordinator state')
state=dict(state='waiting_for_main',pid=os.getpid(),created=now().isoformat(),completed=[],skipped=[],ordering=['normal-only selection','approval query/error','full100/20 budget sensitivity'],prior_waiting_supervisors_replaced=[48724,35928,20928],active_training_interrupted=False)
def save():atomic(S,state)
bridges={k:H/v for k,v in [('normal','normal_gate_status_v1.json'),('approval','approval_status_v1.json'),('budget','budget_status_v1.json')]};interfaces={k:dict(state='waiting_for_main',pid=os.getpid(),coordinator='followup_coordinator_v2.json',completed=[]) for k in bridges}
def bridge(k,**kw):interfaces[k].update(kw);atomic(bridges[k],interfaces[k])
for k in bridges:bridge(k)
save()
while True:
 m=json.loads((H/'replication_status_v1.json').read_text())
 if m['state'] in ['complete','time_budget_stop']:break
 if m['state']=='failed':state.update(state='blocked_prior_failure');save();raise SystemExit(1)
 time.sleep(15)
F=json.loads((H/'family_selection_frozen.json').read_text());r=F['selected'][1]

def source(data,seed):
 for p in (H/'runs').iterdir():
  if not (p/'complete.json').exists():continue
  cfg=json.loads((p/'protocol_frozen.json').read_text())['config']
  if cfg['profile']=='study' and cfg['local_epochs']==20 and not cfg.get('approval_stress') and cfg['mode']==r['mode'] and cfg['shrink_kind']==r['loss'] and Path(cfg['data']).name==data and cfg['seeds']==[seed]:return p
 return None

def launch(kind,j,script,extra=None,estimated_seconds=None):
 if estimated_seconds is not None and now()+datetime.timedelta(seconds=estimated_seconds)>datetime.datetime(2026,11,20,22,2,tzinfo=datetime.timezone.utc):
  state['skipped'].append(dict(**j,kind=kind,reason='Prospective wall-clock estimate would consume report reserve',estimated_seconds=estimated_seconds));save();return False
 if now()>=datetime.datetime(2026,11,20,21,58,tzinfo=datetime.timezone.utc):state['skipped'].append(dict(**j,kind=kind,reason='Launch deadline'));save();return False
 cmd=[sys.executable,'-X','utf8','-B',str(H/script),'--data',str(H/j['data']),'--run-id',j['run'],'--mode',r['mode'],'--loss',r['loss'],'--seeds',str(j['seed'])]+(extra or [])
 state.update(state='running',phase=kind,current=j,started_current=now().isoformat());bridge(kind,state='running',current=j,started_current=now().isoformat());save();start=time.monotonic()
 with (H/(j['run']+'.log')).open('x',encoding='utf-8') as log:p=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=log)
 if p.returncode:raise RuntimeError(f'GPU {j} failed exit {p.returncode}')
 if kind=='normal':
  for tool in ['normal_verify_study.py','normal_verify_bundles.py','normal_verify_guard.py','normal_audit_costs.py','normal_evaluate_study.py','normal_verify_final.py']:
   with (H/(j['run']+'_validation.log')).open('a',encoding='utf-8') as log:p=subprocess.run([sys.executable,'-X','utf8','-B',str(H/tool),j['run']],cwd=H,stdout=log,stderr=log)
   if p.returncode:raise RuntimeError(f'{tool} failed for {j}')
 done=dict(**j,kind=kind,exit_code=0,wall_seconds=time.monotonic()-start,finished=now().isoformat());state['completed'].append(done);interfaces[kind]['completed'].append(done);bridge(kind);save();return True

datasets=[('raw_blocks_v2','nb'),('rt_thing_to_mqtt_v1','rt_tm'),('rt_mqtt_to_thing_v1','rt_mt')]
try:
 for seed in [0,10000,20000]:
  for data,tag in datasets:
   if source(data,seed) is None:state['skipped'].append(dict(kind='normal',data=data,seed=seed,reason='No completed matched main initial model'));save();continue
   j=dict(data=data,seed=seed,run=f"{tag}_normal_gate_{r['mode']}_{r['loss']}_seed{seed}");launch('normal',j,'run_normal_gate.py')
 bridge('normal',state='complete',finished=now().isoformat())
 for seed in [0,10000,20000]:
  for setting in ['query25','poison05']:
   for data,tag in datasets:
    if source(data,seed) is None:state['skipped'].append(dict(kind='approval',data=data,seed=seed,reason='No completed matched main initial model'));save();continue
    j=dict(data=data,seed=seed,setting=setting,run=f"{tag}_{setting}_{r['mode']}_{r['loss']}_stress{seed}");launch('approval',j,'run_approval_study.py',['--setting',setting])
 bridge('approval',state='complete',finished=now().isoformat())
 def estimate(data,seed):
  observed=[x['wall_seconds'] for x in state['completed'] if x['kind']=='budget' and x['data']==data]
  if observed:return max(observed)*1.3
  p=source(data,seed) or source(data,0)
  if p is None:return 999999
  ini=json.loads((p/'initial_sources.json').read_text());init=max(x['cached_initial_training_seconds'] for x in ini.values());rr=json.loads((p/'development_results.json').read_text());update=sum(x['cumulative_training_seconds'] for x in rr if x['stage']==3 and x['method'] in ['new_only','replay','trigger_safe','safe_new']);return max(90,init*(100/30)*(20/8)+update*5*(20/6))*1.15
 # External directions are attempted as a pair before a raw sensitivity estimated to be much longer.
 for seed in [0,10000,20000]:
  pair=[(data,tag,estimate(data,seed)) for data,tag in datasets[1:]]
  enough=now()+datetime.timedelta(seconds=sum(x[2] for x in pair))<=datetime.datetime(2026,11,20,22,2,tzinfo=datetime.timezone.utc)
  for data,tag,seconds in pair:
   j=dict(data=data,seed=seed,run=f"{tag}_{r['mode']}_{r['loss']}_budget{seed}")
   if enough:launch('budget',j,'run_budget_study.py',estimated_seconds=seconds)
   else:state['skipped'].append(dict(**j,kind='budget',reason='Matched external pair does not fit prospective time estimate',estimated_seconds=seconds));save()
  data,tag=datasets[0];j=dict(data=data,seed=seed,run=f"{tag}_{r['mode']}_{r['loss']}_budget{seed}");launch('budget',j,'run_budget_study.py',estimated_seconds=estimate(data,seed))
 bridge('budget',state='complete',finished=now().isoformat(),skipped=[x for x in state['skipped'] if x['kind']=='budget']);state.update(state='complete',finished=now().isoformat());save()
except BaseException as e:
 state.update(state='failed',error=repr(e));save()
 for kind in bridges:
  if interfaces[kind]['state']!='complete':bridge(kind,state='failed',error=repr(e))
 raise
