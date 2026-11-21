"""Development-only study runner. Final NPZ is never opened here."""
from pathlib import Path
import argparse,os,sys,json,copy,time,datetime,hashlib,logging,traceback,inspect
P=argparse.ArgumentParser();P.add_argument('--data',type=Path,required=True);P.add_argument('--run-id',required=True);P.add_argument('--mode',choices=['standard','signed_log'],required=True);P.add_argument('--loss',choices=['norm','squared'],required=True);P.add_argument('--seeds',type=int,nargs='+',default=[0]);P.add_argument('--profile',choices=['smoke','study'],default='study');P.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=P.parse_args()
if a.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch
import sklearn.neighbors
import numpy as np
import normal_gate_core as C
from transforms import InputTransform
from study_data import load_development,activate
from study_metrics import detectors,bundle
from normal_gate_core import evaluate
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'continual/research_ABCDE'))
from policies import mix
HOME=Path(__file__).resolve().parent;ROOT=HOME.parents[1];DATA=a.data.resolve();OUT=HOME/'normal_runs'/a.run_id
if not a.run_id or any(x in a.run_id for x in ['..','/','\\']):raise ValueError('Invalid run ID')
OUT.mkdir(parents=True,exist_ok=True)
import msvcrt
lock=open(OUT/'run.lock','a+b');lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
if (OUT/'status.json').exists():raise RuntimeError('Existing study run preserved')
gpu=None
if a.device=='cuda':
 gpu=open(HOME/'gpu.lock','a+b');gpu.write(b'1');gpu.flush();gpu.seek(0);msvcrt.locking(gpu.fileno(),msvcrt.LK_NBLCK,1)
CFG=dict(profile=a.profile,data=str(DATA.relative_to(ROOT)),mode=a.mode,shrink_kind=a.loss,seeds=a.seeds,device=a.device,stages=3,initial_epochs=2 if a.profile=='smoke' else 30,initial_rounds=2 if a.profile=='smoke' else 8,local_epochs=2 if a.profile=='smoke' else 20,rounds=2 if a.profile=='smoke' else 6,initial_cap=64 if a.profile=='smoke' else 1024,update_cap=64 if a.profile=='smoke' else 256,capacity=64 if a.profile=='smoke' else 128,batch_size=12,lr=1e-5,gradient_clip=5.,scale_floor=.01,normal_validation_quantile=.95,normal_trigger_fpr=.10,author='20222038 이진호')
POLICIES=[dict(name='no_update',frozen=True),dict(name='calibration_only',no_train=True,keep_reference=True),dict(name='reference_only',no_train=True,ratio=.25),dict(name='new_only',ratio=0.),dict(name='replay',ratio=.25),dict(name='safe_replay',ratio=.25,select_round=True),dict(name='trigger_safe',ratio=0.,select_round=True,trigger=True),dict(name='safe_new',ratio=0.,select_round=True)]
POLICIES=[dict(name='normal_gate_new',ratio=0.,select_round=True),dict(name='normal_gate_replay',ratio=.25,select_round=True)]
CFG['selection_ablation']='normal_only'
STATE=dict(state='starting',pid=os.getpid(),started=datetime.datetime.now().astimezone().isoformat(),run_id=a.run_id)
def atomic(path,obj):C.B.atomic(path,obj)
def status(**kw):
 STATE.update(kw);STATE['updated']=datetime.datetime.now().astimezone().isoformat();atomic(OUT/'status.json',STATE)
def cache_initial(clients,architecture,seed):
 activate(clients,0)
 key_data={k:CFG[k] for k in ['data','mode','shrink_kind','initial_epochs','initial_rounds','initial_cap','batch_size','lr','profile']};key_data.update(seed=seed,development_sha256=DEV_HASH,input_dim=CFG['input_dim'],core_sha256=hashlib.sha256((HOME/'federated_core.py').read_bytes()).hexdigest(),transform_sha256=hashlib.sha256((HOME/'transforms.py').read_bytes()).hexdigest(),loader_sha256=hashlib.sha256((HOME/'study_data.py').read_bytes()).hexdigest(),initializer_source_sha256=hashlib.sha256(inspect.getsource(cache_initial).encode('utf-8')).hexdigest(),roles_sha256=hashlib.file_digest((DATA/'roles.json.gz').open('rb'),'sha256').hexdigest())
 key=hashlib.sha256(json.dumps(key_data,sort_keys=True).encode()).hexdigest()[:12];folder=HOME/'initial_models'/f'{DATA.name}_{a.mode}_{a.loss}_seed{seed}_{key}'
 if (folder/'complete.json').exists():
  obj=torch.load(folder/'detector_bundle.pt',map_location='cpu',weights_only=False)
  assert obj['metadata']['cache_key']==key_data
  for c in clients:c['scaler']=obj['clients'][c['id']]['input_scaler']
  return obj,folder
 if folder.exists():raise RuntimeError('Incomplete initial cache preserved: '+str(folder))
 folder.mkdir(parents=True)
 for c in clients:c['scaler']=InputTransform(a.mode).fit(C.payload(c['train']))
 C.B.seeds(seed);model=C.StudySAE(**architecture).to(a.device)
 train={c['id']:c['train'][:CFG['initial_cap']] for c in clients}
 atomic(folder/'initial_input_ids.json',{str(k):[r['id'] for r in v] for k,v in train.items()});atomic(folder/'initial_scaler_fit_ids.json',{str(c['id']):[r['id'] for r in c['train']] for c in clients})
 initial_cfg=dict(CFG,local_epochs=CFG['initial_epochs'],round_checkpoint_dir=str(folder/'training/global_rounds'))
 C.configure(initial_cfg,a.device,status);status(state='initial_training',seed=seed,method='initial',stage=0,round=0)
 model,seconds,history=C.B.fit(model,clients,train,'valid',CFG['initial_rounds'],seed,folder/'training')
 det=detectors(model,clients,{c['id']:c['train'] for c in clients},use_new=False)
 weights=bundle(folder/'detector_bundle.pt',model,clients,det,architecture,dict(cache_key=key_data,initial_training_seconds=seconds,initial_rounds=len(history),initial_global_weights=True))
 obj=torch.load(folder/'detector_bundle.pt',map_location='cpu',weights_only=False)
 atomic(folder/'complete.json',dict(state='complete',weights_sha256=weights,seconds=seconds,completed_at=datetime.datetime.now().astimezone().isoformat()));return obj,folder

def main():
 global DEV_HASH
 torch.set_num_threads(1);logging.getLogger().handlers.clear();logging.getLogger().setLevel(logging.ERROR)
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 clients,roles=load_development(DATA)
 if a.profile=='smoke':clients=clients[:3]
 CFG['clients']=len(clients);CFG['input_dim']=len(clients[0]['train'][0]['x'])
 DEV_HASH=hashlib.file_digest((DATA/'development.npz').open('rb'),'sha256').hexdigest();final_hash=hashlib.file_digest((DATA/'final.npz').open('rb'),'sha256').hexdigest()
 C.configure(CFG,a.device,status)
 protocol=dict(config=CFG,methods=POLICIES,development_npz_sha256=DEV_HASH,final_npz_sha256=final_hash,final_features_opened=False,stage_release='Future stage arrays are not decompressed before activate(stage). Initial learning sees initial old normal only.',normal_verifier='Ground-truth-assisted verified-normal environment; actual attack rows excluded from all clean weight and reference fitting',primary='normal_gate_new',secondary='trigger_safe',threshold='Frozen no-update retains initial old q95. Calibration-only and all updates use max(old q95,current-stage new q95). Same-threshold controls separate calibration from weight adaptation.',selection_score='-0.5*(log(old normal MSE + 1e-12)+log(new normal MSE + 1e-12)); no attack-label metric enters selection',guard='Old normal reconstruction MSE <= initial*1.05; old/new normal FPR <= recalibrated_previous+.02; normal score improves >1e-4. Attack AUC/TPR logged for monitoring only.',trigger='If verified candidate normal FPR under the previous deployed detector <=0.10, skip weight/ref training but update normal calibration',upstream='Same SAE topology hidden50/latent11 and MSEAvg; schema-sized input/output; shared sequential semantics; original two-if global stop; norm versus explicit squared loss is a controlled recipe factor',privacy='Dataset experiments; initial old/server development/validation materials are shared assumptions. No DP or privacy guarantee',coverage='N-BaIoT has 9 actual devices and 7 Mirai-covered clients. RT uses 5 virtual clients, not claimed physical hosts. File order is not proven timestamps.')
 atomic(OUT/'protocol_frozen.json',protocol);atomic(OUT/'client_ids.json',[c['id'] for c in clients]);atomic(OUT/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,device=a.device,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None))
 snapshot=OUT/'source_snapshot';snapshot.mkdir();hashes={}
 for p in HOME.glob('*.py'):hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();(snapshot/p.name).write_bytes(p.read_bytes())
 atomic(OUT/'runtime_hashes.json',hashes);atomic(OUT/'upstream_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').rglob('*.py')})
 architecture=dict(input_dim=CFG['input_dim'],output_dim=CFG['input_dim'],hidden_neus=50,latent_dim=11,shrink_lambda=10,shrink_kind=a.loss)
 records=[];initial_sources={};start_time=time.perf_counter()
 for seed in a.seeds:
  C.configure(CFG,a.device,status);initial,source=cache_initial(clients,architecture,seed);initial_sources[str(seed)]=dict(path=str(source.relative_to(ROOT)),weights_sha256=initial['weights_sha256'],cached_initial_training_seconds=initial['metadata']['initial_training_seconds'])
  atomic(OUT/'initial_sources.json',initial_sources)
  for policy in POLICIES:
   name=policy['name'];activate(clients,0)
   for c in clients:c['scaler']=initial['clients'][c['id']]['input_scaler']
   C.B.seeds(seed);model=C.StudySAE(**architecture).to(a.device);model.load_state_dict(initial['model_state']);model.eval();teacher=copy.deepcopy(model).eval();teacher.requires_grad_(False)
   reference={c['id']:c['train'] for c in clients};bank={c['id']:c['train'][:CFG['capacity']] for c in clients};published=detectors(model,clients,reference,use_new=False)
   initial_metrics=evaluate(model,clients,published,initial_old_only=True)
   folder=OUT/name/f'seed_{seed}';folder.mkdir(parents=True);atomic(folder/'start.json',dict(initial_weights_sha256=C.whash(model),initial_source=str(source.relative_to(ROOT)),initial_old_only=initial_metrics))
   cumulative=0.;steps=0
   for stage in [1,2,3]:
    activate(clients,stage);sf=folder/f'stage_{stage}';sf.mkdir();status(state='preparing_stage',seed=seed,method=name,stage=stage,round=0)
    train={};routing=[];normal_fp=0;normal_count=0
    for c in clients:
     cid=c['id'];items=c['candidate_stages'][stage-1]
     scores=published[cid]['cen'].get_density(C.B.encode(model,c,items));normal=[r for r in items if r['y']==0]
     normal_fp+=sum(int(r['y']==0 and score>published[cid]['threshold']) for r,score in zip(items,scores));normal_count+=len(normal)
     if len(normal)>CFG['update_cap']:
      ii=np.random.default_rng(4422026+stage*100+cid).choice(len(normal),CFG['update_cap'],replace=False);normal=[normal[int(i)] for i in ii]
     train[cid],k=mix(normal,bank[cid],policy.get('ratio',0.),seed+(stage-1)*1000+cid)
     routing.append(dict(client=cid,normal_ids=[r['id'] for r in normal],attack_ids_excluded=[r['id'] for r in items if r['y']!=0],training_ids=[r['id'] for r in train[cid]],replay_rows=k,detections=[dict(id=r['id'],anomaly_flag=bool(score>published[cid]['threshold']),score=float(score)) for r,score in zip(items,scores)]))
    candidate_fpr=normal_fp/normal_count;checkpoint=dict(enabled=False,accepted=True);trained=False;seconds=0.;history=[]
    atomic(sf/'routing_and_input_ids.json',routing)
    if policy.get('frozen',False):pass
    elif policy.get('keep_reference',False):published=detectors(model,clients,reference,use_new=True)
    elif policy.get('no_train',False):reference=train;published=detectors(model,clients,reference,use_new=True)
    elif policy.get('trigger',False) and candidate_fpr<=CFG['normal_trigger_fpr']:
     checkpoint=dict(enabled=True,accepted=False,trigger_skipped=True,normal_candidate_fpr=candidate_fpr);published=detectors(model,clients,reference,use_new=True)
    else:
     previous_ref=reference;previous=evaluate(model,clients,detectors(model,clients,reference,use_new=True))
     update_cfg=dict(CFG,round_checkpoint_dir=str(sf/'global_rounds'));C.configure(update_cfg,a.device,status)
     def callback(current,rnd):return evaluate(current,clients,detectors(current,clients,train,use_new=True))
     model,seconds,history,checkpoint=C.fit_stage(model,teacher,clients,train,bank,policy,seed,stage,sf,callback,initial_metrics,previous)
     cumulative+=seconds;steps+=sum(x['optimizer_steps'] for h in history for x in h['local']);trained=True
     reference=train if checkpoint['accepted'] else previous_ref;published=detectors(model,clients,reference,use_new=True)
    metrics=evaluate(model,clients,published,save=sf/'development_predictions.csv.gz')
    weights=bundle(sf/'detector_bundle.pt',model,clients,published,architecture,dict(config=CFG,method=name,seed=seed,stage=stage,initial_source=str(source.relative_to(ROOT)),checkpoint=checkpoint))
    record=dict(method=name,seed=seed,stage=stage,weights_sha256=weights,metrics=metrics,checkpoint=checkpoint,weight_training_performed=trained,normal_candidate_fpr=candidate_fpr,cumulative_training_seconds=cumulative,cumulative_optimizer_steps=steps,bundle=str((sf/'detector_bundle.pt').relative_to(OUT)),development_score=C.selection_score(metrics,initial_metrics))
    atomic(sf/'record.json',record);records.append(record);atomic(OUT/'development_results.json',records)
   assert C.whash(teacher)==initial['weights_sha256'];del model,teacher
 status(state='complete',records=len(records),wall_seconds=time.perf_counter()-start_time,final_features_opened=False)
 atomic(OUT/'complete.json',dict(state='complete',records=len(records),final_features_opened=False,completed_at=datetime.datetime.now().astimezone().isoformat()))
try:main()
except BaseException as exc:
 status(state='failed',error=repr(exc));traceback.print_exc();raise
