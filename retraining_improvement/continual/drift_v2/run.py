"""FedMSE V2: 개발 전용 분포 변화 및 보관량·검증 보류 비교 실행기."""
import os,sys,argparse,json,time,copy,hashlib,datetime,traceback,logging,csv,gzip,math
from pathlib import Path
P=argparse.ArgumentParser()
P.add_argument('--profile',choices=['smoke','pilot','full'],default='smoke')
P.add_argument('--device',choices=['cpu','cuda'],default='cpu')
P.add_argument('--run-id',required=True)
P.add_argument('--prepare-only',action='store_true')
a=P.parse_args()
if a.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import pandas as pd
import torch
import backend as B
from scenario import load_plan,verified_rows,replay_mix
ROOT=Path(__file__).resolve().parents[2];HOME=ROOT/'continual/drift_v2';OUT=HOME/'output'/a.run_id
if not a.run_id or any(c in a.run_id for c in ['..','/','\\']):raise ValueError('Invalid run-id')
# 실행별 설정은 평가 결과를 보기 전에 고정한다. pilot은 기능 검증이며 최종 성능이 아니다.
CONFIGS={
 'smoke':dict(clients=4,source_cap=300,local_epochs=2,initial_rounds=2,update_rounds=2,stages=2,seeds=[0],capacities=[8,24]),
 'pilot':dict(clients=10,source_cap=None,local_epochs=10,initial_rounds=5,update_rounds=3,stages=3,seeds=[0,10000],capacities=[32,128]),
 'full':dict(clients=50,source_cap=None,local_epochs=100,initial_rounds=20,update_rounds=20,stages=3,seeds=[0,10000,20000,30000,40000],capacities=[32,128])}
CFG=dict(CONFIGS[a.profile],profile=a.profile,device=a.device,batch_size=12,lr=1e-5,shrink_lambda=10,data_seed=5678,max_capacity=max(CONFIGS[a.profile]['capacities']),replay_fraction=.25,normal_validation_quantile=.95,min_verified_normal=6,uncertainty_rates=[0,.2],normal_verifier='정상 여부를 확인할 수 있는 검증자 보조 환경',detector_reference='current arm training rows; includes CEN-refit effect',threshold_policy='recalibrate on fixed old A validation; additionally report initial-threshold FPR/TPR',input_preprocessing='initial A StandardScaler then fixed clip [-20,20]; robustness change from V1 applied to every arm',scenario_protocol='20% scenario fit / 20% holdout / 60% model; dev-only signed-log PC1 median; A initially, B arrivals')
OUT.mkdir(parents=True,exist_ok=True)
import msvcrt
LOCK=open(OUT/'run.lock','a+b');LOCK.write(b'1');LOCK.flush();LOCK.seek(0);msvcrt.locking(LOCK.fileno(),msvcrt.LK_NBLCK,1)
if (OUT/'status.json').exists():raise RuntimeError('Existing run preserved. Use a new run-id.')
GPU_LOCK=None
if a.device=='cuda' and not a.prepare_only:
 GPU_LOCK=open(HOME/'gpu.lock','a+b');GPU_LOCK.write(b'1');GPU_LOCK.flush();GPU_LOCK.seek(0);msvcrt.locking(GPU_LOCK.fileno(),msvcrt.LK_NBLCK,1)
STATE=dict(state='initializing',pid=os.getpid(),started=datetime.datetime.now().astimezone().isoformat(),config=CFG)

def status(state,**kwargs):
 STATE.update(dict(state=state,updated=datetime.datetime.now().astimezone().isoformat(),**kwargs));B.atomic(OUT/'status.json',STATE)


def hashes():
 files=list((ROOT/'src').rglob('*.py'))+[ROOT/'continual'/x for x in ['experiment.py','protocol.py','infer.py']]+list(HOME.glob('*.py'))
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def state_hash(model):
 return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def normal_reconstruction(model,clients):
 """보류 정상의 복원 오차를 기록해 임계값 재보정과 별도로 망각을 관찰한다."""
 result={};device=next(model.parameters()).device;model.eval()
 for domain,key in [('old','old_normal'),('new','new_normal')]:
  summed=0.;count=0
  for c in clients:
   x=c['scaler'].transform(B.payload(c[key]))
   with torch.no_grad():
    for i in range(0,len(x),256):
     batch=torch.tensor(x[i:i+256],dtype=torch.float32,device=device);_,recon,_=model(batch)
     summed+=float(torch.sum((batch-recon)**2).item());count+=batch.numel()
  result[domain]=summed/count
 return result


def frozen_threshold_metrics(folder,thresholds):
 """재보정된 주 지표와 함께 초기 임계값 고정 시의 FPR/TPR을 저장한다."""
 result={}
 with gzip.open(folder/'predictions.csv.gz','rt',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 for domain in ['old','new']:
  q=[x for x in rows if x['domain']==domain];fp=tp=norm=att=0
  for x in q:
   pred=float(x['score'])>thresholds[int(x['client'])]
   if x['label']=='0':norm+=1;fp+=int(pred)
   else:att+=1;tp+=int(pred)
  result[domain]=dict(fpr=fp/norm,tpr=tp/att)
 return result


def arm_specs():
 yield 'no_update',0.,0
 for rate in CFG['uncertainty_rates']:
  for cap in [0]+CFG['capacities']:
   yield f'{"new_only" if cap==0 else "replay"+str(cap)}_u{int(rate*100):02d}',rate,cap


def main():
 torch.set_num_threads(1);B.configure(CFG,a.device)
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 if a.device=='cpu' and torch.cuda.is_available():raise RuntimeError('CPU isolation failed')
 old=json.loads((ROOT/'original_output/status.json').read_text(encoding='utf-8-sig'))
 if old.get('state')!='complete' or old.get('exit_code')!=0:raise RuntimeError('Original must be complete')
 logging.getLogger().handlers.clear();logging.getLogger().setLevel(logging.INFO);logging.getLogger().addHandler(logging.FileHandler(OUT/'training.log',encoding='utf-8'))
 before=hashes();B.atomic(OUT/'config.json',CFG)
 status('preparing',source_hashes=before,seed=None,arm=None,stage=None,environment=dict(python=sys.version,torch=torch.__version__,pandas=pd.__version__,numpy=np.__version__,cuda=torch.version.cuda))
 clients=load_plan(CFG,OUT)
 if a.prepare_only:status('prepared');return
 import ctypes
 ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
 results=[];wall=time.perf_counter()
 try:
  for seed in CFG['seeds']:
   B.seeds(seed);status('initial_training',seed=seed,arm=None,stage=None)
   model=B.Shrink_Autoencoder(input_dim=115,output_dim=115,hidden_neus=50,latent_dim=11,shrink_lambda=10)
   initial,initial_seconds,_=B.fit(model,clients,{c['id']:c['train'] for c in clients},'valid',CFG['initial_rounds'],seed,OUT/f'seed_{seed}/initial_training')
   weights=copy.deepcopy(initial.state_dict());device=next(initial.parameters()).device;initial_hash=state_hash(initial);ref={c['id']:c['train'] for c in clients}
   baseline,detectors=B.evaluate(initial,clients,ref,OUT/f'seed_{seed}/initial_evaluation',0)
   initial_mse=normal_reconstruction(initial,clients);thresholds={k:v[1] for k,v in detectors.items()}
   B.atomic(OUT/f'seed_{seed}/initial_evaluation/reconstruction.json',initial_mse)
   # 각 보류율의 검증 결과는 모든 보관량 비교에서 동일하다.
   verified={}
   for rate in CFG['uncertainty_rates']:
    verified[rate]=[]
    for stage in range(CFG['stages']):
     accepted={};routes=[]
     for c in clients:
      normal,attack,uncertain=verified_rows(c['candidate_stages'][stage],rate);accepted[c['id']]=normal
      for kind,rows in [('normal',normal),('attack',attack),('uncertain',uncertain)]:routes.extend(dict(id=r['id'],client=c['id'],verified=kind) for r in rows)
     folder=OUT/f'seed_{seed}/verified_u{int(rate*100):02d}_stage_{stage+1}';folder.mkdir(parents=True,exist_ok=True)
     B.atomic(folder/'routing.json',dict(environment=CFG['normal_verifier'],items=routes,uncertainty_rate=rate))
     verified[rate].append(accepted)
   for arm,rate,capacity in arm_specs():
    model=B.Shrink_Autoencoder(input_dim=115,output_dim=115,hidden_neus=50,latent_dim=11,shrink_lambda=10).to(device);model.load_state_dict(weights);reference=ref;cumulative=0.;steps=0
    for stage in range(CFG['stages']):
     status('updating',seed=seed,arm=arm,stage=stage+1,completed_records=len(results))
     folder=OUT/f'seed_{seed}/{arm}/stage_{stage+1}';folder.mkdir(parents=True,exist_ok=True)
     B.log_current_detection(model,clients,reference,stage,folder)
     new=verified[rate][stage];train={};replay=0;seconds=0.;skipped=False;history=[]
     if arm!='no_update':
      if min(len(x) for x in new.values())<CFG['min_verified_normal']:skipped=True
      else:
       for c in clients:
        train[c['id']],count=replay_mix(new[c['id']],c['replay'],capacity,seed+stage*1000+c['id']);replay+=count
       model,seconds,history=B.fit(model,clients,train,'new_valid',CFG['update_rounds'],seed+100000+stage,folder/'training');reference=train;cumulative+=seconds
     B.atomic(folder/'training_input_ids.json',{str(cid):[r['id'] for r in rows] for cid,rows in train.items()})
     result,_=B.evaluate(model,clients,reference,folder,stage+1);mse=normal_reconstruction(model,clients);step_count=sum(x['optimizer_steps'] for rr in history for x in rr['local']);steps+=step_count
     record=dict(seed=seed,arm=arm,stage=stage+1,capacity=capacity,uncertainty_rate=rate,metrics=result,reconstruction_mse=mse,old_reconstruction_change=mse['old']-initial_mse['old'],old_auc_change=result['old']['mean_client_auc']-baseline['old']['mean_client_auc'],old_fpr_change=result['old']['fpr']-baseline['old']['fpr'],frozen_initial_threshold=frozen_threshold_metrics(folder,thresholds),initial_training_seconds=initial_seconds,additional_training_seconds=seconds,cumulative_additional_seconds=cumulative,optimizer_steps=step_count,cumulative_optimizer_steps=steps,replay_samples=replay,replay_feature_bytes=sum(min(len(c['replay']),capacity)*115*8 for c in clients),state_sha256=state_hash(model),update_skipped=skipped,minimum_verified_normal=min(len(x) for x in new.values()),model_device=str(device))
     if arm=='no_update' and (record['state_sha256']!=initial_hash or result!=baseline or mse!=initial_mse):raise RuntimeError('No-update invariance failed')
     B.atomic(folder/'evaluation.json',record);results.append(record);B.atomic(OUT/'results.json',results)
   B.atomic(OUT/f'seed_{seed}/complete.json',dict(seed=seed,records=sum(x['seed']==seed for x in results)))
  if hashes()!=before:raise RuntimeError('Source changed during run')
  B.atomic(OUT/'complete.json',dict(completed=datetime.datetime.now().astimezone().isoformat(),records=len(results),wall_seconds=time.perf_counter()-wall,source_unchanged=True))
  status('complete',seed=None,arm=None,stage=None,completed_records=len(results))
 finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
 try:main()
 except BaseException:status('failed',error=traceback.format_exc());raise
