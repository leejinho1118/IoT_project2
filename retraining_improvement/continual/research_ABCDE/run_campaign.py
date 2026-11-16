"""A 학습→검증→분석→보고서→B ... E까지 단일 GPU에서 순차 실행한다."""
import os,sys,argparse,json,hashlib,time,datetime,logging,traceback,subprocess,csv,zipfile,statistics
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('--profile',choices=['smoke','full'],default='full');P.add_argument('--device',choices=['cpu','cuda'],default='cuda');P.add_argument('--run-id',required=True);P.add_argument('--skip-pdf',action='store_true');a=P.parse_args()
if a.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch
from data import load_development,load_final_rows,B,ROOT
from engine import decide_policy,train_round,final_evaluation
from policies import choose_best
from verify_campaign import verify_round,verify_final
HOME=Path(__file__).resolve().parent;OUT=HOME/'output'/a.run_id
if not a.run_id or any(x in a.run_id for x in ['..','/','\\']):raise ValueError('Invalid run-id')
if a.profile=='full' and a.skip_pdf:raise ValueError('Full campaign must create all five reports')
OUT.mkdir(parents=True,exist_ok=True)
import msvcrt
lock=open(OUT/'run.lock','a+b');lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
if (OUT/'status.json').exists():raise RuntimeError('Existing run is preserved; use a new run-id')
GPU=None
if a.device=='cuda':
 GPU=open(ROOT/'continual/drift_v2/gpu.lock','a+b');GPU.write(b'1');GPU.flush();GPU.seek(0);msvcrt.locking(GPU.fileno(),msvcrt.LK_NBLCK,1)
CFG=dict(profile=a.profile,device=a.device,clients=4 if a.profile=='smoke' else 50,seeds=[0] if a.profile=='smoke' else [0,10000,20000,30000,40000],stages=2 if a.profile=='smoke' else 3,local_epochs=2 if a.profile=='smoke' else 100,update_rounds=2 if a.profile=='smoke' else 20,batch_size=12,lr=1e-5,shrink_lambda=10,capacity=24 if a.profile=='smoke' else 128,normal_validation_quantile=.95,selection_score='0.5*(A_auc+B_auc)-0.1*max(0,max(A_fpr,B_fpr)-0.1)',source='cpu_smoke_v2' if a.profile=='smoke' else 'full_v1',author='20222038 이진호')
STATE=dict(state='starting',pid=os.getpid(),started=datetime.datetime.now().astimezone().isoformat(),profile=a.profile,expected_training_records=5*len(CFG['seeds'])*CFG['stages'],expected_final_evaluations=len(CFG['seeds'])*(1+5*CFG['stages']))

def status(state,**kwargs):
 STATE.update(dict(state=state,updated=datetime.datetime.now().astimezone().isoformat(),**kwargs));B.atomic(OUT/'status.json',STATE)


def protected_hashes():
 files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'src/Checkpoint/Results').rglob('*_results.json'))+[ROOT/'continual'/x for x in ['experiment.py','protocol.py','infer.py','original_baseline_summary.json']]+[ROOT/'continual/output/full_v1'/x for x in ['config.json','complete.json','results.json','validation.json']]+list((ROOT/'continual/drift_v2').glob('*.py'))+[ROOT/'continual/drift_v2/output/full_v1'/x for x in ['config.json','complete.json','results.json','validation.json','finalized.json']]
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
 torch.set_num_threads(1)
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 if a.device=='cpu' and torch.cuda.is_available():raise RuntimeError('CPU isolation failed')
 source=ROOT/'continual/drift_v2/output'/CFG['source']
 validation=json.loads((source/'validation.json').read_text(encoding='utf-8'));assert validation['validation']=='passed'
 B.atomic(OUT/'config.json',CFG);protected=protected_hashes();B.atomic(OUT/'protected_artifacts.json',protected)
 runtime={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HOME.glob('*.py')};B.atomic(OUT/'runtime_hashes.json',runtime)
 logging.getLogger().handlers.clear();logging.getLogger().setLevel(logging.INFO);logging.getLogger().addHandler(logging.FileHandler(OUT/'training.log',encoding='utf-8'))
 status('preparing',round=None,seed=None,stage=None,completed_rounds=[])
 clients,final_meta=load_development(source,OUT,CFG)
 B.atomic(OUT/'study_protocol.json',dict(steps=list('ABCDE'),fresh_training_each_round=True,shared_initial_models=str(source),final_open_policy='Only after E development selection is frozen; no final input affects A-E configuration',limitations=['Same dataset and earlier V2 holdouts previously seen; exploratory follow-up, no external replication','Current decision rows have initial-model early-stop provenance; treated as development, not test','CEN reference changes with training rows; weight and reference effects coexist','Original local/shared-model/global-stop semantics retained','All five full runs use 50 clients and five seeds; no automatic budget reduction'],directions={'A':'Uniform replay25 baseline','B':'Uniform replay50 retention','C':'Farthest-first memory at best A/B ratio','D':'Previous-development normal-loss adaptive replay, using best A-C parent','E':'Development gate/rollback on best A-D parent'},selection_score=CFG['selection_score']))
 import ctypes
 ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
 begin=time.perf_counter();analyses=[];policies={}
 try:
  for letter in 'ABCDE':
   policy=decide_policy(letter,analyses,policies);policies[letter]=policy;(OUT/letter).mkdir(exist_ok=True);B.atomic(OUT/letter/'policy.json',policy)
   analysis=train_round(policy,clients,source,OUT,CFG,status)
   status('validating_round',round=letter,seed=None,stage=None)
   verify_round(OUT,letter,CFG)
   analyses.append(analysis);B.atomic(OUT/'analyses.json',analyses)
   next_policy=decide_policy(chr(ord(letter)+1),analyses,policies) if letter!='E' else dict(direction='개발 선택을 고정하고 저장 버전으로 최종 보류 평가를 개방')
   B.atomic(OUT/letter/'next_policy.json',next_policy)
   if letter=='E':
    selection=dict(selected_round=choose_best(analyses),frozen_at=datetime.datetime.now().astimezone().isoformat(),analyses_sha256=hashlib.sha256((OUT/'analyses.json').read_bytes()).hexdigest(),selected_policy=policies[choose_best(analyses)],basis='development only; do not reselect after final results')
    B.atomic(OUT/'selection_frozen.json',selection);selection_sha=hashlib.sha256((OUT/'selection_frozen.json').read_bytes()).hexdigest()
    final_rows=load_final_rows(final_meta,OUT/'selection_frozen.json');final_evaluation(final_rows,OUT,CFG,selection_sha,status)
    status('validating_final',round=letter,seed=None,stage=None);report=verify_final(OUT,CFG);B.atomic(OUT/'final_validation.json',report)
   if not a.skip_pdf:
    status('writing_report',round=letter,seed=None,stage=None)
    runtime_py=Path('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe')
    subprocess.run([str(runtime_py),str(HOME/'build_report.py'),'--out',str(OUT),'--round',letter],check=True)
   B.atomic(OUT/letter/'complete.json',dict(completed=datetime.datetime.now().astimezone().isoformat(),validated=True,analysis='analysis.json',report=None if a.skip_pdf else f'reports/{letter}_Plan_Report.pdf'))
   status('round_complete',round=letter,seed=None,stage=None,completed_rounds=[x['round'] for x in analyses])
  if protected_hashes()!=protected:raise RuntimeError('Protected original/V1/V2 artifact changed')
  if {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HOME.glob('*.py')}!=runtime:raise RuntimeError('Campaign code changed')
  audit=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
  for name,h in audit['source_csv_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
  final_summary=json.loads((OUT/'final_summary.json').read_text(encoding='utf-8'))
  csv_rows=[]
  for x in final_summary['summary']:
   row={'round':x['round']}
   for domain in ['old','new']:
    for metric in ['mean_client_auc','fpr','tpr']:
     row[f'{domain}_{metric}_mean']=x[domain][metric]['mean'];row[f'{domain}_{metric}_std']=x[domain][metric]['std']
   csv_rows.append(row)
  with (OUT/'final_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
  B.atomic(OUT/'complete.json',dict(completed=datetime.datetime.now().astimezone().isoformat(),wall_seconds=time.perf_counter()-begin,rounds=list('ABCDE'),training_records=STATE['expected_training_records'],final_evaluations=STATE['expected_final_evaluations'],selected_round=choose_best(analyses),protected_artifacts=len(protected),source_csv_files=len(audit['source_csv_sha256']),reports_rendered=not a.skip_pdf))
  with zipfile.ZipFile(OUT/'FedMSE_ABCDE_Research_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in list(HOME.glob('*.py'))+list(HOME.glob('*.md'))+list((OUT/'reports').glob('*'))+[OUT/n for n in ['config.json','study_protocol.json','data_audit.json','partition_manifest.json.gz','analyses.json','selection_frozen.json','final_summary.json','final_summary.csv','final_validation.json','complete.json']]+[OUT/letter/n for letter in 'ABCDE' for n in ['policy.json','analysis.json','validation.json','next_policy.json']]:
    if p.is_file():z.write(p,('code/'+p.name) if p.parent==HOME else str(p.relative_to(OUT)))
  status('complete',round=None,seed=None,stage=None,completed_rounds=list('ABCDE'))
 finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
 try:main()
 except BaseException:status('failed',error=traceback.format_exc());raise
