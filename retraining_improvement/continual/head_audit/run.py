"""고정 A 인코더에서 탐지 헤드 원인 분해. 개발로 선택한 뒤 기존 보류 풀 재평가(탐색적)."""
import os,sys,argparse,json,hashlib,time,datetime,pickle,csv,gzip,statistics,traceback,copy
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('--profile',choices=['smoke','full'],default='full');P.add_argument('--device',choices=['cpu','cuda'],default='cuda');P.add_argument('--run-id',required=True);a=P.parse_args()
if a.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,confusion_matrix
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];HOME=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'continual/research_ABCDE'));sys.path.insert(0,str(ROOT/'continual/drift_v2'))
from data import load_development,load_final_rows,B
from head_models import NormalHead,METHODS,DIAGNOSTICS
OUT=HOME/'output'/a.run_id
if any(x in a.run_id for x in ['..','/','\\']):raise ValueError('Invalid run-id')
OUT.mkdir(parents=True,exist_ok=True)
import msvcrt
LOCK=open(OUT/'run.lock','a+b');LOCK.write(b'1');LOCK.flush();LOCK.seek(0);msvcrt.locking(LOCK.fileno(),msvcrt.LK_NBLCK,1)
if (OUT/'status.json').exists():raise RuntimeError('Existing run preserved')
if a.device=='cuda':
 GPU=open(ROOT/'continual/drift_v2/gpu.lock','a+b');GPU.write(b'1');GPU.flush();GPU.seek(0);msvcrt.locking(GPU.fileno(),msvcrt.LK_NBLCK,1)
SOURCE=ROOT/'continual/research_ABCDE/output'/('gpu_pipeline_v1' if a.profile=='smoke' else 'campaign_full_v1')
DATA_SOURCE=ROOT/'continual/drift_v2/output'/('cpu_smoke_v2' if a.profile=='smoke' else 'full_v1')
CFG=json.loads((SOURCE/'config.json').read_text(encoding='utf-8'));CFG.update(device=a.device,profile=a.profile)
STAGE=CFG['stages'];STATE=dict(state='starting',pid=os.getpid(),started=datetime.datetime.now().astimezone().isoformat())

def status(state,**kwargs):
 STATE.update(state=state,updated=datetime.datetime.now().astimezone().isoformat(),**kwargs);B.atomic(OUT/'status.json',STATE)


def view(method,raw,scaled,z):
 return raw if method=='log_input_knn5' else (scaled if method=='input_knn5' else z)


def embedding(model,scaler,rows):
 raw=B.payload(rows);scaled=scaler.transform(raw);z=[];device=next(model.parameters()).device
 with torch.no_grad():
  for i in range(0,len(scaled),256):z.append(model(torch.tensor(scaled[i:i+256],dtype=torch.float32,device=device))[0].cpu().numpy())
 return raw,scaled,np.concatenate(z)


def saturation(scaler,rows):
 raw=B.payload(rows)
 with np.errstate(over='ignore',invalid='ignore'):s=(raw-scaler.mean_)/scaler.scale_
 return dict(rows=len(rows),fraction_values_clipped=float(np.mean(np.abs(s)>20)),fraction_rows_any_clip=float(np.mean(np.any(np.abs(s)>20,axis=1))))


def evaluate_cached(models,features,items,folder,phase,seed):
 results=[];folder.mkdir(parents=True,exist_ok=True)
 for method,clients in models.items():
  all_predictions=[];metric={};begin=time.perf_counter()
  for domain in ['old','new']:
   cms=np.zeros((2,2),dtype=np.int64);aucs=[];client_stats=[]
   for cid,parts in features.items():
    rows=items[cid][domain];y=np.array([r['y'] for r in rows]);raw,scaled,z=parts[domain];head=clients[cid]['head'];score=head.score(view(method,raw,scaled,z));pred=(score>head.threshold).astype(int)
    cm=confusion_matrix(y,pred,labels=[0,1]);cms+=cm;auc=float(roc_auc_score(y,score));aucs.append(auc);tn,fp,fn,tp=cm.ravel()
    client_stats.append(dict(client=cid,auc=auc,fpr=float(fp/(tn+fp)),tpr=float(tp/(tp+fn)),threshold=head.threshold))
    all_predictions.extend(dict(id=r['id'],domain=domain,client=cid,label=int(yy),score=float(ss),prediction=int(pp),threshold=head.threshold) for r,yy,ss,pp in zip(rows,y,score,pred))
   tn,fp,fn,tp=cms.ravel();metric[domain]=dict(mean_client_auc=float(np.mean(aucs)),fpr=float(fp/(tn+fp)),tpr=float(tp/(tp+fn)),confusion_matrix=cms.tolist(),clients=client_stats)
  target=folder/method;target.mkdir(exist_ok=True)
  with gzip.open(target/'predictions.csv.gz','wt',encoding='utf-8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(all_predictions[0]));w.writeheader();w.writerows(all_predictions)
  result=dict(method=method,seed=seed,stage=STAGE,phase=phase,metrics=metric,head_scoring_seconds=time.perf_counter()-begin,diagnostic_only=method in DIAGNOSTICS)
  B.atomic(target/'metrics.json',result);results.append(result)
 return results


def summarize(records):
 result=[]
 for method in METHODS:
  rows=[r for r in records if r['method']==method];x=dict(method=method,diagnostic_only=method in DIAGNOSTICS,seeds=len(rows),metrics={})
  for domain in ['old','new']:
   x['metrics'][domain]={key:dict(mean=statistics.mean(r['metrics'][domain][key] for r in rows),std=statistics.stdev(r['metrics'][domain][key] for r in rows) if len(rows)>1 else None) for key in ['mean_client_auc','fpr','tpr']}
  x['worst_seed_fpr']=max(r['metrics'][d]['fpr'] for r in rows for d in ['old','new'])
  x['minimum_mean_tpr']=min(x['metrics'][d]['tpr']['mean'] for d in ['old','new']);x['maximum_mean_fpr']=max(x['metrics'][d]['fpr']['mean'] for d in ['old','new'])
  x['screening_feasible']=x['maximum_mean_fpr']<=.10 and x['worst_seed_fpr']<=.15 and x['minimum_mean_tpr']>=.80
  x['mean_auc']=statistics.mean(x['metrics'][d]['mean_client_auc']['mean'] for d in ['old','new']);result.append(x)
 return result


def protected():
 files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'continual/drift_v2').glob('*.py'))+list((ROOT/'continual/research_ABCDE').glob('*.py'))+[SOURCE/n for n in ['config.json','analyses.json','final_results.json','selection_frozen.json','complete.json']]
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
 torch.set_num_threads(1);B.configure(CFG,a.device)
 import sklearn,pandas,scipy
 B.atomic(OUT/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pandas.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,cuda=torch.version.cuda))
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 if a.device=='cpu' and torch.cuda.is_available():raise RuntimeError('CPU isolation failed')
 before=protected();runtime={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HOME.glob('*.py')};B.atomic(OUT/'protected_artifacts.json',before);B.atomic(OUT/'runtime_hashes.json',runtime)
 B.atomic(OUT/'protocol.json',dict(frozen_at=datetime.datetime.now().astimezone().isoformat(),methods=METHODS,diagnostics_excluded_from_selection=sorted(DIAGNOSTICS),seed_ids=CFG['seeds'],clients=CFG['clients'],stage=STAGE,encoder='A final-stage weights frozen; no neural weight retraining',reference='Same A bank up to128 old normals and up to128 current verified new normals; balanced equal-size components',calibration='Original threshold for centroid_original; max(q95 old normals,q95 new normals) for all others; not a finite-sample guarantee',selection='Eligible methods require mean old/new FPR<=10%, worst-seed FPR<=15%, min mean TPR>=80%; maximize min TPR then mean AUC. If no feasible candidate, do not claim deployability.',scope='Exploratory on previously seen heldout pools; no new external independent confirmation. Raw input heads are diagnostics, not FedMSE improvement claims.'))
 status('preparing');clients,final_meta=load_development(DATA_SOURCE,OUT,CFG);dev=[];fit_audit=[];clip=[];reproduction=[]
 source_groups={}
 with gzip.open(OUT/'partition_manifest.json.gz','rt',encoding='utf-8') as f:source_groups=json.load(f)
 for seed in CFG['seeds']:
  status('fitting_heads',seed=seed)
  src=SOURCE/'A'/f'seed_{seed}'/f'stage_{STAGE}'/'published';bundle=torch.load(src/'detector_bundle.pt',map_location='cpu',weights_only=False)
  model=B.Shrink_Autoencoder(**bundle['architecture']).to(a.device);model.load_state_dict(bundle['model_state']);model.eval()
  bank_ids=json.loads((SOURCE/'A'/f'seed_{seed}'/'memory_ids.json').read_text(encoding='utf-8'));heads={m:{} for m in METHODS};features={};items={}
  by={r['id']:r for c in clients for key in ['train','calib_old','calib_new','decision_old','decision_new','decision_attack'] for r in c[key]}
  for c in clients:
   cid=c['id'];scaler=bundle['clients'][cid]['input_scaler'];old=[by[i] for i in bank_ids[str(cid)]];new=[r for r in c['candidate_stages'][STAGE-1] if r['y']==0]
   n=min(128,len(old),len(new));old=old[:n];new=new[:n]
   if any(r['y']!=0 for r in old+new):raise ValueError('Attack entered normal reference')
   eold=embedding(model,scaler,old);enew=embedding(model,scaler,new);calold=embedding(model,scaler,c['calib_old']);calnew=embedding(model,scaler,c['calib_new'])
   items[cid]={domain:c[key]+c['decision_attack'] for domain,key in [('old','decision_old'),('new','decision_new')]}
   features[cid]={domain:embedding(model,scaler,rows) for domain,rows in items[cid].items()}
   for domain,key in [('old_normal','decision_old'),('new_normal','decision_new'),('attack','decision_attack')]:clip.append(dict(seed=seed,client=cid,domain=domain,**saturation(scaler,c[key])))
   for method in METHODS:
    begin=time.perf_counter();head=NormalHead(method).fit(view(method,*eold),view(method,*enew),bundle['clients'][cid]['cen'],bundle['clients'][cid]['threshold']).calibrate(view(method,*calold),view(method,*calnew))
    original=method in ['centroid_original','threshold_updated'];ref=bundle['clients'][cid]['reference_ids'] if original else [r['id'] for r in old+new]
    heads[method][cid]=dict(head=head,input_scaler=scaler,reference_ids=ref,calibration_ids=[r['id'] for r in c['calib_old']]+([] if method=='centroid_original' else [r['id'] for r in c['calib_new']]))
    fit_audit.append(dict(seed=seed,client=cid,method=method,reference_ids=ref,calibration_ids=heads[method][cid]['calibration_ids'],head_fit_seconds=time.perf_counter()-begin,serialized_head_bytes=len(pickle.dumps(head)),reference_rows=len(ref),normal_only=True))
  for method in METHODS:
   folder=OUT/'models'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
   torch.save(dict(method=method,stage=STAGE,architecture=bundle['architecture'],model_state=bundle['model_state'],clients=heads[method],source_model_sha256=hashlib.sha256((src/'detector_bundle.pt').read_bytes()).hexdigest(),diagnostic_only=method in DIAGNOSTICS),folder/f'{method}.pt')
  dev+=evaluate_cached(heads,features,items,OUT/'development'/f'seed_{seed}','development',seed);B.atomic(OUT/'development_results.json',dev)
  # 원래 점수와 보정만 한 점수는 순위가 같아야 한다. 기준 A도 재현되어야 한다.
  expected=json.loads((SOURCE/'A'/f'seed_{seed}'/f'stage_{STAGE}'/'record.json').read_text(encoding='utf-8'))['metrics']
  orig=next(r for r in dev if r['seed']==seed and r['method']=='centroid_original');adjusted=next(r for r in dev if r['seed']==seed and r['method']=='threshold_updated')
  with gzip.open(src/'predictions.csv.gz','rt',encoding='utf-8') as f:old_pred={(r['domain'],r['id']):r for r in csv.DictReader(f)}
  with gzip.open(OUT/'development'/f'seed_{seed}/centroid_original/predictions.csv.gz','rt',encoding='utf-8') as f:new_pred={(r['domain'],r['id']):r for r in csv.DictReader(f)}
  assert set(old_pred)==set(new_pred)
  flips=[];maximum=0.
  for key,row in old_pred.items():
   newrow=new_pred[key];err=abs(float(row['score'])-float(newrow['score']));maximum=max(maximum,err)
   assert err<=2e-6*max(1,abs(float(row['score'])))
   if row['prediction']!=newrow['prediction']:
    assert abs(float(row['score'])-float(row['threshold']))<=2e-6*max(1,abs(float(row['threshold'])))
    flips.append(dict(id=row['id'],domain=row['domain'],old_score=float(row['score']),new_score=float(newrow['score']),threshold=float(row['threshold'])))
  for domain in ['old','new']:
   assert abs(orig['metrics'][domain]['mean_client_auc']-expected[domain]['mean_client_auc'])<1e-4
   assert orig['metrics'][domain]['mean_client_auc']==adjusted['metrics'][domain]['mean_client_auc']
  reproduction.append(dict(seed=seed,max_absolute_score_error=maximum,boundary_prediction_flips=len(flips),items=flips,scope='Same frozen weights; tiny arithmetic differences near strict > threshold are recorded, not hidden. All head comparisons use the same freshly computed embeddings.'))
  B.atomic(OUT/'baseline_reproduction.json',reproduction)
 summary=summarize(dev);B.atomic(OUT/'development_summary.json',summary);B.atomic(OUT/'fit_audit.json',fit_audit);B.atomic(OUT/'saturation_audit.json',clip)
 eligible=[r for r in summary if not r['diagnostic_only']];feasible=[r for r in eligible if r['screening_feasible']]
 chosen=sorted(feasible,key=lambda r:(-r['minimum_mean_tpr'],-r['mean_auc'],r['method']))[0] if feasible else sorted(eligible,key=lambda r:(r['maximum_mean_fpr'],-r['minimum_mean_tpr'],r['method']))[0]
 B.atomic(OUT/'selection_frozen.json',dict(selected_method=chosen['method'],screening_feasible=bool(feasible),frozen_at=datetime.datetime.now().astimezone().isoformat(),development_summary_sha256=hashlib.sha256((OUT/'development_summary.json').read_bytes()).hexdigest(),reason='meets prespecified screening constraints' if feasible else 'NO method met all constraints; best relative low-FPR candidate only, not deployable'))
 selection_sha=hashlib.sha256((OUT/'selection_frozen.json').read_bytes()).hexdigest();final_rows=load_final_rows(final_meta,OUT/'selection_frozen.json');final=[]
 for seed in CFG['seeds']:
  status('final_evaluation',seed=seed,selected_method=chosen['method'])
  heads={m:torch.load(OUT/'models'/f'seed_{seed}'/f'{m}.pt',map_location='cpu',weights_only=False)['clients'] for m in METHODS}
  state=torch.load(OUT/'models'/f'seed_{seed}'/'centroid_original.pt',map_location='cpu',weights_only=False);model=B.Shrink_Autoencoder(**state['architecture']).to(a.device);model.load_state_dict(state['model_state']);model.eval();features={};items={}
  for c in clients:
   cid=c['id'];scaler=heads['centroid_original'][cid]['input_scaler'];items[cid]={d:final_rows[f'{cid}/{n}']+final_rows[f'{cid}/{at}'] for d,n,at in [('old','old_normal','old_attack'),('new','new_normal','new_attack')]};features[cid]={d:embedding(model,scaler,rows) for d,rows in items[cid].items()}
  recs=evaluate_cached(heads,features,items,OUT/'final'/f'seed_{seed}','final',seed)
  for r in recs:r['selection_sha256']=selection_sha
  final+=recs;B.atomic(OUT/'final_results.json',final)
 B.atomic(OUT/'final_summary.json',summarize(final))
 if protected()!=before:raise RuntimeError('Protected source/results changed')
 if {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HOME.glob('*.py')}!=runtime:raise RuntimeError('Audit code changed during run')
 for path,h in json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))['source_csv_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
 status('complete',seed=None,selected_method=chosen['method'],screening_feasible=bool(feasible),development_records=len(dev),final_records=len(final))
 B.atomic(OUT/'complete.json',dict(completed=datetime.datetime.now().astimezone().isoformat(),frozen_encoder=True,methods=len(METHODS),development_records=len(dev),final_records=len(final),selected_method=chosen['method'],screening_feasible=bool(feasible),protected_artifacts=len(before)))

if __name__=='__main__':
 try:
  with threadpool_limits(limits=1):main()
 except BaseException:status('failed',error=traceback.format_exc());raise
