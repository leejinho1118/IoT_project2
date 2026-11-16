"""선택된 잔차 기억의 검증자 오판 민감도. 공격을 정상으로 잘못 넣는 격리된 스트레스 시험."""
import sys,argparse,json,time,gzip,hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,confusion_matrix
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'continual/research_ABCDE'));sys.path.insert(0,str(ROOT/'continual/drift_v2'))
from data import load_development,B
from residual_models import NormalHead
from head_models import NormalHead as GuardHead
P=argparse.ArgumentParser();P.add_argument('--run-id',required=True);a=P.parse_args();HOME=Path(__file__).resolve().parent;PRIMARY=HOME/'output/residual_full_v1';SOURCE=ROOT/'continual/research_ABCDE/output/campaign_full_v1';OUT=HOME/'output'/a.run_id
if OUT.exists():raise RuntimeError('Existing stress result preserved')
OUT.mkdir(parents=True);torch.set_num_threads(1);cfg=json.loads((SOURCE/'config.json').read_text(encoding='utf-8'));method=json.loads((PRIMARY/'selection_frozen.json').read_text(encoding='utf-8'))['selected_method'];B.configure(cfg,'cuda')
clients,_=load_development(ROOT/'continual/drift_v2/output/full_v1',OUT,cfg);results=[];audit=[]
def embed(model,scaler,rows):
 x=scaler.transform(B.payload(rows));z=[]
 with torch.no_grad():
  for start in range(0,len(x),256):z.append(model(torch.tensor(x[start:start+256],dtype=torch.float32,device='cuda'))[0].cpu().numpy())
 return np.concatenate([x,np.concatenate(z)],axis=1)
with threadpool_limits(limits=1):
 for seed in cfg['seeds']:
  B.atomic(OUT/'status.json',dict(state='running',seed=seed));source=SOURCE/f'A/seed_{seed}/stage_3/published/detector_bundle.pt';b=torch.load(source,map_location='cpu',weights_only=False);model=B.Shrink_Autoencoder(**b['architecture']).cuda();model.load_state_dict(b['model_state']);model.eval();bank=json.loads((SOURCE/f'A/seed_{seed}/memory_ids.json').read_text(encoding='utf-8'));data={}
  for c in clients:
   cid=c['id'];scaler=b['clients'][cid]['input_scaler'];by={r['id']:r for r in c['train']};old=[by[i] for i in bank[str(cid)]];new=[r for r in c['candidate_stages'][2] if r['y']==0];bad=[r for r in c['candidate_stages'][2] if r['y']==1];n=min(128,len(old),len(new));old=old[:n];new=new[:n]
   data[cid]=dict(old=embed(model,scaler,old),new=embed(model,scaler,new),trusted=embed(model,scaler,c['new_valid'][:32]),trusted_ids=[r['id'] for r in c['new_valid'][:32]],bad=embed(model,scaler,bad),bad_ids=[r['id'] for r in bad],cal_old=embed(model,scaler,c['calib_old']),cal_new=embed(model,scaler,c['calib_new']),n=n,queries={domain:(c[key]+c['decision_attack'],embed(model,scaler,c[key]+c['decision_attack'])) for domain,key in [('old','decision_old'),('new','decision_new')]})
  for rate in [0,.01,.05,.10]:
   for guarded in [False,True]:
    metrics={domain:dict(cm=np.zeros((2,2),dtype=np.int64),aucs=[]) for domain in ['old','new']};injected=total=rejected_bad=rejected_normal=remaining_bad=promoted=0
    for cid,d in data.items():
     count=min(int(2*d['n']*rate),d['n'],len(d['bad']));new=d['new'].copy();new[:count]=d['bad'][:count];injected+=count;total+=2*d['n'];mask=np.ones(len(new),dtype=bool)
     if guarded:
      ng=min(32,len(d['old']),len(d['trusted']));guard=GuardHead('input_knn5').fit(d['old'][:ng,:115],d['trusted'][:ng,:115])
      guard.threshold=max(float(np.quantile(guard.score(d['cal_old'][:,:115]),.99)),float(np.quantile(guard.score(d['cal_new'][:,:115]),.99)))
      mask=guard.score(new[:,:115])<=guard.threshold
     rejected_bad+=int(np.sum(~mask[:count]));rejected_normal+=int(np.sum(~mask[count:]));remaining_bad+=int(np.sum(mask[:count]));promoted+=int(np.sum(mask))
     selected=new[mask]
     if len(selected)<5:selected=d['trusted']
     nref=min(len(d['old']),len(selected));h=NormalHead(method).fit(d['old'][:nref],selected[:nref]).calibrate(d['cal_old'],d['cal_new'])
     audit.append(dict(seed=seed,rate=rate,guarded=guarded,client=cid,injected_actual_attack_ids=d['bad_ids'][:count],trusted_normal_anchor_ids=d['trusted_ids'],scope='Verifier-error stress only. Gate uses features, never actual query labels. Extra trusted B anchors are required.'))
     for domain,(rows,x) in d['queries'].items():
      y=np.array([r['y'] for r in rows]);score=h.score(x);pred=(score>h.threshold).astype(int);metrics[domain]['cm']+=confusion_matrix(y,pred,labels=[0,1]);metrics[domain]['aucs'].append(float(roc_auc_score(y,score)))
    record=dict(seed=seed,requested_reference_contamination=rate,guarded=guarded,actual_contamination=injected/total,injected_attacks=injected,reference_rows=total,guard_rejected_actual_attacks=rejected_bad,guard_rejected_normal_candidates=rejected_normal,remaining_actual_attacks=remaining_bad,promoted_candidates=promoted,metrics={})
    for domain,m in metrics.items():
     tn,fp,fn,tp=m['cm'].ravel();record['metrics'][domain]=dict(mean_client_auc=float(np.mean(m['aucs'])),fpr=float(fp/(tn+fp)),tpr=float(tp/(tp+fn)),confusion_matrix=m['cm'].tolist())
    results.append(record);B.atomic(OUT/'results.json',results)
B.atomic(OUT/'injection_audit.json',audit);B.atomic(OUT/'complete.json',dict(records=len(results),method=method,scope='Development-only stress. Candidate attacks are disjoint from development evaluation attacks. Attacks were deliberately misverified as normal to quantify a risk; no primary model bundle overwritten. Extra32 audited B normals/client are a new assumption, not a free guarantee.'));B.atomic(OUT/'status.json',dict(state='complete'))
print('Trusted-anchor guard stress complete',len(results))
