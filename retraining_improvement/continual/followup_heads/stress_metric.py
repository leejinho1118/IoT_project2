"""선택된 잔차 기억의 검증자 오판 민감도. 공격을 정상으로 잘못 넣는 격리된 스트레스 시험."""
import sys,argparse,json,time,gzip,hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,confusion_matrix
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'continual/research_ABCDE'));sys.path.insert(0,str(ROOT/'continual/drift_v2'))
from data import load_development,B
from metric_models import NormalHead
P=argparse.ArgumentParser();P.add_argument('--run-id',required=True);a=P.parse_args();HOME=Path(__file__).resolve().parent;PRIMARY=HOME/'output/metrics_v1';SOURCE=ROOT/'continual/research_ABCDE/output/campaign_full_v1';OUT=HOME/'output'/a.run_id
if OUT.exists():raise RuntimeError('Existing stress result preserved')
OUT.mkdir(parents=True);torch.set_num_threads(1);cfg=json.loads((SOURCE/'config.json').read_text(encoding='utf-8'));method=json.loads((PRIMARY/'selection_frozen.json').read_text(encoding='utf-8'))['selected_method'];B.configure(cfg,'cpu')
clients,_=load_development(ROOT/'continual/drift_v2/output/full_v1',OUT,cfg);results=[];audit=[]
def embed(model,scaler,rows):
 # 입력 기반 선택 모델은 SAE를 실행하지 않는다.
 return scaler.transform(B.payload(rows))
with threadpool_limits(limits=1):
 for seed in cfg['seeds']:
  B.atomic(OUT/'status.json',dict(state='running',seed=seed));source=SOURCE/f'A/seed_{seed}/stage_3/published/detector_bundle.pt';b=torch.load(source,map_location='cpu',weights_only=False);model=B.Shrink_Autoencoder(**b['architecture']).cpu();model.load_state_dict(b['model_state']);model.eval();bank=json.loads((SOURCE/f'A/seed_{seed}/memory_ids.json').read_text(encoding='utf-8'));data={}
  for c in clients:
   cid=c['id'];scaler=b['clients'][cid]['input_scaler'];by={r['id']:r for r in c['train']};old=[by[i] for i in bank[str(cid)]];new=[r for r in c['candidate_stages'][2] if r['y']==0];bad=[r for r in c['candidate_stages'][2] if r['y']==1];n=min(128,len(old),len(new));old=old[:n];new=new[:n]
   data[cid]=dict(old=embed(model,scaler,old),new=embed(model,scaler,new),bad=embed(model,scaler,bad),bad_ids=[r['id'] for r in bad],cal_old=embed(model,scaler,c['calib_old']),cal_new=embed(model,scaler,c['calib_new']),n=n,queries={domain:(c[key]+c['decision_attack'],embed(model,scaler,c[key]+c['decision_attack'])) for domain,key in [('old','decision_old'),('new','decision_new')]})
  for rate in [0,.01,.05,.10]:
   metrics={domain:dict(cm=np.zeros((2,2),dtype=np.int64),aucs=[]) for domain in ['old','new']};injected=total=0
   for cid,d in data.items():
    count=min(int(2*d['n']*rate),d['n'],len(d['bad']));new=d['new'].copy();new[:count]=d['bad'][:count];h=NormalHead(method).fit(d['old'],new).calibrate(d['cal_old'],d['cal_new']);injected+=count;total+=2*d['n'];audit.append(dict(seed=seed,rate=rate,client=cid,injected_actual_attack_ids=d['bad_ids'][:count],scope='Deliberate verifier-error stress only; never primary normal training'))
    for domain,(rows,x) in d['queries'].items():
     y=np.array([r['y'] for r in rows]);score=h.score(x);pred=(score>h.threshold).astype(int);metrics[domain]['cm']+=confusion_matrix(y,pred,labels=[0,1]);metrics[domain]['aucs'].append(float(roc_auc_score(y,score)))
   record=dict(seed=seed,requested_reference_contamination=rate,actual_contamination=injected/total,injected_attacks=injected,reference_rows=total,metrics={})
   for domain,m in metrics.items():
    tn,fp,fn,tp=m['cm'].ravel();record['metrics'][domain]=dict(mean_client_auc=float(np.mean(m['aucs'])),fpr=float(fp/(tn+fp)),tpr=float(tp/(tp+fn)),confusion_matrix=m['cm'].tolist())
   results.append(record);B.atomic(OUT/'results.json',results)
B.atomic(OUT/'injection_audit.json',audit);B.atomic(OUT/'complete.json',dict(records=len(results),method=method,scope='Development-only stress. Candidate attacks are disjoint from development evaluation attacks. Attacks were deliberately misverified as normal to quantify a risk; no primary model bundle overwritten.'));B.atomic(OUT/'status.json',dict(state='complete'))
print('Verifier-error stress complete',len(results))
