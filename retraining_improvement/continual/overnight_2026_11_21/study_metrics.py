"""Unchanged CEN scoring with explicit normal-only threshold provenance."""
from pathlib import Path
import json,copy,gzip,csv,collections
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
import federated_core as C


def detectors(model,clients,reference,use_new=True):
    result={}
    for c in clients:
        cid=c['id'];ref=reference[cid]
        if any(r['y']!=0 for r in ref):raise ValueError('Non-normal fitted CEN reference')
        cen=C.B.CentroidBasedOneClassClassifier();cen.fit(C.B.encode(model,c,ref))
        old=cen.get_density(C.B.encode(model,c,c['calib_old']));q_old=float(np.quantile(old,.95))
        q_new=None
        if use_new and c['calib_new']:
            new=cen.get_density(C.B.encode(model,c,c['calib_new']));q_new=float(np.quantile(new,.95))
        threshold=max(q_old,q_new) if q_new is not None else q_old
        result[cid]=dict(cen=cen,threshold=threshold,old_threshold=q_old,new_threshold=q_new,reference_ids=[r['id'] for r in ref],threshold_old_ids=[r['id'] for r in c['calib_old']],threshold_new_ids=[r['id'] for r in c['calib_new']] if q_new is not None else [],calibration='max(old normal q95, available current-stage normal q95)' if q_new is not None else 'initial old normal q95')
    return result


def evaluate(model,clients,det,final=None,save=None,initial_old_only=False):
    result={};pred=[]
    domains=['old'] if initial_old_only else ['old','new']
    for domain in domains:
        cms=np.zeros((2,2),dtype=np.int64);aucs=[];aps=[];per_client=[];groups=collections.defaultdict(lambda:[0,0]);normal_groups=collections.defaultdict(lambda:[0,0])
        for c in clients:
            cid=c['id'];normal=(c['decision_old'] if domain=='old' else c['decision_new']) if final is None else final.get(f'{cid}/final_{domain}_normal',[])
            attack=c['decision_attack'] if final is None else final.get(f'{cid}/final_{domain}_attack',[])
            items=normal+attack
            if not items:continue
            d=det[cid];score=d['cen'].get_density(C.B.encode(model,c,items));y=np.asarray([r['y'] for r in items]);prediction=(score>d['threshold']).astype(int)
            if not np.isfinite(score).all():raise ValueError('Nonfinite anomaly score')
            cm=confusion_matrix(y,prediction,labels=[0,1]);cms+=cm;tn,fp,fn,tp=cm.ravel()
            auc=float(roc_auc_score(y,score)) if len(np.unique(y))==2 else None
            if auc is not None:aucs.append(auc);aps.append(float(average_precision_score(y,score)))
            per_client.append(dict(client=cid,device=c['device'],auc=auc,fpr=float(fp/(tn+fp)) if tn+fp else None,tpr=float(tp/(tp+fn)) if tp+fn else None,normal=int(tn+fp),attack=int(tp+fn)))
            for row,yy,ss,pp in zip(items,y,score,prediction):
                if yy:
                    g=groups[f"{row.get('family','attack')}/{row.get('attack_type','unknown')}"];g[0]+=int(pp);g[1]+=1
                else:
                    g=normal_groups[row.get('normal_pattern') or row.get('device',str(cid))];g[0]+=int(pp);g[1]+=1
                if save:pred.append(dict(id=row['id'],client=cid,device=c['device'],domain=domain,label=int(yy),score=float(ss),prediction=int(pp),threshold=d['threshold'],family=row.get('family'),attack_type=row.get('attack_type'),label_conflict=bool(row.get('label_conflict',False))))
        tn,fp,fn,tp=cms.ravel()
        result[domain]=dict(mean_client_auc=float(np.mean(aucs)) if aucs else None,mean_client_pr_auc=float(np.mean(aps)) if aps else None,fpr=float(fp/(tn+fp)) if tn+fp else None,tpr=float(tp/(tp+fn)) if tp+fn else None,confusion_matrix=cms.tolist(),auc_clients=len(aucs),normal_clients=sum(x['normal']>0 for x in per_client),attack_clients=sum(x['attack']>0 for x in per_client),clients=per_client,attack_type_tpr={k:dict(detected=v[0],total=v[1],tpr=v[0]/v[1]) for k,v in groups.items()},normal_pattern_fpr={k:dict(false_positive=v[0],total=v[1],fpr=v[0]/v[1]) for k,v in normal_groups.items()})
    if initial_old_only:result['new']=copy.deepcopy(result['old']);result['initial_old_only']=True
    if save:
        with gzip.open(save,'wt',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(pred[0]));w.writeheader();w.writerows(pred)
    return result


def bundle(path,model,clients,det,architecture,metadata):
    obj=dict(architecture=architecture,model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},clients={c['id']:dict(input_scaler=c['scaler'],device=c['device'],**det[c['id']]) for c in clients},weights_sha256=C.whash(model),metadata=metadata)
    torch.save(obj,path);return obj['weights_sha256']
