"""Prospective normal-only selection: attack labels are monitoring only."""
import numpy as np
import torch
import federated_core as base
from federated_core import *
from study_metrics import evaluate as original_evaluate

def normal_score(metrics,initial):
    return -.5*sum(np.log(metrics[d]['normal_reconstruction_mse']+1e-12) for d in ['old','new'])
def normal_eligible(current,previous,initial):
    return current['old']['normal_reconstruction_mse']<=initial['old']['normal_reconstruction_mse']*1.05 and all(current[d]['fpr']<=previous[d]['fpr']+.02 for d in ['old','new'])
# fit_stage resolves these globals in its defining module; no upstream file is edited.
base.selection_score=normal_score;base.eligible=normal_eligible
selection_score=normal_score

def evaluate(model,clients,det,final=None,save=None,initial_old_only=False):
    result=original_evaluate(model,clients,det,final=final,save=save,initial_old_only=initial_old_only)
    if final is not None:return result
    for domain in ['old'] if initial_old_only else ['old','new']:
        values=[]
        for c in clients:
            rows=c['decision_old' if domain=='old' else 'decision_new'];x=torch.as_tensor(c['scaler'].transform(base.B.payload(rows)),dtype=torch.float32,device=next(model.parameters()).device)
            with torch.no_grad():_,recon,_=model(x);values.append(float(torch.mean((recon-x)**2)))
        result[domain]['normal_reconstruction_mse']=float(np.mean(values))
    if initial_old_only:result['new']['normal_reconstruction_mse']=result['old']['normal_reconstruction_mse']
    return result

_original_fit_stage=base.fit_stage
def fit_stage(*args,**kwargs):
    result=_original_fit_stage(*args,**kwargs)
    result[3]['guard']='Normal-only: old reconstruction MSE <= initial*1.05; old/new FPR <= recalibrated previous +.02; normal score gain >1e-4; no attack AUC/TPR in selection'
    folder=args[8]
    atomic(folder/'checkpoint_selection.json',result[3])
    return result

