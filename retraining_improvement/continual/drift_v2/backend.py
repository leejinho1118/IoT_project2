"""원본 모듈을 사용하는 V2 학습 백엔드. 기존 V1 함수의 실행 의미를 보존한다."""
import sys,os,json,time,random,copy,pickle,csv,gzip,logging
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,confusion_matrix
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from Model import Shrink_Autoencoder
from Model.Centroid import CentroidBasedOneClassClassifier
from DataLoader import IoTDataset
from Trainer import ClientTrainer,GlobalAggregator

def configure(config,device):
    global CFG,ARGS
    CFG=config
    ARGS=SimpleNamespace(device=device)

def atomic(path,obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    for i in range(100):
        try: os.replace(tmp,path); return
        except PermissionError:
            if i==99: raise
            time.sleep(.05)

def seeds(n): random.seed(n); np.random.seed(n); torch.manual_seed(n)

def payload(rows): return np.asarray([r['x'] for r in rows],dtype=np.float64).reshape(-1,115)

def metadata(rows): return [{k:v for k,v in r.items() if k!='x'} for r in rows]

def loader(c,rows):
    if any(x['y']!=0 for x in rows): raise ValueError('Attack entered normal training')
    x=c['scaler'].transform(payload(rows)); return DataLoader(IoTDataset(x,np.zeros(len(x))),batch_size=CFG['batch_size'],pin_memory=ARGS.device=='cuda')

def fit(model,clients,train_rows,validation_key,rounds,seed,folder):
    seeds(seed)
    agg=GlobalAggregator(model,update_type='mse_avg')
    minimum=min(len(c['dev']) for c in clients)
    rr=np.random.default_rng(seed)
    dev=np.concatenate([payload(c['dev'])[rr.choice(len(c['dev']),minimum,replace=False)] for c in clients])
    agg.create_dev_dataset({'dataset':dev})
    selection=random.Random(seed); history=[]; best=float('inf'); worse=0; begin=time.perf_counter()
    for r in range(rounds):
        chosen=selection.sample(range(len(clients)),max(1,int(.5*len(clients))))
        weights=[]; trained=[]
        total=sum(len(train_rows[clients[i]['id']]) for i in chosen)
        for i in chosen:
            c=clients[i]; rows=train_rows[c['id']]
            # Deliberately retain upstream shared-model sequential semantics.
            trainer=ClientTrainer(model=agg.model,save_dir=str(folder/f'round_{r+1}'/f'client_{c["id"]}'),epoch=CFG['local_epochs'],lr_rate=CFG['lr'],update_type='mse_avg')
            trainer.run(loader(c,rows),loader(c,c[validation_key]))
            weights.append((copy.deepcopy(trainer.model.state_dict()),total,len(rows)))
            tracking=Path(trainer.save_dir)/'training_tracking.pkl'
            with tracking.open('rb') as f: epochs=len(pickle.load(f))
            trained.append(dict(client=c['id'],rows=len(rows),epochs=epochs,optimizer_steps=epochs*len(loader(c,rows))))
        agg.update(weights)
        history.append(dict(round=r+1,participants=[clients[i]['id'] for i in chosen],global_loss=agg.val_loss,local=trained))
        # Preserve upstream two-if early-stop behavior exactly.
        if agg.val_loss<best: best=agg.val_loss; worse=0
        if agg.val_loss>=best:
            worse+=1
            if worse>1: break
    elapsed=time.perf_counter()-begin
    atomic(folder/'training.json',dict(seconds=elapsed,rounds=history,original_model_sharing=True,original_global_early_stop=True,max_rounds=rounds))
    return agg.model,elapsed,history

def encode(model,c,rows):
    x=c['scaler'].transform(payload(rows)); values=[]; device=next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for start in range(0,len(x),256): values.append(model(torch.tensor(x[start:start+256],dtype=torch.float32,device=device))[0].cpu().numpy())
    return np.concatenate(values)

def detector(model,c,reference):
    if any(r['y']!=0 for r in reference): raise ValueError('Attack entered CEN reference')
    cen=CentroidBasedOneClassClassifier(); cen.fit(encode(model,c,reference))
    threshold=float(np.quantile(cen.get_density(encode(model,c,c['valid'])),CFG['normal_validation_quantile']))
    return cen,threshold

def evaluate(model,clients,reference,folder,stage):
    folder.mkdir(parents=True,exist_ok=True); result={}; detectors={}; prediction_rows=[]
    for c in clients: detectors[c['id']]=detector(model,c,reference[c['id']])
    for domain,normal_key,attack_key in [('old','old_normal','old_attack'),('new','new_normal','new_attack')]:
        rows=[]; cms=np.zeros((2,2),dtype=int)
        for c in clients:
            cen,threshold=detectors[c['id']]; items=c[normal_key]+c[attack_key]
            y=np.array([x['y'] for x in items]); score=cen.get_density(encode(model,c,items)); pred=(score>threshold).astype(int)
            if not np.isfinite(score).all(): raise ValueError('Nonfinite anomaly score')
            cm=confusion_matrix(y,pred,labels=[0,1]); cms+=cm; tn,fp,fn,tp=cm.ravel()
            rows.append(dict(client=c['id'],auc=float(roc_auc_score(y,score)),fpr=float(fp/(tn+fp)),tpr=float(tp/(tp+fn)),threshold=threshold,normal=int(tn+fp),attack=int(tp+fn)))
            prediction_rows.extend(dict(id=x['id'],domain=domain,client=c['id'],label=int(yy),score=float(ss),prediction=int(pp),threshold=threshold) for x,yy,ss,pp in zip(items,y,score,pred))
        tn,fp,fn,tp=cms.ravel(); result[domain]=dict(mean_client_auc=float(np.mean([x['auc'] for x in rows])),fpr=float(fp/(tn+fp)),tpr=float(tp/(tp+fn)),confusion_matrix=cms.tolist(),clients=rows)
    atomic(folder/'metrics.json',result)
    with gzip.open(folder/'predictions.csv.gz','wt',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    bundle=dict(stage=stage,model_state={k:v.cpu().clone() for k,v in model.state_dict().items()},architecture=dict(input_dim=115,output_dim=115,hidden_neus=50,latent_dim=11,shrink_lambda=10),clients={c['id']:dict(input_scaler=c['scaler'],cen=detectors[c['id']][0],threshold=detectors[c['id']][1],threshold_source_ids=[x['id'] for x in c['valid']],reference_ids=[x['id'] for x in reference[c['id']]]) for c in clients},config=CFG)
    torch.save(bundle,folder/'detector_bundle.pt')
    return result,detectors

def log_current_detection(model,clients,reference,stage,folder):
    rows=[]
    for c in clients:
        items=c['candidate_stages'][stage]
        cen,threshold=detector(model,c,reference[c['id']])
        score=cen.get_density(encode(model,c,items))
        rows.extend(dict(id=x['id'],client=c['id'],score=float(v),anomaly_flag=bool(v>threshold),threshold=threshold) for x,v in zip(items,score))
    atomic(folder/'detections_before_update.json',dict(policy='Current arm model detects arrivals before this update. Shared verifier outcomes are fixed across arms; no prediction is treated as a true label.',items=rows))
