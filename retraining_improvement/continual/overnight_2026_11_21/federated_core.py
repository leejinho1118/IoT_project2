"""Shared neural trainer for prospectively frozen raw/external studies.
Derived from completed V3 fit_stage; upstream SAE/MSEAvg code stays unchanged.
"""
from pathlib import Path
from types import SimpleNamespace
import sys,copy,time,random,json,hashlib
import numpy as np
import torch
import sklearn.neighbors
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'continual/drift_v2'))
import backend as B
sys.path.insert(0,str(ROOT/'continual/retraining_v3'))
from losses import retraining_loss
from checkpoint_guard import eligible
from transforms import feature_map
ORIGINAL_AGGREGATOR=B.GlobalAggregator
CFG={};args=SimpleNamespace(device='cpu');STATUS_CALLBACK=lambda **kw:None

class FeatureMapAggregator(ORIGINAL_AGGREGATOR):
    def create_dev_dataset(self,dataset):
        mapped=feature_map(dataset['dataset'],CFG['mode'])
        super().create_dev_dataset({'dataset':mapped})
    def update(self,local_models):
        super().update(local_models)
        directory=CFG.get('round_checkpoint_dir')
        if directory:
            folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
            self._checkpoint_round=getattr(self,'_checkpoint_round',0)+1
            path=folder/f'round_{self._checkpoint_round}.pt'
            if path.exists():raise RuntimeError('Round checkpoint would be overwritten')
            torch.save(dict(model_state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},weights_sha256=whash(self.model)),path)


class StudySAE(B.Shrink_Autoencoder):
    def __init__(self,shrink_kind='norm',**architecture):
        super().__init__(**architecture);self.shrink_kind=shrink_kind
    def forward(self,x):
        if self.shrink_kind=='norm':return super().forward(x)
        z=self.encoder(x);recon=self.decoder(z)
        loss=torch.nn.functional.mse_loss(x,recon)+self.shrink_lambda*torch.mean(torch.sum(z*z,dim=1))
        return z,recon,loss

def payload(rows):
    if not rows:return np.empty((0,CFG['input_dim']),dtype=np.float64)
    return np.asarray([r['x'] for r in rows],dtype=np.float64).reshape(-1,CFG['input_dim'])

def configure(config,device,status_callback=None):
    global CFG,args,STATUS_CALLBACK
    CFG=dict(config);args=SimpleNamespace(device=device);STATUS_CALLBACK=status_callback or (lambda **kw:None)
    B.payload=payload;B.GlobalAggregator=FeatureMapAggregator;B.configure(CFG,device)

def atomic(path,data):B.atomic(path,data)
def status(**kw):STATUS_CALLBACK(**kw)
def selection_score(metrics,initial):
    old=metrics['old'];new=metrics['new']
    return .5*(old['mean_client_auc']+new['mean_client_auc'])+.25*min(old['tpr'],new['tpr'])-.5*max(0.,max(old['fpr'],new['fpr'])-.10)

def whash(model):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()

def fit_stage(model,teacher,clients,train,bank,policy,seed,stage,folder,callback,initial_dev,previous_dev):
 B.seeds(seed+100000+stage);selection=random.Random(seed+100000+stage)
 agg=B.GlobalAggregator(model,update_type='mse_avg')
 rr=np.random.default_rng(seed+stage);minimum=min(len(c['dev']) for c in clients)
 dev=np.concatenate([B.payload(c['dev'])[rr.choice(len(c['dev']),minimum,replace=False)] for c in clients]);agg.create_dev_dataset({'dataset':dev})
 start=time.perf_counter();prepared={};teacher.eval();device=args.device
 for c in clients:
  cid=c['id'];rs=train[cid];assert all(r['y']==0 for r in rs)
  x=torch.as_tensor(c['scaler'].transform(B.payload(rs)),dtype=torch.float32,device=device)
  old_ids={r['id'] for r in bank[cid]};old=torch.tensor([r['id'] in old_ids for r in rs],dtype=torch.bool,device=device)
  bx=torch.as_tensor(c['scaler'].transform(B.payload(bank[cid])),dtype=torch.float32,device=device)
  with torch.no_grad():
   tz=teacher.encoder(x).detach();bz=teacher.encoder(bx);center=bz.mean(0).detach();scale=bz.std(0,unbiased=False).clamp_min(CFG['scale_floor']).detach()
  valid_rows=c['valid']+c['new_valid']
  vx=torch.as_tensor(c['scaler'].transform(B.payload(valid_rows)),dtype=torch.float32,device=device)
  prepared[cid]=(x,old,tz,center,scale,vx)
 history=[];best=float('inf');worse=0
 previous_state=copy.deepcopy(model.state_dict());chosen_state=None;chosen_score=selection_score(previous_dev,initial_dev);chosen_round=0;selection_audit=[]
 for rnd in range(1,CFG['rounds']+1):
  chosen=selection.sample(range(len(clients)),max(1,int(.5*len(clients))))
  weights=[];local=[];total=sum(len(train[clients[i]['id']]) for i in chosen)
  for i in chosen:
   c=clients[i];cid=c['id'];x,old,tz,center,scale,vx=prepared[cid]
   opt=torch.optim.Adam(model.parameters(),lr=CFG['lr']);minval=float('inf');patience=0;tracking=[];clipped=0
   for epoch in range(CFG['local_epochs']):
    model.train();train_loss=0.
    for offset in range(0,len(x),CFG['batch_size']):
     sl=slice(offset,offset+CFG['batch_size']);batch=x[sl];opt.zero_grad(set_to_none=True)
     z,recon,base=model(batch)
     negative_z=None
     if policy.get('negative_weight',0.):
      permutation=torch.rand(batch.shape,device=device).argsort(dim=0)
      corrupted=torch.gather(batch,0,permutation)
      negative_z=model.encoder(corrupted)
     loss,parts=retraining_loss(batch,z,recon,base,old[sl],tz[sl],center,scale,policy,model.shrink_lambda,negative_z=negative_z)
     if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['gradient_clip'],error_if_nonfinite=True)
     clipped+=int(float(norm)>CFG['gradient_clip']);opt.step();train_loss+=float(loss.detach())
    model.eval();val=0.
    with torch.no_grad():
     for offset in range(0,len(vx),CFG['batch_size']):val+=float(model(vx[offset:offset+CFG['batch_size']])[2])
    val/=int(np.ceil(len(vx)/CFG['batch_size']));tracking.append(dict(epoch=epoch+1,train_loss=train_loss/int(np.ceil(len(x)/CFG['batch_size'])),validation_base_loss=val))
    if val<minval:minval=val;patience=0
    else:
     patience+=1
     if patience>=3:break
   steps=len(tracking)*int(np.ceil(len(x)/CFG['batch_size']))
   local.append(dict(client=cid,epochs=len(tracking),rows=len(x),optimizer_steps=steps,synthetic_negative_presentations=len(tracking)*len(x) if policy.get('negative_weight',0.) else 0,clipped_steps=clipped,tracking=tracking))
   weights.append((copy.deepcopy(model.state_dict()),total,len(x)))
  agg.update(weights);model=agg.model
  dev_metrics=callback(model,rnd)
  if policy.get('select_round',False):
   score=selection_score(dev_metrics,initial_dev)
   passes=eligible(dev_metrics,previous_dev,initial_dev)
   if passes and score>chosen_score+1e-4:
    chosen_state=copy.deepcopy(model.state_dict());chosen_score=score;chosen_round=rnd
   selection_audit.append(dict(round=rnd,score=score,guard_pass=bool(passes),state_sha256=whash(model)))
  history.append(dict(round=rnd,participants=[clients[i]['id'] for i in chosen],global_loss=float(agg.val_loss),local=local,development=dev_metrics))
  atomic(folder/'training.json',dict(rounds=history,seconds=time.perf_counter()-start))
  status(state='training',round=rnd,global_loss=float(agg.val_loss),development=dev_metrics)
  # Preserve the upstream two-if global patience behavior.
  if agg.val_loss<best:best=agg.val_loss;worse=0
  if agg.val_loss>=best:
   worse+=1
   if worse>1:break
 checkpoint=dict(enabled=bool(policy.get('select_round',False)),accepted=True)
 if policy.get('select_round',False):
  model.load_state_dict(chosen_state if chosen_state is not None else previous_state)
  checkpoint.update(accepted=chosen_state is not None,selected_round=chosen_round,selected_score=chosen_score,candidate_checks=selection_audit,selected_weights_sha256=whash(model),guard='old AUC >= initial - .03; old/new FPR <= previous + .02; old/new TPR >= previous - .02; primary score improves > 1e-4')
  atomic(folder/'checkpoint_selection.json',checkpoint)
 return model,time.perf_counter()-start,history,checkpoint
