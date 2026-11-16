"""저장된 정상 탐지 헤드를 115열 CSV에 적용한다. 정답·A/B 도메인을 읽지 않는다."""
import os,sys,argparse,json,csv,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'continual/drift_v2'))
import numpy as np
import pandas as pd
import torch
import scenario,metric_models,models
from Model import Shrink_Autoencoder
P=argparse.ArgumentParser();P.add_argument('--bundle',required=True);P.add_argument('--client',type=int,required=True);P.add_argument('--input',required=True);P.add_argument('--out',required=True);a=P.parse_args()
target=Path(a.out)
if target.exists():raise FileExistsError('Existing predictions preserved')
torch.set_num_threads(1);bundle=torch.load(a.bundle,map_location='cpu',weights_only=False);c=bundle['clients'][a.client];method=bundle['method']
x=pd.read_csv(a.input,header=None).to_numpy(dtype=np.float64)
if x.ndim!=2 or x.shape[1]!=115 or not len(x) or not np.isfinite(x).all():raise ValueError('Need finite headerless 115-column CSV')
if method=='log_input_knn5':features=x
elif method.startswith('input_'):features=c['input_scaler'].transform(x)
else:
 model=Shrink_Autoencoder(**bundle['architecture']);model.load_state_dict(bundle['model_state']);model.eval();scaled=c['input_scaler'].transform(x);features=[]
 with torch.no_grad():
  for start in range(0,len(x),256):features.append(model(torch.tensor(scaled[start:start+256],dtype=torch.float32))[0].numpy())
 features=np.concatenate(features)
 if method.startswith('residual_'):features=np.concatenate([scaled,features],axis=1)
score=c['head'].score(features);threshold=c['head'].threshold;pred=(score>threshold).astype(int);target.parent.mkdir(parents=True,exist_ok=True)
with target.open('w',encoding='utf-8',newline='') as f:
 w=csv.writer(f);w.writerow(['row_index','client','score','threshold','prediction']);w.writerows((i,a.client,float(s),threshold,int(p)) for i,(s,p) in enumerate(zip(score,pred)))
proof=dict(method=method,client=a.client,rows=len(x),labels_used=False,domain_oracle_used=False,device='cpu',bundle_sha256=hashlib.sha256(Path(a.bundle).read_bytes()).hexdigest(),input_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest())
target.with_suffix('.metadata.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))
