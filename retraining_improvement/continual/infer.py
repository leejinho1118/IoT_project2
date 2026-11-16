"""Label-free CPU inference from an exported continual detector bundle."""
from pathlib import Path
import os,sys,argparse,json,csv,hashlib
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from Model import Shrink_Autoencoder
P=argparse.ArgumentParser()
P.add_argument('--bundle',required=True)
P.add_argument('--client',type=int,required=True)
P.add_argument('--input',required=True)
P.add_argument('--out',required=True)
a=P.parse_args()
torch.set_num_threads(1)
if torch.cuda.is_available(): raise RuntimeError('Expected CPU-only inference process')
source=Path(a.input); target=Path(a.out)
if target.exists(): raise FileExistsError('Prediction output exists; refusing to overwrite')
bundle=torch.load(a.bundle,map_location='cpu',weights_only=False)
if a.client not in bundle['clients']: raise ValueError('Unknown client; an explicit matching scaler/CEN is required')
c=bundle['clients'][a.client]
x=pd.read_csv(source,header=None).to_numpy(dtype=np.float64)
if x.ndim!=2 or x.shape[1]!=115 or not len(x) or not np.isfinite(x).all(): raise ValueError('Expected finite, headerless CSV with exactly 115 input features and no label column')
model=Shrink_Autoencoder(**bundle['architecture']); model.load_state_dict(bundle['model_state']); model.eval()
z=[]; scaled=c['input_scaler'].transform(x)
with torch.no_grad():
 for i in range(0,len(x),256): z.append(model(torch.tensor(scaled[i:i+256],dtype=torch.float32))[0].numpy())
scores=c['cen'].get_density(np.concatenate(z))
if not np.isfinite(scores).all(): raise ValueError('Nonfinite anomaly score')
threshold=float(c['threshold']); pred=(scores>threshold).astype(int)
target.parent.mkdir(parents=True,exist_ok=True)
with target.open('w',encoding='utf-8',newline='') as f:
 writer=csv.writer(f); writer.writerow(['row_index','client','score','threshold','prediction'])
 writer.writerows((i,a.client,float(s),threshold,int(p)) for i,(s,p) in enumerate(zip(scores,pred)))
proof=dict(bundle=str(Path(a.bundle).resolve()),bundle_sha256=hashlib.sha256(Path(a.bundle).read_bytes()).hexdigest(),input=str(source.resolve()),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=len(x),features=115,client=a.client,labels_used=False,device='cpu',torch=torch.__version__,pandas=pd.__version__,stage=bundle['stage'])
target.with_suffix('.metadata.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps(proof,indent=2))
