"""Development-only distribution diagnostic; not used to train/select any model."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json,datetime,hashlib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from study_data import load_development,activate
from transforms import InputTransform
H=Path(__file__).resolve().parent;OUT=H/'development_shift_diagnostic.json';assert not OUT.exists();records=[]
for dataset in ['raw_blocks_v2','rt_thing_to_mqtt_v1','rt_mqtt_to_thing_v1']:
 clients,_=load_development(H/dataset)
 for c in clients:c['scaler']=InputTransform('signed_log').fit(np.asarray([r['x'] for r in c['train']]))
 for stage in [1,2,3]:
  activate(clients,stage)
  for c in clients:
   normal=[r for r in c['candidate_stages'][stage-1] if r['y']==0];n=min(128,len(c['train']),len(normal));rng=np.random.default_rng(20261121+stage*100+c['id']);old=[c['train'][int(i)] for i in rng.choice(len(c['train']),n,replace=False)];new=[normal[int(i)] for i in rng.choice(len(normal),n,replace=False)];x=c['scaler'].transform(np.array([r['x'] for r in old]));y=c['scaler'].transform(np.array([r['x'] for r in new]));z=np.concatenate([x,y]);d=cdist(z,z,'sqeuclidean');positive=d[d>0];sigma2=float(np.median(positive)) if len(positive) else 1.;kernel=np.exp(-d/(2*sigma2));v=np.r_[np.full(n,1/n),np.full(n,-1/n)];value=float(v@kernel@v);assert value>=-1e-12;null=[]
   for _ in range(199):vv=rng.permutation(v);null.append(float(vv@kernel@vv))
   nominal=(1+sum(a>=value-1e-15 for a in null))/200.;w=np.array([wasserstein_distance(x[:,i],y[:,i]) for i in range(x.shape[1])]);records.append(dict(dataset=dataset,stage=stage,client=c['id'],device=c['device'],samples_each=n,rbf_bandwidth_sigma_squared=sigma2,biased_MMD_squared=max(0,value),nominal_permutation_p=nominal,standardized_wasserstein_median=float(np.median(w)),standardized_wasserstein_p90=float(np.quantile(w,.9)),old_ids=[r['id'] for r in old],new_ids=[r['id'] for r in new]))
summary=[]
for dataset in ['raw_blocks_v2','rt_thing_to_mqtt_v1','rt_mqtt_to_thing_v1']:
 for stage in [1,2,3]:
  rr=[r for r in records if r['dataset']==dataset and r['stage']==stage];mm=[r['biased_MMD_squared'] for r in rr];summary.append(dict(dataset=dataset,stage=stage,clients=len(rr),MMD_squared_min=float(min(mm)),MMD_squared_median=float(np.median(mm)),MMD_squared_max=float(max(mm))))
obj=dict(state='complete',created=datetime.datetime.now().astimezone().isoformat(),source='https://www.jmlr.org/papers/v13/gretton12a.html',scope='Descriptive RBF biased MMD squared and feature Wasserstein distance on development normals only. No final features or labels opened. No training or selection uses these values.',method='Initial normal signed-log StandardScaler clipped20; equal fixed samples <=128; pooled positive median distance bandwidth; 199 label permutations',limits='File rows may be correlated. Nominal permutation p assumes exchangeability and is not a proven time-drift claim; 57 comparisons are exploratory, not corrected confirmatory tests. MMD values are kernel/sample/scaler dependent and not universal cross-dataset units.',records=records,summary=summary);OUT.write_text(json.dumps(obj,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
