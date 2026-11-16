"""정상만 사용하는 거리·밀도·분리 트리 비교. SAE 포함 여부를 명시한다."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'head_audit'))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from residual_models import NormalHead as PriorHead
METHODS=['centroid_original','threshold_updated','input_knn5','residual_knn_w025','input_whiten_knn','input_density_knn','input_isoforest','input_two_cov']
# 본 비교는 단순 독립 방법도 동등하게 평가하며 FedMSE 개선으로 자동 분류하지 않는다.
DIAGNOSTICS=set()
NEW=set(METHODS)-{'centroid_original','threshold_updated','input_knn5','residual_knn_w025'}
class NormalHead(PriorHead):
 def __init__(self,method):
  if method in NEW:self.method=method
  else:super().__init__(method)
 def fit(self,old,new,original=None,original_threshold=None):
  if self.method not in NEW:return super().fit(old,new,original,original_threshold)
  old=np.asarray(old,dtype=float);new=np.asarray(new,dtype=float);x=np.concatenate([old,new])
  if x.ndim!=2 or x.shape[1]!=115 or not np.isfinite(x).all():raise ValueError('Need finite 115-feature reference')
  self.scaler=StandardScaler().fit(x);self.scaler.scale_=np.maximum(self.scaler.scale_,1e-6);z=self.scaler.transform(x)
  self.reference_rows=len(x);self.input_features=115
  if self.method=='input_whiten_knn':
   cov=LedoitWolf().fit(z);v,u=np.linalg.eigh(cov.covariance_);self.whitener=(u/np.sqrt(np.maximum(v,1e-6)))@u.T
   self.model=NearestNeighbors(n_neighbors=min(5,len(z)),algorithm='brute',n_jobs=1).fit(z@self.whitener)
  elif self.method=='input_density_knn':
   self.model=NearestNeighbors(n_neighbors=min(6,len(z)),algorithm='brute',n_jobs=1).fit(z)
   # 참조 자체의 이웃 거리는 자기 자신 한 개를 제외한다. 중복은 floor로 처리한다.
   d=self.model.kneighbors(z,return_distance=True)[0][:,1:];self.radius=np.maximum(d.mean(axis=1),1e-6)
  elif self.method=='input_isoforest':
   self.model=IsolationForest(n_estimators=128,max_samples=min(256,len(z)),random_state=5678,n_jobs=1).fit(z)
  else:self.components=[LedoitWolf().fit(self.scaler.transform(a)) for a in [old,new]]
  return self
 def score(self,x):
  if self.method not in NEW:return super().score(x)
  x=np.asarray(x,dtype=float)
  if not np.isfinite(x).all():raise ValueError('Nonfinite query')
  z=self.scaler.transform(x)
  if self.method=='input_whiten_knn':s=self.model.kneighbors(z@self.whitener,return_distance=True)[0].mean(axis=1)
  elif self.method=='input_density_knn':
   d,i=self.model.kneighbors(z,n_neighbors=min(5,self.reference_rows),return_distance=True);s=d.mean(axis=1)/self.radius[i].mean(axis=1)
  elif self.method=='input_isoforest':s=-self.model.score_samples(z)
  else:s=np.min(np.stack([c.mahalanobis(z) for c in self.components]),axis=0)
  if not np.isfinite(s).all():raise ValueError('Nonfinite score')
  return np.asarray(s)
