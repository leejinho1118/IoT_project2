"""거리 선택의 통제 비교. 참조/보정/SAE 가중치를 바꾸지 않는다."""
from models import NormalHead as PriorHead
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
METHODS=['centroid_original','threshold_updated','input_knn5','residual_knn_w025','input_l1','input_cosine','input_chebyshev','residual_l1']
DIAGNOSTICS=set();NEW={'input_l1','input_cosine','input_chebyshev','residual_l1'}
class NormalHead(PriorHead):
 def __init__(self,method):
  if method in NEW:self.method=method
  else:super().__init__(method)
 def fit(self,old,new,original=None,original_threshold=None):
  if self.method not in NEW:return super().fit(old,new,original,original_threshold)
  x=np.concatenate([old,new]).astype(float)
  if not np.isfinite(x).all():raise ValueError('Nonfinite normal reference')
  self.reference_rows=len(x);self.input_features=x.shape[1]
  if self.method=='residual_l1':
   self.scaler=StandardScaler().fit(x[:,:115]);self.latent=StandardScaler().fit(x[:,115:]);self.latent.scale_=np.maximum(self.latent.scale_,1e-6)
  else:self.scaler=StandardScaler().fit(x)
  self.scaler.scale_=np.maximum(self.scaler.scale_,1e-6)
  metric={'input_l1':'manhattan','residual_l1':'manhattan','input_cosine':'cosine','input_chebyshev':'chebyshev'}[self.method]
  self.model=NearestNeighbors(n_neighbors=min(5,len(x)),algorithm='brute',metric=metric,n_jobs=1).fit(self.transform_metric(x));return self
 def transform_metric(self,x):
  if self.method=='residual_l1':return np.concatenate([self.scaler.transform(x[:,:115])/np.sqrt(115),.25*self.latent.transform(x[:,115:])/np.sqrt(11)],axis=1)
  return self.scaler.transform(x)
 def score(self,x):
  if self.method not in NEW:return super().score(x)
  x=np.asarray(x,dtype=float)
  if not np.isfinite(x).all():raise ValueError('Nonfinite query')
  s=self.model.kneighbors(self.transform_metric(x),return_distance=True)[0].mean(axis=1)
  if not np.isfinite(s).all():raise ValueError('Nonfinite score')
  return s
