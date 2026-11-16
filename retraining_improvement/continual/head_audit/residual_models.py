"""잠재 표현에 입력 특징 잔차를 보조 연결하는 정상 기억 탐지기."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from head_models import NormalHead as BaseHead
METHODS=['centroid_original','threshold_updated','input_knn5','log_input_knn5','residual_knn_w025','residual_knn_w050','residual_knn_w100']
DIAGNOSTICS={'input_knn5','log_input_knn5'}
WEIGHTS={'residual_knn_w025':.25,'residual_knn_w050':.5,'residual_knn_w100':1.}


class NormalHead(BaseHead):
    def __init__(self,method):
        if method in WEIGHTS:self.method=method;self.weight=WEIGHTS[method]
        else:super().__init__(method)
    def fit(self,old,new,original=None,original_threshold=None):
        if self.method not in WEIGHTS:return super().fit(old,new,original,original_threshold)
        old=np.asarray(old,dtype=np.float64);new=np.asarray(new,dtype=np.float64);x=np.concatenate([old,new])
        if x.shape[1]!=126 or not np.isfinite(x).all():raise ValueError('Need 115 input plus11 latent features')
        self.input_scaler=StandardScaler().fit(x[:,:115]);self.latent_scaler=StandardScaler().fit(x[:,115:])
        self.input_scaler.scale_=np.maximum(self.input_scaler.scale_,1e-6);self.latent_scaler.scale_=np.maximum(self.latent_scaler.scale_,1e-6)
        self.reference_rows=len(x);self.input_features=126;self.model=NearestNeighbors(n_neighbors=min(5,len(x)),algorithm='brute',n_jobs=1).fit(self.transform(x));return self
    def transform(self,x):
        # 블록 차원 차이로 한쪽이 자동 지배하지 않도록 sqrt(dim)으로 정규화한다.
        return np.concatenate([self.input_scaler.transform(x[:,:115])/np.sqrt(115),self.weight*self.latent_scaler.transform(x[:,115:])/np.sqrt(11)],axis=1)
    def score(self,x):
        if self.method not in WEIGHTS:return super().score(x)
        score=self.model.kneighbors(self.transform(np.asarray(x,dtype=np.float64)),return_distance=True)[0].mean(axis=1)
        if not np.isfinite(score).all():raise ValueError('Nonfinite residual score')
        return score
