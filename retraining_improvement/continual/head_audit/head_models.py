"""정상 기준만으로 적합하는 탐지 헤드. score는 클수록 이상, 실제 A/B 정답 없이 동작."""
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors,LocalOutlierFactor
from sklearn.covariance import LedoitWolf
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from Model.Centroid import CentroidBasedOneClassClassifier
METHODS=['centroid_original','threshold_updated','reference_balanced','centroid_two','latent_knn5','latent_lof20','latent_mahalanobis','input_knn5','log_input_knn5']
DIAGNOSTICS={'input_knn5','log_input_knn5'}


def updated_threshold(old_scores,new_scores):
    # 두 정상 보정 풀을 각각 보호하는 경험적 기준. 분포무관 보장은 주장하지 않는다.
    return max(float(np.quantile(old_scores,.95)),float(np.quantile(new_scores,.95)))


class NormalHead:
    def __init__(self,method):
        if method not in METHODS:raise ValueError('Unknown head')
        self.method=method
    def _log(self,x):return np.sign(x)*np.log1p(np.abs(x)) if self.method=='log_input_knn5' else x
    def fit(self,old,new,original=None,original_threshold=None):
        old=np.asarray(old,dtype=np.float64);new=np.asarray(new,dtype=np.float64)
        if old.ndim!=2 or new.ndim!=2 or len(old)<2 or len(new)<2:raise ValueError('Need two disjoint normal reference components')
        if not np.isfinite(old).all() or not np.isfinite(new).all():raise ValueError('Nonfinite fit input')
        self.reference_rows=len(old)+len(new);self.input_features=old.shape[1]
        if self.method in ['centroid_original','threshold_updated']:
            if original is None:raise ValueError('Original CEN required')
            self.cen=original;self.threshold=float(original_threshold);return self
        if self.method=='reference_balanced':
            self.cen=CentroidBasedOneClassClassifier();self.cen.fit(np.concatenate([old,new]));return self
        if self.method=='centroid_two':
            self.components=[]
            for x in [old,new]:
                cen=CentroidBasedOneClassClassifier();cen.fit(x);q=max(float(np.quantile(cen.get_density(x),.95)),1e-8);self.components.append((cen,q))
            return self
        x=self._log(np.concatenate([old,new]));self.scaler=StandardScaler().fit(x)
        # 두 정상 모드 모두로 적합. 거의 상수인 차원의 수치 폭주를 막는 고정 하한.
        self.scaler.scale_=np.maximum(self.scaler.scale_,1e-6);z=self.scaler.transform(x)
        if self.method=='latent_lof20':
            self.model=LocalOutlierFactor(n_neighbors=min(20,len(z)-1),novelty=True,n_jobs=1);self.model.fit(z)
        elif self.method=='latent_mahalanobis':self.model=LedoitWolf().fit(z)
        else:self.model=NearestNeighbors(n_neighbors=min(5,len(z)),algorithm='brute',n_jobs=1).fit(z)
        return self
    def score(self,x):
        x=np.asarray(x,dtype=np.float64)
        if self.method in ['centroid_original','threshold_updated','reference_balanced']:score=self.cen.get_density(x)
        elif self.method=='centroid_two':score=np.min(np.stack([cen.get_density(x)/q for cen,q in self.components]),axis=0)
        else:
            z=self.scaler.transform(self._log(x))
            if self.method=='latent_lof20':score=-self.model.score_samples(z)
            elif self.method=='latent_mahalanobis':score=self.model.mahalanobis(z)
            else:score=self.model.kneighbors(z,return_distance=True)[0].mean(axis=1)
        if not np.isfinite(score).all():raise ValueError('Nonfinite anomaly score')
        return np.asarray(score,dtype=np.float64)
    def calibrate(self,old_normal,new_normal):
        if self.method!='centroid_original':self.threshold=updated_threshold(self.score(old_normal),self.score(new_normal))
        return self
    def predict(self,x):return (self.score(x)>self.threshold).astype(int)
