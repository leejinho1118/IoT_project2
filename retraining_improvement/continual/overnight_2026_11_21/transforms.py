"""Initial-normal-only feature transforms; never refit on evaluation or arrivals."""
import numpy as np
from sklearn.preprocessing import StandardScaler

def feature_map(x,mode):
    x=np.asarray(x,dtype=np.float64)
    if mode=='standard':return x
    if mode=='signed_log':return np.sign(x)*np.log1p(np.abs(x))
    raise ValueError('Unknown transform mode')

class InputTransform:
    def __init__(self,mode='standard',clip=20.):self.mode=mode;self.clip=float(clip);self.scaler=StandardScaler()
    def fit(self,x):
        x=feature_map(x,self.mode)
        if not np.isfinite(x).all():raise ValueError('Nonfinite initial normal input')
        self.scaler.fit(x);return self
    def transform(self,x):
        x=feature_map(x,self.mode)
        z=self.scaler.transform(x)
        if not np.isfinite(z).all():raise ValueError('Nonfinite transformed feature')
        return np.clip(z,-self.clip,self.clip)
    def saturation(self,x):
        z=self.scaler.transform(feature_map(x,self.mode));return float(np.mean(np.abs(z)>self.clip))
