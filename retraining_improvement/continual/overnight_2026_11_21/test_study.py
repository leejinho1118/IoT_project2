"""CPU unit checks for schema handling, transforms, release boundaries and aggregation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import unittest,copy
from pathlib import Path
import numpy as np
import torch
from transforms import InputTransform,feature_map
import federated_core as C
import study_data

class StudyTests(unittest.TestCase):
 def test_log_is_odd_and_finite(self):
  x=np.array([[-1e30,-1.,0.,1.,1e30]]);mapped=feature_map(x,'signed_log');self.assertTrue(np.isfinite(mapped).all());self.assertTrue(np.allclose(mapped,-mapped[:,::-1]))
 def test_transform_does_not_refit(self):
  m=InputTransform('signed_log').fit(np.array([[1.,2.],[3.,4.]]));mean=m.scaler.mean_.copy();m.transform(np.array([[1e100,-1e100]]));self.assertTrue(np.array_equal(mean,m.scaler.mean_))
 def test_clipping_bound(self):
  m=InputTransform().fit(np.array([[0.,0.],[1.,1.]]));self.assertLessEqual(np.abs(m.transform([[1e50,-1e50]])).max(),20.)
 def test_schema_81_and_115(self):
  for dim in [81,115]:
   m=C.StudySAE(input_dim=dim,output_dim=dim,hidden_neus=50,latent_dim=11);z,r,l=m(torch.zeros(4,dim));self.assertEqual(tuple(z.shape),(4,11));self.assertEqual(tuple(r.shape),(4,dim));self.assertTrue(torch.isfinite(l))
 def test_norm_matches_upstream(self):
  torch.manual_seed(1);base=C.B.Shrink_Autoencoder(input_dim=5,output_dim=5,hidden_neus=4,latent_dim=2);m=C.StudySAE(input_dim=5,output_dim=5,hidden_neus=4,latent_dim=2);m.load_state_dict(base.state_dict());x=torch.randn(3,5);self.assertTrue(torch.equal(m(x)[2],base(x)[2]))
 def test_squared_loss_formula(self):
  m=C.StudySAE(input_dim=5,output_dim=5,hidden_neus=4,latent_dim=2,shrink_kind='squared');x=torch.ones(3,5);z,r,l=m(x);self.assertTrue(torch.allclose(l,torch.mean((x-r)**2)+10*torch.sum(z*z,dim=1).mean()))
 def test_mseavg_formula_unchanged(self):
  class Toy(torch.nn.Module):
   def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(1.))
   def forward(self,x):
    r=torch.zeros_like(x)+self.w;return x,r,torch.mean((x-r)**2)
  C.configure(dict(mode='standard',input_dim=1),'cpu')
  results=[]
  for cls in [C.ORIGINAL_AGGREGATOR,C.FeatureMapAggregator]:
   agg=cls(Toy(),update_type='mse_avg');agg.dev_dataset=np.ones((2,1));agg.update([({'w':torch.tensor(2.)},2,1),({'w':torch.tensor(4.)},2,1)]);results.append(float(agg.model.w.detach()))
  self.assertAlmostEqual(results[0],2.2,places=6);self.assertEqual(results[0],results[1])
 def test_future_features_lazy(self):
  root=Path(__file__).resolve().parents[2];data=root/'continual/overnight_2026_11_21/raw_blocks_v2';real=np.load;reads=[]
  class Tracked:
   def __init__(self,obj):self.obj=obj
   def __enter__(self):return self
   def __exit__(self,*args):self.obj.close()
   def __getitem__(self,key):reads.append(key);return self.obj[key]
  try:
   np.load=lambda *a,**k:Tracked(real(*a,**k))
   clients,_=study_data.load_development(data);self.assertFalse(any('candidate' in s or 'new_valid' in s or 'calib_new' in s or 'decision_new' in s for s in reads));reads.clear()
   study_data.activate(clients,1);self.assertTrue(reads);self.assertTrue(all(s.endswith('_0') for s in reads));self.assertTrue(all(c['candidate_stages'][1] is None and c['candidate_stages'][2] is None for c in clients))
  finally:np.load=real

if __name__=='__main__':unittest.main()
