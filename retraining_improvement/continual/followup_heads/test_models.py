import unittest,pickle
import numpy as np
from models import NormalHead,NEW
class Tests(unittest.TestCase):
 def test_all_finite_serializable(self):
  rng=np.random.default_rng(5678);a=rng.normal(size=(32,115));b=rng.normal(2,1,(32,115));q=rng.normal(size=(9,115))
  for m in NEW:
   h=NormalHead(m).fit(a,b).calibrate(a+.01,b+.01);s=h.score(q);self.assertTrue(np.isfinite(s).all());np.testing.assert_allclose(s,pickle.loads(pickle.dumps(h)).score(q))
 def test_extreme_outlier_orientation(self):
  rng=np.random.default_rng(1);a=rng.normal(size=(32,115));b=rng.normal(2,1,(32,115))
  for m in NEW:
   h=NormalHead(m).fit(a,b);self.assertGreater(h.score(np.full((1,115),100.))[0],np.median(h.score(a+.01)))
 def test_calibration_does_not_refit(self):
  rng=np.random.default_rng(1);a=rng.normal(size=(32,115));b=rng.normal(2,1,(32,115))
  for m in NEW:
   h=NormalHead(m).fit(a,b);mean=h.scaler.mean_.copy();h.calibrate(a,b);np.testing.assert_array_equal(mean,h.scaler.mean_)
if __name__=='__main__':unittest.main()
