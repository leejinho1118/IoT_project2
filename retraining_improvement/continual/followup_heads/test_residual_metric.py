import unittest,pickle
import numpy as np
from metric_models import NormalHead
class ResidualMetricTests(unittest.TestCase):
 def test_block_fit_serialization_and_calibration_isolation(self):
  rng=np.random.default_rng(5678);a=rng.normal(size=(32,126));b=rng.normal(2,1,(32,126));q=rng.normal(size=(10,126))
  h=NormalHead('residual_l1').fit(a,b);mean=h.scaler.mean_.copy();h.calibrate(a+.01,b+.01);np.testing.assert_array_equal(mean,h.scaler.mean_);np.testing.assert_allclose(h.score(q),pickle.loads(pickle.dumps(h)).score(q))
 def test_latent_block_participates(self):
  rng=np.random.default_rng(5678);a=rng.normal(size=(32,126));b=rng.normal(2,1,(32,126));q=rng.normal(size=(10,126));h=NormalHead('residual_l1').fit(a,b);s=h.score(q);q[:,115:]+=100;self.assertGreater(h.score(q).mean(),s.mean())
if __name__=='__main__':unittest.main()
