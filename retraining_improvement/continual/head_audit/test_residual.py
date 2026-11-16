import unittest,pickle
import numpy as np
from residual_models import NormalHead
class ResidualTests(unittest.TestCase):
 def test_uses_input_and_latent_blocks_without_query_domain(self):
  rng=np.random.default_rng(4);old=rng.normal(0,1,(40,126));new=rng.normal(2,1,(40,126));h=NormalHead('residual_knn_w025').fit(old,new)
  q=rng.normal(1,1,(10,126));score=h.score(q);self.assertEqual(len(score),10);self.assertTrue(np.isfinite(score).all())
  changed=q.copy();changed[:,115:]+=100;self.assertGreater(h.score(changed).mean(),score.mean())
 def test_calibration_does_not_refit_any_normalization(self):
  rng=np.random.default_rng(4);old=rng.normal(0,1,(40,126));new=rng.normal(2,1,(40,126));h=NormalHead('residual_knn_w050').fit(old,new);m=h.input_scaler.mean_.copy();h.calibrate(old+.01,new+.01)
  np.testing.assert_array_equal(m,h.input_scaler.mean_)
 def test_saved_residual_head_reproduces_scores(self):
  rng=np.random.default_rng(4);old=rng.normal(0,1,(40,126));new=rng.normal(2,1,(40,126));h=NormalHead('residual_knn_w100').fit(old,new);q=rng.normal(1,1,(10,126));np.testing.assert_allclose(h.score(q),pickle.loads(pickle.dumps(h)).score(q))
if __name__=='__main__':unittest.main()
