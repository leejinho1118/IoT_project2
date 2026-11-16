import unittest,pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from head_models import NormalHead,updated_threshold
from Model.Centroid import CentroidBasedOneClassClassifier
class HeadTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(123);self.old=rng.normal(-5,.2,(100,2));self.new=rng.normal(5,.2,(100,2));self.cal_old=rng.normal(-5,.2,(40,2));self.cal_new=rng.normal(5,.2,(40,2))
 def test_threshold_change_cannot_change_auc(self):
  cen=CentroidBasedOneClassClassifier();cen.fit(self.old)
  a=NormalHead('centroid_original').fit(self.old,self.new,cen,1)
  b=NormalHead('threshold_updated').fit(self.old,self.new,cen,1).calibrate(self.cal_old,self.cal_new)
  q=np.vstack([self.cal_old,self.cal_new,np.zeros((20,2))]);y=np.r_[np.zeros(80),np.ones(20)]
  np.testing.assert_array_equal(a.score(q),b.score(q));self.assertEqual(roc_auc_score(y,a.score(q)),roc_auc_score(y,b.score(q)))
 def test_two_components_do_not_treat_gap_as_normal_center(self):
  h=NormalHead('centroid_two').fit(self.old,self.new)
  self.assertGreater(h.score(np.zeros((1,2)))[0],max(h.score(np.array([[-5,-5],[5,5]]))))
 def test_calibration_is_disjoint_from_reference_fit(self):
  h=NormalHead('latent_knn5').fit(self.old,self.new);mean=h.scaler.mean_.copy();h.calibrate(self.cal_old*100,self.cal_new*100)
  np.testing.assert_array_equal(mean,h.scaler.mean_)
 def test_group_calibration_keeps_empirical_group_tail_small(self):
  old=np.arange(100);new=np.arange(100)+100;t=updated_threshold(old,new)
  self.assertLessEqual(np.mean(old>t),.05);self.assertLessEqual(np.mean(new>t),.05)
 def test_lof_score_orientation_and_new_query_semantics(self):
  h=NormalHead('latent_lof20').fit(self.old,self.new)
  self.assertGreater(h.score(np.array([[100,100]]))[0],h.score(np.array([[-5,-5]]))[0])
 def test_all_new_heads_serialize_and_score_without_domain_label(self):
  for name in ['reference_balanced','centroid_two','latent_knn5','latent_lof20','latent_mahalanobis','input_knn5','log_input_knn5']:
   h=NormalHead(name).fit(self.old,self.new).calibrate(self.cal_old,self.cal_new);q=np.array([[-5,-5],[5,5],[0,0]])
   np.testing.assert_allclose(h.score(q),pickle.loads(pickle.dumps(h)).score(q));self.assertTrue(np.isfinite(h.score(q)).all())
 def test_nonfinite_fit_is_rejected(self):
  with self.assertRaises(ValueError):NormalHead('latent_knn5').fit([[np.nan,1],[2,3]],self.new)
if __name__=='__main__':unittest.main()
