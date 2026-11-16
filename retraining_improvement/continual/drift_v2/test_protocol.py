"""분포 변화 연구의 표본 보류·리플레이 안전 조건 검증."""
import unittest
import numpy as np
from scenario import signed_log,verified_rows,replay_mix,uncertainty_verdict,BoundedStandardScaler
class DriftProtocolTests(unittest.TestCase):
 def test_signed_log_finite_and_reversible(self):
  x=np.array([-1e200,-3.,0.,3.,1e200]);z=signed_log(x)
  self.assertTrue(np.isfinite(z).all());np.testing.assert_allclose(np.sign(z)*np.expm1(np.abs(z)),x)
 def test_verification_rejects_actual_attacks(self):
  rows=[dict(id=f'x{i}',y=i%2) for i in range(200)]
  normal,attack,uncertain=verified_rows(rows,.2)
  self.assertTrue(all(r['y']==0 for r in normal));self.assertTrue(all(r['y']==1 for r in attack));self.assertTrue(uncertain)
  self.assertEqual(len(normal)+len(attack)+len(uncertain),len(rows))
 def test_uncertainty_does_not_depend_on_detection_score(self):
  row=dict(id='candidate3',y=0,score=1000)
  a=uncertainty_verdict(row,.2);row['score']=-1000
  self.assertEqual(a,uncertainty_verdict(row,.2))
 def test_zero_capacity_preserves_all_new_rows(self):
  new=[dict(id=f'n{i}') for i in range(40)];bank=[dict(id=f'o{i}') for i in range(128)]
  mixed,count=replay_mix(new,bank,0,7);self.assertEqual(new,mixed);self.assertEqual(count,0)
 def test_capacity_never_changes_training_sample_budget(self):
  new=[dict(id=f'n{i}') for i in range(200)];bank=[dict(id=f'o{i}') for i in range(128)]
  for cap in [32,128]:
   mixed,count=replay_mix(new,bank,cap,7);self.assertEqual(len(mixed),200);self.assertEqual(count,min(50,cap));self.assertEqual(len({r['id'] for r in mixed}),200)
   self.assertTrue({r['id'] for r in mixed if r['id'].startswith('o')}<= {r['id'] for r in bank[:cap]})
 def test_same_seed_reproduces_exact_update_input(self):
  new=[dict(id=f'n{i}') for i in range(40)];bank=[dict(id=f'o{i}') for i in range(128)]
  self.assertEqual(replay_mix(new,bank,32,13),replay_mix(new,bank,32,13))
 def test_shifted_near_constant_features_cannot_overflow_model_input(self):
  initial=np.array([[0.,1e-30],[0.,2e-30]])
  scaler=BoundedStandardScaler().fit(initial)
  shifted=scaler.transform(np.array([[1e100,1e100]]))
  self.assertTrue(np.isfinite(shifted).all());self.assertTrue((np.abs(shifted)<=20).all())
if __name__=='__main__':unittest.main()
