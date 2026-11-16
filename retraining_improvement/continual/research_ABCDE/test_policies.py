import unittest,copy
import numpy as np
from policies import mix,memory_bank,adaptive_ratio,gate_accept,development_score,choose_best
from data import split_validation
from scenario import BoundedStandardScaler
class CampaignTests(unittest.TestCase):
 def test_validation_roles_are_disjoint(self):
  a,b,c=split_validation(list(range(30)));self.assertFalse(set(a)&set(b));self.assertFalse(set(a)&set(c));self.assertFalse(set(b)&set(c));self.assertEqual(len(a)+len(b)+len(c),30)
 def test_ratio_changes_old_count_not_training_budget(self):
  new=[dict(id=f'n{i}',y=0) for i in range(80)];bank=[dict(id=f'o{i}',y=0) for i in range(50)]
  for ratio,count in [(.25,20),(.5,40),(.75,50)]:
   rows,k=mix(new,bank,ratio,123);self.assertEqual(len(rows),80);self.assertEqual(k,count)
 def test_attack_cannot_enter_normal_update(self):
  with self.assertRaises(ValueError):mix([dict(id='attack',y=1)]*4,[dict(id='old',y=0)],.25,1)
 def test_diverse_bank_covers_both_old_modes(self):
  rows=[dict(id=str(i),x=np.repeat(float(i//10),115),y=0) for i in range(30)];scaler=BoundedStandardScaler().fit(np.array([r['x'] for r in rows]));selected=memory_bank(rows,scaler,3,'diverse');self.assertEqual({r['x'][0] for r in selected},{0.,1.,2.})
 def test_adaptive_ratio_uses_past_reconstruction(self):
  self.assertEqual(adaptive_ratio(.25,dict(old=1,new=10),dict(old=2,new=5)),.5)
  self.assertEqual(adaptive_ratio(.25,dict(old=1,new=10),dict(old=1,new=9)),.125)
  self.assertEqual(adaptive_ratio(.25,dict(old=1,new=10),None),.25)
 def test_gate_rejects_forgetting_even_when_new_auc_rises(self):
  base={d:dict(mean_client_auc=.9,fpr=.05) for d in ['old','new']};candidate=copy.deepcopy(base);candidate['old']['mean_client_auc']=.85;candidate['new']['mean_client_auc']=1.
  self.assertFalse(gate_accept(candidate,base,base)[0])
 def test_gate_accepts_development_improvement(self):
  base={d:dict(mean_client_auc=.8,fpr=.05) for d in ['old','new']};candidate=copy.deepcopy(base);candidate['new']['mean_client_auc']=.9
  self.assertTrue(gate_accept(candidate,base,base)[0])
 def test_selection_preserves_negative_experiment(self):
  analyses=[dict(round='A',development_score_mean=.8,training_seconds_mean=10),dict(round='B',development_score_mean=.7,training_seconds_mean=8)]
  self.assertEqual(choose_best(analyses),'A');self.assertEqual(len(analyses),2)
if __name__=='__main__':unittest.main()
