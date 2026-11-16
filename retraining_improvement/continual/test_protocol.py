import unittest
import numpy as np
from protocol import fingerprint,verify_candidates,fixed_budget_mix,check_partition
class ProtocolTests(unittest.TestCase):
 def test_verification_is_not_detection(self):
  items=[{'id':'false_positive','anomaly_flag':True},{'id':'attack','anomaly_flag':False},{'id':'unknown','anomaly_flag':True}]
  a,b,c=verify_candidates(items,{'false_positive':'normal','attack':'attack','unknown':'uncertain'}.__getitem__)
  self.assertEqual([x['id'] for x in a],['false_positive']); self.assertEqual(len(b),1); self.assertEqual(len(c),1)
 def test_split_blocks_duplicate_features_even_with_different_ids(self):
  fp=fingerprint(np.arange(115)); a={'id':'a','fingerprint':fp}; b={'id':'b','fingerprint':fp}
  with self.assertRaises(ValueError): check_partition({'train':[a],'future':[b]})
 def test_negative_zero_does_not_bypass_deduplication(self):
  self.assertEqual(fingerprint(np.zeros(115)),fingerprint(-np.zeros(115)))
 def test_replay_keeps_optimizer_sample_budget_fixed(self):
  new=[{'id':f'n{i}'} for i in range(20)]; old=[{'id':f'o{i}'} for i in range(3)]
  mixed,count=fixed_budget_mix(new,old,.5,np.random.default_rng(3))
  self.assertEqual(len(mixed),20); self.assertEqual(count,3); self.assertEqual(len({x['id'] for x in mixed}),20)
 def test_empty_normal_verification_cannot_train(self):
  with self.assertRaises(ValueError): fixed_budget_mix([],[],.25,np.random.default_rng(1))
 def test_nonfinite_features_rejected(self):
  with self.assertRaises(ValueError): fingerprint([float('nan')]*115)
if __name__=='__main__': unittest.main()
