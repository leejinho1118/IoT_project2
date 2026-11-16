import unittest
from stream_memory import PriorityNormalMemory
class StreamMemoryTests(unittest.TestCase):
 def test_fixed_cap_and_order_invariance(self):
  items=[dict(id=str(i)) for i in range(100)];a=PriorityNormalMemory(12);b=PriorityNormalMemory(12)
  a.update(items,lambda _: 'normal');b.update(list(reversed(items)),lambda _: 'normal')
  self.assertEqual([r['id'] for r in a.items()],[r['id'] for r in b.items()]);self.assertEqual(len(a.items()),12)
 def test_unknown_and_attacks_are_not_promoted(self):
  a=PriorityNormalMemory(12);a.update([dict(id='n'),dict(id='a'),dict(id='u')],{'n':'normal','a':'attack','u':'uncertain'}.__getitem__);self.assertEqual([r['id'] for r in a.items()],['n'])
 def test_duplicate_id_does_not_consume_memory(self):
  a=PriorityNormalMemory(12);a.update([dict(id='n'),dict(id='n')],lambda _:'normal');self.assertEqual(len(a.items()),1)
if __name__=='__main__':unittest.main()
