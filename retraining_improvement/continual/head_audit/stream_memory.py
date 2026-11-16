"""검증 정상만 받는 고정 용량 우선순위 reservoir. 새로 들어온 순서에 의존하지 않는다."""
import hashlib
class PriorityNormalMemory:
 def __init__(self,capacity=128,salt='fedmse-normal-5678'):
  if capacity<1:raise ValueError('Positive capacity required')
  self.capacity=capacity;self.salt=salt;self.rows={};self.seen=0
 def priority(self,rid):return hashlib.sha256((self.salt+'|'+rid).encode()).hexdigest()
 def update(self,items,verifier):
  accepted=held=rejected=0
  for row in items:
   verdict=verifier(row['id'])
   if verdict=='uncertain':held+=1;continue
   if verdict=='attack':rejected+=1;continue
   if verdict!='normal':raise ValueError('Unknown verification result')
   accepted+=1;self.seen+=1;self.rows[row['id']]=row
   if len(self.rows)>self.capacity:del self.rows[max(self.rows,key=lambda rid:(self.priority(rid),rid))]
  return dict(accepted=accepted,held=held,rejected=rejected,stored=len(self.rows))
 def items(self,limit=None):
  keys=sorted(self.rows,key=lambda rid:(self.priority(rid),rid));return [self.rows[k] for k in keys[:limit]]
