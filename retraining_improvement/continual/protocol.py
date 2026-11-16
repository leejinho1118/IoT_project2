"""Leakage checks, verifier routing and fixed-size replay sampling."""
import hashlib
import numpy as np

def fingerprint(row):
    a=np.asarray(row,dtype='<f8').copy()
    if not np.isfinite(a).all(): raise ValueError('Nonfinite feature')
    a[a==0]=0  # Canonicalize negative zero.
    return hashlib.sha256(a.tobytes()).hexdigest()

def verify_candidates(items,verifier):
    accepted=[]; attacks=[]; uncertain=[]
    for item in items:
        verdict=verifier(item['id'])
        if verdict=='normal': accepted.append(item)
        elif verdict=='attack': attacks.append(item)
        elif verdict=='uncertain': uncertain.append(item)
        else: raise ValueError('Invalid verifier response')
    return accepted,attacks,uncertain

def fixed_budget_mix(new_items,replay_items,fraction,rng):
    n=len(new_items)
    if not n: raise ValueError('No verified new normal data')
    count=min(int(n*fraction),len(replay_items),n-1)
    old_idx=rng.choice(len(replay_items),size=count,replace=False) if count else []
    new_idx=rng.choice(n,size=n-count,replace=False)
    result=[new_items[int(i)] for i in new_idx]+[replay_items[int(i)] for i in old_idx]
    rng.shuffle(result)
    assert len(result)==n
    return result,count

def check_partition(groups):
    seen_ids=set(); seen_fp=set(); counts={}
    for name,rows in groups.items():
        ids={x['id'] for x in rows}; fps={x['fingerprint'] for x in rows}
        if len(ids)!=len(rows) or len(fps)!=len(rows): raise ValueError('Within-partition duplicates: '+name)
        if ids&seen_ids or fps&seen_fp: raise ValueError('Cross-partition leakage: '+name)
        seen_ids.update(ids); seen_fp.update(fps); counts[name]=len(rows)
    return counts
