"""Declared approval perturbations; never mutates ground-truth source rows."""
import numpy as np

def approval_view(items,setting,seed,stage,cid):
    rng=np.random.default_rng(seed+stage*10007+cid*97+20261121)
    if setting=='query25':
        n=max(1,int(np.ceil(.25*len(items))))
        chosen=set(map(int,rng.choice(len(items),n,replace=False)))
    elif setting=='poison05':chosen=set(range(len(items)))
    else:raise ValueError(setting)
    rows=[dict(r,actual_y=int(r['y']),approval_corrupted=False) for i,r in enumerate(items) if i in chosen]
    if setting=='poison05':
        normal=sum(r['y']==0 for r in rows);attacks=[i for i,r in enumerate(rows) if r['y']==1]
        count=min(len(attacks),int(round(normal*.05/.95)))
        for i in rng.choice(attacks,count,replace=False):rows[int(i)]['y']=0;rows[int(i)]['approval_corrupted']=True
    audit=dict(setting=setting,all_candidate_rows=len(items),queried_rows=len(rows),query_ids=[r['id'] for r in rows],unqueried_ids=[r['id'] for i,r in enumerate(items) if i not in chosen],approved_rows=sum(r['y']==0 for r in rows),false_normal_approvals=sum(r['approval_corrupted'] for r in rows),protected_clean_materials='Initial anchors/current validation, calibration and development labels remain clean; query fraction is candidate-only, not total label budget')
    return rows,audit
