import json, os, numpy as np
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__)); D=HERE+'/runs'
def load(v,c,runs):
    o={}
    for r in runs:
        f=f'{D}/{v}/{c}/Run_{r}.jsonl'
        if os.path.exists(f):
            L=[list(json.loads(l).values())[0] for l in open(f)]
            if len(L)==40: o[r]=L
    return o
auc=lambda L:np.array([np.mean(x['global_auc'])*100 for x in L])
def reach(a,t=97.0):
    for i in range(len(a)):
        if (a[i:]>=t).all(): return i+1
def ci(d):
    n=len(d); m=d.mean(); h=stats.t.ppf(.975,n-1)*d.std(ddof=1)/np.sqrt(n); return m,m-h,m+h
out=['# 다른 데이터셋 확장 결과 (자동 생성; CRITERIA_dataset.md)','',
     '**CIC 결과는 인위적 비IID(KMeans 군집 분할, 기기 식별자 없음·자연 비IID 아님) 한계가 있음.**','',
     '|데이터셋|data_seed|n|D0 R31~40|D1 R31~40|짝차(%p)|95% 구간|판정|R20 짝차|R40 짝차|발산 D0/D1|0.97유지 중앙값(미도달) D0/D1|스텝 D0/D1|순서동일|','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
res={}
for ds,seeds,runs,judge in (('ton',(1234,2024),range(10,20),1),('cic_km',(1234,2024),range(10,20),1),('cic_iid',(1234,),range(10,15),0)):
    for s in seeds:
        A=load('mse',f'D0_{ds}_s{s}',runs); B=load('mse_m',f'D1_{ds}_s{s}',runs); ks=sorted(set(A)&set(B))
        if len(ks)<2: out.append(f'|{ds}|{s}|{len(ks)}|-|-|-|-|-|-|-|-|-|-|-|'); res.setdefault(ds,[]).append(None); continue
        a={r:auc(A[r]) for r in ks}; b={r:auc(B[r]) for r in ks}
        m,lo,hi=ci(np.array([b[r][30:].mean()-a[r][30:].mean() for r in ks]))
        d20=ci(np.array([b[r][19]-a[r][19] for r in ks]))[0]; d40=ci(np.array([b[r][39]-a[r][39] for r in ks]))[0]
        dv=lambda X:sum(1 for r in ks if (X[r][4:]<90).any())
        def rc(X):
            rr=[reach(X[r]) for r in ks]; ok=[x for x in rr if x]; return f"{np.median(ok) if ok else '-'}({len(rr)-len(ok)})"
        st=lambda X:np.mean([sum(sum(x['diag2']['steps']) for x in X[r]) for r in ks])
        same=all(p['join_clients']==q['join_clients'] for r in ks for p,q in zip(A[r],B[r]))
        ok=lo>0.3; res.setdefault(ds,[]).append((ok,len(ks)))
        out.append(f"|{ds}{'(참고)' if not judge else ''}|{s}|{len(ks)}|{np.mean([a[r][30:].mean() for r in ks]):.2f}|{np.mean([b[r][30:].mean() for r in ks]):.2f}|{m:+.2f}|[{lo:+.2f}, {hi:+.2f}]|{'통과' if ok else '불통과'}|{d20:+.2f}|{d40:+.2f}|{dv(a)}/{dv(b)}|{rc(a)}/{rc(b)}|{st(A):.0f}/{st(B):.0f}|{same}|")
out.append('')
for ds in ('ton','cic_km'):
    r=res.get(ds,[])
    if len(r)==2 and all(x and x[1]==10 for x in r):
        k=sum(x[0] for x in r); out.append(f"{ds} 데이터셋 판정: "+('**통과**(두 data_seed 모두)' if k==2 else '**부분**(하나만 통과)' if k==1 else '**불통과**'))
    else: out.append(f'{ds} 데이터셋 판정: 미완')
open(HERE+'/RESULTS_dataset.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
