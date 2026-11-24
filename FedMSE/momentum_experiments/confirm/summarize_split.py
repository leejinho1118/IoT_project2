import json, os, numpy as np
from scipy import stats
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'runs')
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
out=['# 새 분할 반복 결과 (자동 생성; CRITERIA_split.md)','']; passes={}
for lr,runs,judge in (('1e-5',range(10,20),True),('5e-5',range(10,15),False)):
    out.append(f"## lr={lr} ({'판정' if judge else '탐색, 판정 아님'})")
    out+=['|data_seed|n|D0 R31~40|D1 R31~40|짝차(%p)|95% 구간|R20 짝차|R40 짝차|발산 D0/D1|0.97 유지 중앙값(미도달) D0/D1|스텝 D0/D1|순서동일|','|---|---|---|---|---|---|---|---|---|---|---|---|']
    for s in (2024,4321,7777):
        A=load('mse',f'D0_s{s}_lr{lr}',runs); B=load('mse_m',f'D1_s{s}_lr{lr}',runs); ks=sorted(set(A)&set(B))
        if len(ks)<2: out.append(f'|{s}|{len(ks)}|-|-|-|-|-|-|-|-|-|-|'); continue
        a={r:auc(A[r]) for r in ks}; b={r:auc(B[r]) for r in ks}
        d=np.array([b[r][30:].mean()-a[r][30:].mean() for r in ks]); m,lo,hi=ci(d)
        d20=ci(np.array([b[r][19]-a[r][19] for r in ks])); d40=ci(np.array([b[r][39]-a[r][39] for r in ks]))
        dv=lambda X:sum(1 for r in ks if (X[r][4:]<90).any())
        def rc(X):
            rr=[reach(X[r]) for r in ks]; ok=[x for x in rr if x]; return f"{np.median(ok) if ok else '-'}({len(rr)-len(ok)})"
        st=lambda X:np.mean([sum(sum(x['diag2']['steps']) for x in X[r]) for r in ks])
        same=all(p['join_clients']==q['join_clients'] for r in ks for p,q in zip(A[r],B[r]))
        out.append(f"|{s}|{len(ks)}|{np.mean([a[r][30:].mean() for r in ks]):.2f}|{np.mean([b[r][30:].mean() for r in ks]):.2f}|{m:+.2f}|[{lo:+.2f}, {hi:+.2f}]|{d20[0]:+.2f}|{d40[0]:+.2f}|{dv(a)}/{dv(b)}|{rc(a)}/{rc(b)}|{st(A):.0f}/{st(B):.0f}|{same}|")
        if judge: passes[s]=(lo,len(ks))
    out.append('')
if passes:
    ok=len(passes)==3 and all(n==10 and lo>0.3 for lo,n in passes.values())
    out.append('통과 조건(lr1e-5, 세 분할 모두 하한>+0.3%p): '+('**통과**' if ok else '**불통과/미완** — '+', '.join(f'{s}: 하한 {lo:+.2f} (n={n})' for s,(lo,n) in passes.items())))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'RESULTS_split.md'),'w').write('\n'.join(out)+'\n'); print('\n'.join(out))
