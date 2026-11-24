import json, os, numpy as np
from scipy import stats
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'runs')
CELLS={'D0':('mse','D0_mse_par'),'D1':('mse_m','D1_msem_par')}
def load(k):
    v,c=CELLS[k]; out={}
    for r in range(10,20):
        f=f'{D}/{v}/{c}/Run_{r}.jsonl'
        if os.path.exists(f):
            L=[list(json.loads(l).values())[0] for l in open(f)]
            if len(L)==40: out[r]=L
    return out
A,B=load('D0'),load('D1'); ks=sorted(set(A)&set(B))
auc=lambda L:np.array([np.mean(x['global_auc'])*100 for x in L])
def reach(a,t=97.0):
    for i in range(len(a)):
        if (a[i:]>=t).all(): return i+1
    return None
rows={k:{r:auc(X[r]) for r in X} for k,X in (('D0',A),('D1',B))}
out=['# 직접 대조 결과 (자동 생성; CRITERIA_direct.md)','',f'완료 짝 런 n={len(ks)}']
if len(ks)>=2:
    d=np.array([rows['D1'][r][30:40].mean()-rows['D0'][r][30:40].mean() for r in ks]); n=len(d)
    h=stats.t.ppf(.975,n-1)*d.std(ddof=1)/np.sqrt(n); m=d.mean()
    out+=[f'주 지표 R31~40 평균 AUC 짝차 D1−D0: {m:+.2f}%p, 95% [{m-h:+.2f}, {m+h:+.2f}] → '+('통과(하한>+0.3)' if m-h>0.3 else '불통과')]
    for nm,sl in (('R20',19),('R40',39)):
        dd=np.array([rows['D1'][r][sl]-rows['D0'][r][sl] for r in ks]); hh=stats.t.ppf(.975,n-1)*dd.std(ddof=1)/np.sqrt(n)
        out.append(f'{nm} 짝차: {dd.mean():+.2f}%p [{dd.mean()-hh:+.2f}, {dd.mean()+hh:+.2f}]')
    out+=['','|셀|R31~40 평균|R20|R40|발산 런|0.97 유지 도달 라운드(중앙값, 미도달 수)|학습 스텝 총합 평균|','|---|---|---|---|---|---|---|']
    for k,X in (('D0',A),('D1',B)):
        rr=[reach(rows[k][r]) for r in ks]; ok=[x for x in rr if x]
        steps=np.mean([sum(sum(x['diag2']['steps']) for x in X[r]) for r in ks])
        dv=sum(1 for r in ks if (rows[k][r][4:]<90).any())
        out.append(f"|{k}|{np.mean([rows[k][r][30:40].mean() for r in ks]):.2f}|{np.mean([rows[k][r][19] for r in ks]):.2f}|{np.mean([rows[k][r][39] for r in ks]):.2f}|{dv}|{np.median(ok) if ok else '-'} ({len(rr)-len(ok)})|{steps:.0f}|")
    same=all(a['join_clients']==b['join_clients'] for r in ks for a,b in zip(A[r],B[r])); out.append(f'\n참여 순서 동일 검증: {same}')
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'RESULTS_direct.md'),'w').write('\n'.join(out)+'\n'); print('\n'.join(out))
