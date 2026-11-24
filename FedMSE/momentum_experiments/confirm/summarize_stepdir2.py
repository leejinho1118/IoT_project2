import json,os,numpy as np
R=os.path.dirname(os.path.abspath(__file__))+'/runs'
def ld(v,c):
    L=[list(json.loads(l).values())[0] for l in open(f'{R}/{v}/{c}/Run_10.jsonl')]
    return L
auc=lambda L:np.array([np.mean(x['global_auc'])*100 for x in L])
out=['# 걸음 맞춤 대조 + 방향 로깅 (run 10, n=1, 탐색용; CRITERIA_stepdir2.md)','']
for ds,name in (('ton','ToN'),('cic_km','CIC-km (인위적 비IID)')):
    old={'D0L':('mse',f'D0_{ds}_s1234'),'D1L':('mse_m',f'D1_{ds}_s1234'),'D2L':('mse_dm',f'D2_{ds}_s1234')}
    new={'D0L':('mse',f'D0L_{ds}_s1234'),'D1L':('mse_m',f'D1L_{ds}_s1234'),'D2L':('mse_dm',f'D2L_{ds}_s1234'),'D0M':('mse',f'D0M_{ds}_s1234')}
    out+= [f'## {name}','']
    cur={k:ld(*v) for k,v in new.items()}
    a={k:auc(v) for k,v in cur.items()}
    out.append('재현 점검(로깅 런 vs 기존 run10 최대 AUC 차, 기준 1e-6): '+', '.join(f"{k} {np.abs(a[k]-auc(ld(*old[k]))).max():.2g}" for k in old))
    out.append('');out.append('|방법|AUC R1~5|R6~15|R31~40|낙하 수(R5이후)|이동 중앙값|pair_cos 중앙|cos_mean 중앙|cos_prev 중앙(R2~40)|')
    out.append('|---|---|---|---|---|---|---|---|---|')
    for k in ('D0L','D0M','D1L','D2L'):
        L=cur[k];A=a[k];dr=((A[5:]-A[4:-1])<=-2).sum()
        gd=np.median([x['diag2']['global_disp'] for x in L]);d=[x['diag2'] for x in L]
        pc=np.median([x['pair_cos'] for x in d]);cm=np.median([x['cos_mean'] for x in d]);cp=np.median([x['cos_prev'] for x in d[1:] if x['cos_prev'] is not None])
        out.append(f"|{k}|{A[:5].mean():.1f}|{A[5:15].mean():.1f}|{A[30:].mean():.1f}|{dr}|{gd:.3f}|{pc:.2f}|{cm:.2f}|{cp:.2f}|")
    f=lambda k:a[k][30:].mean()
    L1=f('D0L')-f('D1L');LM=f('D0L')-f('D0M')
    pat='판정 불가(L1≤0.3)' if L1<=0.3 else ('걸음 크기 설명' if LM>=2/3*L1 else '방향(누적) 설명' if LM<=L1/3 else '혼재')
    out+=['',f'L1=D0L−D1L={L1:+.2f}%p, LM=D0L−D0M={LM:+.2f}%p → 사전 패턴: {pat}','']
open(R+'/../RESULTS_stepdir2.md','w').write('\n'.join(out)+'\n');print('\n'.join(out))
