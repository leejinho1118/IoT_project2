import json, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); D=HERE+'/runs'
def load(v,c,r=10):
    f=f'{D}/{v}/{c}/Run_{r}.jsonl'
    return [list(json.loads(l).values())[0] for l in open(f)] if os.path.exists(f) else []
auc=lambda L:np.array([np.mean(x['global_auc'])*100 for x in L])
out=['# 검증 MSE 가드 모멘텀 (data_seed 1234, run10 1런, 40라운드, β=0.9 η=1; 탐색용·n=1이라 구간 없음)','','|데이터셋|설정|완료라운드|R31~40 평균|D0 대비(%p)|D1 대비(%p)|R5+ 최저|붕괴(<90)|리셋 라운드 수|가드 평가 시간 평균(초/라운드)|AUC 곡선 R1,3,5,10,20,30,40|','|---|---|---|---|---|---|---|---|---|---|---|']
for ds,d0,d1,g in (('ToN','D0_ton_s1234','D1_ton_s1234','O2g_ton_s1234'),('N-BaIoT','D0_mse_par','D1_msem_par','O2g_nbaiot_s1234')):
    A={}
    for nm,v,c in (('D0','mse',d0),('D1','mse_m',d1),('가드','mse_mg',g)):
        L=load(v,c)[:40]; A[nm]=L
    b0=auc(A['D0'])[30:40].mean(); b1=auc(A['D1'])[30:40].mean()
    for nm,L in A.items():
        n=len(L)
        if not L: out.append(f'|{ds}|{nm}|0|-|-|-|-|-|-|-|-|'); continue
        a=auc(L); m=a[30:40].mean() if n>=40 else float('nan')
        cur=','.join(f'{a[i-1]:.1f}' if i<=n else '-' for i in (1,3,5,10,20,30,40))
        gr=[x['guard'] for x in L if 'guard' in x]
        rs=str(sum(g_['reset'] for g_ in gr)) if gr else '-'; tm=f"{np.mean([g_['time'] for g_ in gr]):.3f}" if gr else '-'
        out.append(f"|{ds}|{nm}|{n}|{m:.2f}|{m-b0:+.2f}|{m-b1:+.2f}|{a[4:].min():.1f}|{'예' if (a[4:]<90).any() else '아니오'}|{rs}|{tm}|{cur}|")
open(HERE+'/RESULTS_guard.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
