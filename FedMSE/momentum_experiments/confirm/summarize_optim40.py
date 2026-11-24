import json, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); D=HERE+'/runs'
def load(v,c,r=10):
    f=f'{D}/{v}/{c}/Run_{r}.jsonl'
    return [list(json.loads(l).values())[0] for l in open(f)] if os.path.exists(f) else []
auc=lambda L:np.array([np.mean(x['global_auc'])*100 for x in L])
arms=[('D0 (MSEAvg)','mse','D0_ton_s1234'),('모멘텀 β=0.9 η=0.03','mse_m','O1y_mom_b0.9_e0.03_ton_s1234'),('모멘텀 β=0.9 η=0.1','mse_m','O1y_mom_b0.9_e0.1_ton_s1234')]
out=['# 모멘텀 40라운드 확인 (ToN data_seed 1234, run10 1런; n=1이라 구간 없음)','','|설정|완료라운드|R31~40 평균|D0 대비(%p)|최저 AUC(R5+)|붕괴(<90)|AUC 곡선 R1,3,5,10,20,30,40|','|---|---|---|---|---|---|---|']
base=auc(load('mse','D0_ton_s1234')[:40])[30:40].mean()
for nm,v,c in arms:
    L=load(v,c)[:40]; n=len(L)
    if not L: out.append(f'|{nm}|0|-|-|-|-|-|'); continue
    a=auc(L); m=a[30:40].mean() if n>=40 else float('nan')
    cur=','.join(f'{a[i-1]:.1f}' if i<=n else '-' for i in (1,3,5,10,20,30,40))
    out.append(f"|{nm}|{n}|{m:.2f}|{m-base:+.2f}|{a[4:].min():.1f}|{'예' if (a[4:]<90).any() else '아니오'}|{cur}|")
open(HERE+'/RESULTS_optim40.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
