import json, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); D=HERE+'/runs'
def load(v,c,r=10):
    f=f'{D}/{v}/{c}/Run_{r}.jsonl'
    return [list(json.loads(l).values())[0] for l in open(f)] if os.path.exists(f) else []
d0=load('mse','D0_ton_s1234')[:20]
auc=lambda L:np.array([np.mean(x['global_auc'])*100 for x in L])
arms=[('D0 (MSEAvg)','mse','D0_ton_s1234')]
for e in ('0.003','0.01','0.03'): arms+= [(f'FedAdam η={e}','mse_adam',f'O1_adam_e{e}_ton_s1234')]
for e in ('0.003','0.01','0.03'): arms+= [(f'FedYogi η={e}','mse_yogi',f'O1_yogi_e{e}_ton_s1234')]
arms+=[('모멘텀 β=0.5 η=1','mse_m','O1_mom_b0.5_e1_ton_s1234'),('모멘텀 β=0.9 η=0.1','mse_m','O1_mom_b0.9_e0.1_ton_s1234')]
out=['# 서버 옵티마이저 1단계 (ToN data_seed 1234, run10 1런, 20라운드; 탐색용·n=1이라 구간 없음)','','|설정|완료라운드|R11~20 평균 AUC|D0 대비(%p)|R20|최저 AUC(R5+)|붕괴(<90)|서버 이동거리 중앙값|마지막 검증손실|AUC 곡선 R1,3,5,8,12,16,20|','|---|---|---|---|---|---|---|---|---|---|']
base=auc(d0)[10:20].mean() if len(d0)>=20 else None
for nm,v,c in arms:
    L=load(v,c)[:20]
    if not L: out.append(f'|{nm}|0|-|-|-|-|-|-|-|-|'); continue
    a=auc(L); n=len(L); r1120=a[10:20].mean() if n>=20 else float('nan')
    cur=','.join(f'{a[i-1]:.1f}' if i<=n else '-' for i in (1,3,5,8,12,16,20))
    mn=a[4:].min() if n>=5 else float('nan'); col=int((a[4:]<90).any()) if n>=5 else 0
    dd=f'{r1120-base:+.2f}' if base is not None and n>=20 else '-'
    out.append(f"|{nm}|{n}|{r1120:.2f}|{dd}|{a[19]:.2f}|{mn:.1f}|{'예' if col else '아니오'}|{np.median([x['diag2']['global_disp'] for x in L]):.2f}|{L[-1]['global_loss']:.2f}|{cur}|" if n>=20 else f"|{nm}|{n}|(진행 중)|-|-|{mn:.1f}|{'예' if col else '아니오'}|-|{L[-1]['global_loss']:.2f}|{cur}|")
open(HERE+'/RESULTS_optim1.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
