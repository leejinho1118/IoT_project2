import json, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); D=HERE+'/runs'
def load(v,c,r=10):
    f=f'{D}/{v}/{c}/Run_{r}.jsonl'
    return [list(json.loads(l).values())[0] for l in open(f)] if os.path.exists(f) else []
auc=lambda L:np.array([np.mean(x['global_auc'])*100 for x in L])
out=['# 동적 모멘텀 개선 A·B (data_seed 1234, run10 1런, 40라운드, β=0.9 η=1; 탐색용·n=1)','','A: β_t=0.9·min(1,M_prev/M_now)^2 / B: 최종 모델 MSE 재확인 후 필요 시 β_t 반감(최대 1회)','','|데이터셋|설정|완료라운드|R31~40 평균|D0 대비(%p)|D1 대비(%p)|R5+ 최저|붕괴(<90)|β_t 평균|β_t 최소|β_t<0.9 라운드 수|β_t 사분위(25/50/75)|B 반감 라운드 수|평가시간(초/라운드)|평가 횟수/라운드|AUC 곡선 R1,3,5,10,20,30,40|','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for ds,t in (('ToN','ton'),('N-BaIoT','nbaiot')):
    d0,d1=('D0_ton_s1234','D1_ton_s1234') if t=='ton' else ('D0_mse_par','D1_msem_par')
    A={nm:load(v,c)[:40] for nm,v,c in (('D0','mse',d0),('D1','mse_m',d1),('가드(리셋)','mse_mg',f'O2g_{t}_s1234'),('동적','mse_md',f'O3d_{t}_s1234'),('A(제곱)','mse_mda',f'O4a_{t}_s1234'),('B(재확인)','mse_mdb',f'O4b_{t}_s1234'))}
    b0=auc(A['D0'])[30:40].mean(); b1=auc(A['D1'])[30:40].mean()
    for nm,L in A.items():
        n=len(L)
        if not L: out.append(f'|{ds}|{nm}|0|'+'-|'*13); continue
        a=auc(L); m=a[30:40].mean() if n>=40 else float('nan')
        cur=','.join(f'{a[i-1]:.1f}' if i<=n else '-' for i in (1,3,5,10,20,30,40))
        G=[x['guard'] for x in L if 'guard' in x]; bt=[g['beta_t'] for g in G if 'beta_t' in g]
        e=(f"{np.mean(bt):.3f}|{np.min(bt):.3f}|{sum(b<0.8999 for b in bt)}/{len(bt)}|{np.percentile(bt,25):.2f}/{np.percentile(bt,50):.2f}/{np.percentile(bt,75):.2f}") if bt else '-|-|-|-'
        hv=str(sum(g.get('halved',False) for g in G)) if nm.startswith('B') else '-'
        tm=f"{np.mean([g['time'] for g in G]):.4f}" if G else '-'
        ev=f"{np.mean([g.get('evals',2) for g in G]):.1f}" if bt else '-'
        out.append(f"|{ds}|{nm}|{n}|{m:.2f}|{m-b0:+.2f}|{m-b1:+.2f}|{a[4:].min():.1f}|{'예' if (a[4:]<90).any() else '아니오'}|{e}|{hv}|{tm}|{ev}|{cur}|")
open(HERE+'/RESULTS_dyn2.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
