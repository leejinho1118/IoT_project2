import json,os,numpy as np
from scipy import stats
R=os.path.dirname(os.path.abspath(__file__))+'/runs'
def runs(v,c,only=None):
    d=f'{R}/{v}/{c}';o={}
    if not os.path.isdir(d): return o
    for f in sorted(os.listdir(d)):
        if f.startswith('Run_'):
            r=int(f[4:-6])
            if only and r not in only: continue
            L=[list(json.loads(l).values())[0] for l in open(f'{d}/{f}')]
            if len(L)==40: o[r]=L
    return o
mean_auc=lambda L:np.array([np.mean(x['global_auc'])*100 for x in L])
cells=[('N-BaIoT',1234,'D0_mse_par','D1_msem_par','D2_mse_par'),('N-BaIoT',2024,'D0_s2024_lr1e-5','D1_s2024_lr1e-5','D2_s2024_lr1e-5'),
('N-BaIoT',4321,'D0_s4321_lr1e-5','D1_s4321_lr1e-5','D2_s4321_lr1e-5'),('N-BaIoT',7777,'D0_s7777_lr1e-5','D1_s7777_lr1e-5','D2_s7777_lr1e-5'),
('ToN',1234,'D0_ton_s1234','D1_ton_s1234','D2_ton_s1234'),('ToN',2024,'D0_ton_s2024','D1_ton_s2024','D2_ton_s2024'),('ToN',4321,'D0_ton_s4321',None,'D2_ton_s4321'),
('CIC-km',1234,'D0_cic_km_s1234','D1_cic_km_s1234','D2_cic_km_s1234'),('CIC-km',4321,'D0_cic_km_s4321',None,'D2_cic_km_s4321')]
out=['# 원인 분석 1단계: 기존 로그 분석 (탐색·사후; DESIGN_why_20261124.md 사전 예측 사용, 신규 실행 없음)','']
# H1
out+=['## H1 정체 돌파','','(a) D0 클라이언트 평균 AUC의 R6~40 선형 기울기(%p/라운드, 런 평균 곡선) · R6~15 대비 R31~40 차이','','|데이터셋|시드|기울기|R31~40−R6~15|','|---|---|---|---|']
for ds,s,c0,c1,c2 in cells:
    A=runs('mse',c0)
    if not A: continue
    m=np.mean([mean_auc(L) for L in A.values()],axis=0); sl=np.polyfit(np.arange(5,40),m[5:],1)[0]
    out.append(f'|{ds}|{s}|{sl:+.3f}|{m[30:].mean()-m[5:15].mean():+.2f}|')
out+=['','(b)(c) 클라이언트별: D0의 R31~40 클라이언트 AUC vs D1이 올린/내린 양(같은 run 짝), 스피어만 상관(D0 낮을수록 이득 큰가), 이득의 하위3 D0 클라이언트 몫','','|데이터셋|시드|클라이언트 수|D0 최저·최고 AUC|D1−D0 클라이언트별 최소/최대|스피어만 ρ(D0 AUC, 이득)|이득 합 중 D0 하위3 몫|','|---|---|---|---|---|---|---|']
for ds,s,c0,c1,c2 in cells:
    if not c1: continue
    A=runs('mse',c0);B=runs('mse_m',c1);ks=sorted(set(A)&set(B))
    if not ks: continue
    n=len(A[ks[0]][0]['global_auc'])
    a=np.mean([[np.mean([x['global_auc'][i] for x in A[r][30:]]) for i in range(n)] for r in ks],axis=0)*100
    b=np.mean([[np.mean([x['global_auc'][i] for x in B[r][30:]]) for i in range(n)] for r in ks],axis=0)*100
    g=b-a;rho=stats.spearmanr(a,g)[0];low=np.argsort(a)[:3];share=g[low].sum()/g.sum() if g.sum()!=0 else float('nan')
    out.append(f'|{ds}|{s}|{n}|{a.min():.1f}·{a.max():.1f}|{g.min():+.2f}/{g.max():+.2f}|{rho:+.2f}|{share:.2f}|')
# H2
out+=['','## H2 라운드 간 방향 일관성','','D0의 cos_prev(방향 로깅 런, ToN·CIC만 존재, N-BaIoT 없음)와 D1의 cos(Δ_t, v_{t−1}) 중앙값(run 10, guard.cos; N-BaIoT·ToN은 O5c 재실행, CIC는 D1L)','','|데이터셋|D0 cos_prev 중앙|D1 cos(Δ,v) 중앙|D1 cos 음수 비율|','|---|---|---|---|']
def gcos(v,c):
    L=runs(v,c,[10])[10]; x=np.array([r['guard']['cos'] for r in L if r.get('guard',{}).get('cos') is not None]); return np.median(x),(x<0).mean()
def d0cp(c):
    L=runs('mse',c,[10])[10]; return np.median([r['diag2']['cos_prev'] for r in L if r['diag2'].get('cos_prev') is not None])
for ds,d0,d1v,d1c in (('N-BaIoT',None,'mse_m','O5c_nbaiot_s1234'),('ToN','D0L_ton_s1234','mse_m','D1L_ton_s1234'),('CIC-km','D0L_cic_km_s1234','mse_m','D1L_cic_km_s1234')):
    m,neg=gcos(d1v,d1c); out.append(f"|{ds}|{'미측정' if d0 is None else f'{d0cp(d0):.2f}'}|{m:+.2f}|{neg:.0%}|")
# H3
out+=['','## H3 클라이언트 정렬도','','E(‖평균 델타‖/평균 개별 노름, D0 전 셀 중앙): N-BaIoT 0.84~0.85, CIC 0.76~0.77, ToN 0.66~0.72 (RESULTS_stepdir.md). pair_cos(직접 지표): ToN 0.18, CIC 0.43(D0L run10), N-BaIoT 미측정.']
# H4
out+=['','## H4 D2 초반 undershoot (ToN)','','|시드|D0 R1~5|D2 R1~5|D2−D0 R1~5|D2−D0 R6~15|D2−D0 R31~40|D2/D0 걸음비(R1~5)|','|---|---|---|---|---|---|---|']
for s,c0,c2 in ((1234,'D0_ton_s1234','D2_ton_s1234'),(2024,'D0_ton_s2024','D2_ton_s2024'),(4321,'D0_ton_s4321','D2_ton_s4321')):
    a=mean_auc(runs('mse',c0,[10])[10]);b=mean_auc(runs('mse_dm',c2,[10])[10])
    ga=np.median([x['diag2']['global_disp'] for x in runs('mse',c0,[10])[10][:5]]);gb=np.median([x['diag2']['global_disp'] for x in runs('mse_dm',c2,[10])[10][:5]])
    out.append(f'|{s}|{a[:5].mean():.1f}|{b[:5].mean():.1f}|{b[:5].mean()-a[:5].mean():+.2f}|{b[5:15].mean()-a[5:15].mean():+.2f}|{b[30:].mean()-a[30:].mean():+.2f}|{gb/ga:.2f}|')
# H5
out+=['','## H5 검증 손실(global_loss) 급등','','D1/D0의 같은 run 최대 손실 배율(런 중앙), D1 손실이 D0 손실의 2배를 넘은 라운드 수(런 평균, 40라운드 중)','','|데이터셋|시드|D1 최대손실/D0 최대손실|D1>2×D0 라운드 수|D0M>2×D0 라운드 수(있을 때)|','|---|---|---|---|---|']
for ds,s,c0,c1,c2 in cells:
    if not c1: continue
    A=runs('mse',c0);B=runs('mse_m',c1);ks=sorted(set(A)&set(B))
    if not ks: continue
    ratio=np.median([max(x['global_loss'] for x in B[r])/max(x['global_loss'] for x in A[r]) for r in ks])
    over=np.mean([sum(b['global_loss']>2*a['global_loss'] for a,b in zip(A[r],B[r])) for r in ks])
    dm='-'
    if ds in('ToN','CIC-km') and s==1234:
        k={'ToN':'ton','CIC-km':'cic_km'}[ds];M=runs('mse',f'D0M_{k}_s1234',[10])[10];A0=runs('mse',f'D0L_{k}_s1234',[10])[10]
        dm=str(sum(m['global_loss']>2*a['global_loss'] for a,m in zip(A0,M)))
    out.append(f'|{ds}|{s}|{ratio:.2f}|{over:.1f}|{dm}|')
open(R+'/../RESULTS_why1.md','w').write('\n'.join(out)+'\n');print('\n'.join(out))
