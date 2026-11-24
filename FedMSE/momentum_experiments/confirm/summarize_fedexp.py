import json,os,numpy as np
from scipy import stats
D=os.path.dirname(os.path.abspath(__file__))+'/runs'
def load(v,c,r):
    f=f'{D}/{v}/{c}/Run_{r}.jsonl'
    if not os.path.exists(f): return None
    return [list(json.loads(l).values())[0] for l in open(f)]
def auc(L): return np.array([np.mean(x['global_auc'])*100 for x in L])
G=[('N-BaIoT','FX_mse_par','D0_mse_par','D1_msem_par'),('ToN','FX_ton_s1234','D0_ton_s1234','D1_ton_s1234'),('CIC-km','FX_cic_km_s1234','D0_cic_km_s1234','D1_cic_km_s1234')]
R=range(10,15); out=['# FedExP(mse_fx) 5런 결과 (seed 1234, run 10~14, 탐색용·통과판정 없음; 설계 DESIGN_fedexp_20261124.md)','']
def ci(d):
    m=d.mean(); h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)); return m,m-h,m+h
for ds,fx,d0,d1 in G:
    out.append(f'## {ds}')
    X={r:load('mse_fx',fx,r) for r in R}; A={r:load('mse',d0,r) for r in R}; B={r:load('mse_m',d1,r) for r in R}
    rows=[];eta=[];dr=[];dep=[];dd=[];r20=[];r40=[];dB=[];dn=[];rep=[];same=[]
    for r in R:
        x,a,b=auc(X[r]),auc(A[r]),auc(B[r])
        rows.append((x[30:].mean(),a[30:].mean(),b[30:].mean()))
        r20.append(x[19]-a[19]); r40.append(x[39]-a[39])
        e=[p['diag2']['eta_g'] for p in X[r]]; eta+=e
        dr.append((int((np.diff(x)<=-2).sum()),int((np.diff(a)<=-2).sum()),int((np.diff(b)<=-2).sum())))
        dep.append(int((x<a-3).sum()))
        dd.append((np.mean([p['diag2']['global_disp'] for p in X[r]]),np.mean([p['diag2']['global_disp'] for p in A[r]]),np.mean([p['diag2']['global_disp'] for p in B[r]])))
        same.append(all(p['join_clients']==q['join_clients'] for p,q in zip(X[r],A[r])))
        rep.append(np.allclose(X[r][0]['diag2']['local_disp'],A[r][0]['diag2']['local_disp'],rtol=1e-4))
        dn.append(int(x[4:].mean()<90))
    M=np.array(rows); d=M[:,0]-M[:,1]; m,lo,hi=ci(d); d1_=M[:,0]-M[:,2]; m1,lo1,hi1=ci(d1_)
    out.append('|run|FX R31~40|D0|D1|FX−D0|FX−D1|')
    out.append('|---|---|---|---|---|---|')
    for r,(x,a,b) in zip(R,M): out.append(f'|{r}|{x:.2f}|{a:.2f}|{b:.2f}|{x-a:+.2f}|{x-b:+.2f}|')
    out.append(f'\n- FX−D0 평균 {m:+.2f}%p, 95% 구간 [{lo:+.2f}, {hi:+.2f}] (n=5, df=4)')
    out.append(f'- FX−D1 평균 {m1:+.2f}%p, 95% 구간 [{lo1:+.2f}, {hi1:+.2f}]')
    e=np.array(eta); out.append(f'- η_g: 중앙값 {np.median(e):.3f}, 최대 {e.max():.2f}, η_g>1.5 비율 {np.mean(e>1.5)*100:.1f}%, 최소 {e.min():.3f}')
    out.append(f'- 2%p 이상 낙하 횟수(런별 FX/D0/D1): {dr}')
    out.append(f'- D0보다 3%p 이상 낮은 라운드 수(런별): {dep}')
    out.append(f'- 평균 서버 이동거리 FX/D0/D1: {np.mean([t[0] for t in dd]):.3f}/{np.mean([t[1] for t in dd]):.3f}/{np.mean([t[2] for t in dd]):.3f}')
    out.append(f'- R20 FX−D0 {np.mean(r20):+.2f}, R40 FX−D0 {np.mean(r40):+.2f}')
    out.append(f'- 발산(R5 이후 평균<90) 런 수: {sum(dn)}; 참여순서 D0와 동일: {sum(same)}/5; 라운드1 델타노름 D0와 일치: {sum(rep)}/5')
    if ds=='ToN':
        cols=[(k,i) for i,k in enumerate(X[10][0]['devices'])]
        out.append('- 호스트별 R31~40 AUC (FX/D0/D1), 클라이언트 인덱스 기준:')
        def pc(Z): return np.mean([[np.mean([p['global_auc'][j] for p in Z[r][30:]]) for j in range(len(Z[r][0]['global_auc']))] for r in R],axis=0)*100
        px,pa,pb=pc(X),pc(A),pc(B)
        for j,dv in enumerate(X[10][0]['devices']): out.append(f'  - {dv}: {px[j]:.2f}/{pa[j]:.2f}/{pb[j]:.2f}')
    out.append('')
open(D+'/../RESULTS_fedexp.md','w').write('\n'.join(out)); print('\n'.join(out))
