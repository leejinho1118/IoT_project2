import json,os,numpy as np
from scipy import stats
R=os.path.dirname(os.path.abspath(__file__))+'/runs'
def runs(v,c,only=None):
    d=f'{R}/{v}/{c}';o={}
    for f in sorted(os.listdir(d)):
        if f.startswith('Run_'):
            r=int(f[4:-6])
            if only and r not in only: continue
            L=[list(json.loads(l).values())[0] for l in open(f'{d}/{f}')]
            if len(L)==40: o[r]=L
    return o
LOSE=['192.168.1.195','192.168.1.190','192.168.1.152']
def mat(L): # rounds x hosts, names
    names=L[0]['devices'];return np.array([x['global_auc'] for x in L])*100,names
def disp(L): return np.array([x['diag2']['global_disp'] for x in L])
out=['# ToN 호스트별 라운드 곡선 (탐색·사후; CRITERIA_why3.md)','']
def analyze(label,V,ks,A,Dp=None,peak=True):
    res={'lose':dict(drops=[],dep=[],rec=[],rho_x=[],rho_y=[],pk=[]),'ceil':dict(drops=[],dep=[],rec=[],rho_x=[],rho_y=[],pk=[])}
    for r in ks:
        a,names=mat(A[r]);v,_=mat(V[r]);dv=disp(V[r]);d0=np.median(disp(A[r]))
        d0f=a[30:].mean(0);ceil=[n for n,x in zip(names,d0f) if x>=99];
        thr=np.percentile(dv,75);pk=(dv>=thr)
        for grp,hs in (('lose',LOSE),('ceil',ceil)):
            for h in hs:
                i=names.index(h);x=v[:,i];dx=np.diff(x)  # change at round t (index 1..)
                drops=np.where(dx<=-2)[0]+1
                res[grp]['drops'].append(len(drops))
                res[grp]['pk']+= [bool(pk[t] or pk[t-1]) for t in drops]
                gap=x-a[:,i];dep=np.where(gap<=-3)[0]
                res[grp]['dep'].append(dep[0]+1 if len(dep) else np.nan)
                res[grp]['rec'].append(abs(x[30:].mean()-a[30:,i].mean())<=3 if len(dep) else np.nan)
                res[grp]['rho_x']+=list(dv[1:]/d0);res[grp]['rho_y']+=list(dx)
    s=[f'### {label} (런 {len(ks)})','','|군|호스트-런 수|낙하 횟수 평균|이탈한 비율|이탈 라운드 중앙|이탈 후 회복 비율|낙하의 정점+직후 비율|ρ(걸음비, AUC변화)|','|---|---|---|---|---|---|---|---|']
    stat={}
    for grp in ('lose','ceil'):
        g=res[grp];dep=np.array(g['dep'],float);ok=~np.isnan(dep)
        rho=stats.spearmanr(g['rho_x'],g['rho_y'])[0]
        pk=np.mean(g['pk']) if g['pk'] else float('nan')
        rec=np.array(g['rec'],float);rec=np.nanmean(rec) if ok.any() else float('nan')
        s.append(f"|{'손해' if grp=='lose' else '천장'}|{len(g['drops'])}|{np.mean(g['drops']):.2f}|{ok.mean():.0%}|{np.nanmedian(dep) if ok.any() else '-'}|{rec:.0%}|{pk:.0%}|{rho:+.2f}|")
        stat[grp]=dict(drops=np.mean(g['drops']),rho=rho,pk=pk)
    out.extend(s+['']);return stat
S={}
for seed in (1234,2024):
    A=runs('mse',f'D0_ton_s{seed}');B=runs('mse_m',f'D1_ton_s{seed}');ks=sorted(set(A)&set(B))
    out.append(f'## 시드 {seed}')
    S[(seed,'D1')]=analyze('D1(vs D0, 10런)',B,ks,A)
    C=runs('mse_dm',f'D2_ton_s{seed}',[10]);S[(seed,'D2')]=analyze('D2(vs D0, run10)',C,[10],A)
    if seed==1234:
        M=runs('mse',f'D0M_ton_s1234',[10]);S[(seed,'D0M')]=analyze('D0M(vs D0, run10)',M,[10],A)
        B10={10:B[10]};S[(seed,'D1r10')]=analyze('D1(run10만)',B10,[10],A)
# D0 own drops baseline
d=[]
for seed in (1234,2024):
    A=runs('mse',f'D0_ton_s{seed}')
    for r,L in A.items():
        a,names=mat(L)
        for h in LOSE: d.append((np.diff(a[:,names.index(h)])<=-2).sum())
out.append(f'D0 자체 손해 호스트 낙하 횟수 평균(두 시드 모든 런): {np.mean(d):.2f}')
# criteria tally
for seed in (1234,2024):
    D1=S[(seed,'D1')];D2=S[(seed,'D2')]
    c=[D1['lose']['drops']>=3*max(D1['ceil']['drops'],1e-9) if D1['ceil']['drops']>0 else D1['lose']['drops']>0,
       (D1['lose']['rho']<=-0.3 and abs(D1['ceil']['rho'])<0.2),D1['lose']['pk']>=0.6,
       (S[(seed,'D0M')]['lose']['drops']>=D1['lose']['drops']/2) if seed==1234 else None, D2['lose']['drops']<=D1['lose']['drops']/2]
    out.append(f'시드 {seed} 지지 항목(i~v; None=해당 없음): {c}')
open(R+'/../RESULTS_why3.md','w').write('\n'.join(out)+'\n');print('\n'.join(out))
