import json,os,numpy as np
D=os.path.dirname(os.path.abspath(__file__))+'/runs'
def runs(v,c):
    o={};d=f'{D}/{v}/{c}'
    if not os.path.isdir(d): return o
    for f in os.listdir(d):
        if f.startswith('Run_'):
            L=[list(json.loads(l).values())[0] for l in open(f'{d}/{f}')]
            if len(L)==40: o[int(f[4:-6])]=dict(auc=np.array([np.mean(x['global_auc'])*100 for x in L]),gd=np.array([x['diag2']['global_disp'] for x in L]),E=np.array([x['diag2']['E'] for x in L]))
    return o
S=[slice(0,5),slice(5,15),slice(30,40)]
def stat(R,ks,key,f):
    return [f(np.array([R[r][key][s] for r in ks]).ravel() if key!='auc' else np.mean([R[r][key][s].mean() for r in ks])) for s in S]
def drops(R,ks): return np.mean([(np.diff(R[r]['auc'])[1:]<=-2).sum() for r in ks]) if False else np.mean([((R[r]['auc'][2:]-R[r]['auc'][1:-1])<=-2).sum() for r in ks])
cells=[('N-BaIoT',1234,'D0_mse_par','D1_msem_par','D2_mse_par')]+[('N-BaIoT',s,f'D0_s{s}_lr1e-5',f'D1_s{s}_lr1e-5',f'D2_s{s}_lr1e-5') for s in (2024,4321,7777)]
cells+=[('ToN',s,f'D0_ton_s{s}',f'D1_ton_s{s}',f'D2_ton_s{s}') for s in (1234,2024,4321)]+[('CIC-km',s,f'D0_cic_km_s{s}',f'D1_cic_km_s{s}',f'D2_cic_km_s{s}') for s in (1234,4321)]
out=['# 진단 비교 (탐색·사후; CRITERIA_diag.md)','','구간 = R1~5 / R6~15 / R31~40. AUC=런평균(%), 이동=global_disp 중앙값, 비율=대 D0, E=global/local, 낙하=AUC 2%p 이상 하락 라운드 수.','']
for ds,s,c0,c1,c2 in cells:
    A=runs('mse',c0);B=runs('mse_m',c1);C=runs('mse_dm',c2)
    out.append(f'## {ds} 시드 {s}')
    out.append('|방법|런|AUC R1~5|R6~15|R31~40|이동 R1~5|R6~15|R31~40|비율(대D0) R1~5|R6~15|R31~40|E 중앙(R1~5/R6~15/R31~40)|낙하|'); out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for label,R,ks in (('D0 (D1짝 런)',A,sorted(set(A)&set(B))),('D1',B,sorted(set(A)&set(B))),('D0 (run10)',A,[10] if 10 in A else []),('D2 (run10)',C,[10] if 10 in C and 10 in A else [])):
        if not ks or not R: continue
        au=stat(R,ks,'auc',lambda x:x); mv=stat(R,ks,'gd',np.median); e=stat(R,ks,'E',np.median)
        base={'D0 (D1짝 런)':None}
        ref=A; mref=stat(ref,ks,'gd',np.median)
        rt=[m/b for m,b in zip(mv,mref)]
        out.append(f"|{label}|{len(ks)}|{au[0]:.1f}|{au[1]:.1f}|{au[2]:.1f}|{mv[0]:.3f}|{mv[1]:.3f}|{mv[2]:.3f}|{rt[0]:.2f}|{rt[1]:.2f}|{rt[2]:.2f}|{e[0]:.2f}/{e[1]:.2f}/{e[2]:.2f}|{drops(R,ks):.1f}|")
    out.append('')
open(D+'/../RESULTS_diag.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
