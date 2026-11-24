import json,os,numpy as np
D=os.path.dirname(os.path.abspath(__file__))+'/runs'
def runs(v,c):
    o={}
    d=f'{D}/{v}/{c}'
    if not os.path.isdir(d): return o
    for f in os.listdir(d):
        if f.startswith('Run_'):
            L=[list(json.loads(l).values())[0] for l in open(f'{d}/{f}')]
            if len(L)==40: o[int(f[4:-6])]=np.array([np.mean(x['global_auc'])*100 for x in L])
    return o
def t95(a):
    i=np.where(a>=95)[0]; 
    for k in range(len(a)):
        pass
    return int(i[0])+1 if len(i) else 41
cells=[]
for s,d0,d1 in [(1234,'D0_mse_par','D1_msem_par')]+[(s,f'D0_s{s}_lr1e-5',f'D1_s{s}_lr1e-5') for s in (2024,4321,7777)]: cells.append(('N-BaIoT',s,d0,d1,d0,f'D2_mse_par' if s==1234 else f'D2_s{s}_lr1e-5'))
for s in (1234,2024,4321): cells.append(('ToN',s,f'D0_ton_s{s}',f'D1_ton_s{s}',None,f'D2_ton_s{s}'))
for s in (1234,4321): cells.append(('CIC-km',s,f'D0_cic_km_s{s}',f'D1_cic_km_s{s}',None,f'D2_cic_km_s{s}'))
out=['# 모멘텀 이득 vs 난이도 (탐색, CRITERIA_difficulty.md)','','사후 탐색(이득 부호를 이미 본 뒤 동결). D2는 run 10 한 런.','',
'|데이터셋|시드|D0 런수|T95 중앙값|분류|예측 부호|D1 이득(%p, n)|D1 일치|D2 이득(%p, n=1)|D2 일치|','|---|---|---|---|---|---|---|---|---|---|']
agree={}
for ds,s,c0,c1,_,c2 in cells:
    A=runs('mse',c0); B=runs('mse_m',c1); C=runs('mse_dm',c2)
    if not A: out.append(f'|{ds}|{s}|0|-|-|-|-|-|-|-|'); continue
    T=float(np.median([t95(a) for a in A.values()])); easy=T<=3; pred='≤0' if easy else '>0'
    ok=lambda g:(g<=0) if easy else (g>0)
    k=sorted(set(A)&set(B)); g1=np.mean([B[r][30:].mean()-A[r][30:].mean() for r in k]) if k else None
    g2=C[10][30:].mean()-A[10][30:].mean() if 10 in C and 10 in A else None
    m1=('O' if ok(g1) else 'X') if g1 is not None else '-'; m2=('O' if ok(g2) else 'X') if g2 is not None else '-'
    agree.setdefault(ds,[]).append((m1,m2))
    out.append(f"|{ds}|{s}|{len(A)}|{T:g}|{'쉬움' if easy else '어려움'}|{pred}|{g1:+.2f} (n={len(k)})|{m1}|{g2:+.2f}|{m2}|" if g1 is not None and g2 is not None else f"|{ds}|{s}|{len(A)}|{T:g}|{'쉬움' if easy else '어려움'}|{pred}|{'-' if g1 is None else f'{g1:+.2f} (n={len(k)})'}|{m1}|{'-' if g2 is None else f'{g2:+.2f}'}|{m2}|")
out.append(''); 
for ds,v in agree.items(): out.append(f"{ds}: D1 일치 {sum(x[0]=='O' for x in v)}/{sum(x[0]!='-' for x in v)}, D2 일치 {sum(x[1]=='O' for x in v)}/{sum(x[1]!='-' for x in v)}")
open(D+'/../RESULTS_difficulty.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
