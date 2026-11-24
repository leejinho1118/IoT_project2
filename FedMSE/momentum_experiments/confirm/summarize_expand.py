"""CRITERIA_expand.md v2 요약. 사용: python3 summarize_expand.py"""
import json, os, numpy as np
from scipy import stats
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'runs'); DELTA=0.3
P={'A':'fedavg','B':'fedavgm','C3':'eta10','C10':'eta10','X1':'fedavgm','X2':'fedavgm','E':'fedavg'}
def load(cell):
    v=P[cell.split('_')[0]]; out={}
    for r in range(10):
        f=f'{D}/{v}/{cell}/Run_{r}.jsonl'
        if not os.path.exists(f): continue
        L=[list(json.loads(l).values())[0] for l in open(f)]
        if len(L)==40: out[r]=[float(np.mean(x['global_auc']))*100 for x in L]
    return out
def diff(a,b,rd,rb=None):
    rb=rd if rb is None else rb
    ks=sorted(set(a)&set(b)); d=np.array([a[k][rd-1]-b[k][rb-1] for k in ks]); return d
def ci(d):
    n=len(d)
    if n<2: return (np.nan,)*3+(n,)
    m=d.mean(); h=stats.t.ppf(.975,n-1)*d.std(ddof=1)/np.sqrt(n); p=stats.ttest_1samp(d,0).pvalue
    return m,m-h,m+h,n,p
def verdict(lo,hi):
    if lo>DELTA: return '실질 우월'
    if lo>0: return '우월(소폭)'
    if lo>=-DELTA and hi<=DELTA: return '실질 동등'
    return '미결'
C={}
for t in ('lr1e-5','lr5e-5'):
    for k in ('A','B','E','C3','C10','X1','X2'): 
        c=f'{k}_{t}'
        if os.path.exists(f'{D}/{P[k]}/{c}'): C[c]=load(c)
out=['# 확대 검증 결과 (자동 생성)','','## 셀별 평균 AUC(%) [완료 런 n]','|셀|n|R20|R40|','|---|---|---|---|']
for c,v in C.items():
    if v: out.append(f'|{c}|{len(v)}|{np.mean([x[19] for x in v.values()]):.2f}|{np.mean([x[39] for x in v.values()]):.2f}|')
    else: out.append(f'|{c}|0|-|-|')
def row(name,a,b,rd,rb=None):
    if a not in C or b not in C: return None,f'|{name}|-|-|-|0|-|'
    m,lo,hi,n,*p=ci(diff(C[a],C[b],rd,rb)); return (p[0] if p else None),f'|{name}|{m:+.2f}|[{lo:+.2f}, {hi:+.2f}]|{verdict(lo,hi) if n>1 else "-"}|{n}|{"" if not p else f"{p[0]:.4f}"}|'
out+=['','## 확증 대비 (Holm 보정 3개; p는 비보정)','|대비|평균차(%p)|95%구간|판정|n|p|','|---|---|---|---|---|---|']
conf=[('H1 B−E R20 (lr1e-5)','B_lr1e-5','E_lr1e-5',20),('H2 B−E R40 (lr1e-5)','B_lr1e-5','E_lr1e-5',40),('H3 B′−E′ R20 (lr5e-5)','B_lr5e-5','E_lr5e-5',20)]
ps=[]
for nm,a,b,r in conf:
    p,l=row(nm,a,b,r); out.append(l); ps.append(p)
valid=[(i,p) for i,p in enumerate(ps) if p is not None]
if len(valid)==3:
    order=sorted(valid,key=lambda x:x[1]); adj={}; run=0
    for rank,(i,p) in enumerate(order): run=max(run,min(1,(3-rank)*p)); adj[i]=run
    out.append(''); out.append('Holm 보정 p: '+', '.join(f'{conf[i][0].split()[0]}={adj[i]:.4f}' for i in range(3)))
out+=['','## 탐색적 대비','|대비|평균차(%p)|95%구간|판정|n|p|','|---|---|---|---|---|---|']
for nm,a,b,r,rb in [('B−A R20','B_lr1e-5','A_lr1e-5',20,None),('B−A R40','B_lr1e-5','A_lr1e-5',40,None),('B−C3 R20','B_lr1e-5','C3_lr1e-5',20,None),('B−C10 R20','B_lr1e-5','C10_lr1e-5',20,None),
    ('B−C10 R40','B_lr1e-5','C10_lr1e-5',40,None),('B′−A′ R20','B_lr5e-5','A_lr5e-5',20,None),('A′−A R20 (학습부족)','A_lr5e-5','A_lr1e-5',20,None),('A40−A20 (학습부족)','A_lr1e-5','A_lr1e-5',40,20),
    ('X1−B R20','X1_lr1e-5','B_lr1e-5',20,None),('X2−B R20','X2_lr1e-5','B_lr1e-5',20,None)]:
    out.append(row(nm,a,b,r,rb)[1])
# 격차 변화
def gapchg(nm,ca,cb,ra,rb_):
    try:
        A1,B1,A0,B0=C[ca[0]],C[ca[1]],C[cb[0]],C[cb[1]]
    except KeyError: out.append(f'|{nm}|-|-|-|0|-|'); return
    ks=sorted(set(A1)&set(B1)&set(A0)&set(B0)); d=np.array([(B1[k][ra-1]-A1[k][ra-1])-(B0[k][rb_-1]-A0[k][rb_-1]) for k in ks]); m,lo,hi,n,*p=ci(d)
    out.append(f'|{nm}|{m:+.2f}|[{lo:+.2f}, {hi:+.2f}]|{verdict(lo,hi) if n>1 else "-"}|{n}|{"" if not p else f"{p[0]:.4f}"}|')
gapchg('격차변화 (B′−A′)−(B−A) R20',('A_lr5e-5','B_lr5e-5'),('A_lr1e-5','B_lr1e-5'),20,20)
gapchg('격차변화 (B40−A40)−(B20−A20)',('A_lr1e-5','B_lr1e-5'),('A_lr1e-5','B_lr1e-5'),40,20)
div=[f'{c}: {sum(1 for x in v.values() if any(a<90 for a in x[4:]))}' for c,v in C.items()]
out+=['','## 발산 런 수(R5+ 평균 AUC<90%)',', '.join(div)]
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'RESULTS_expand.md'),'w').write('\n'.join(out)+'\n'); print('\n'.join(out))
