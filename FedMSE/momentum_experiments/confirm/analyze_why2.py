import json,os,sys,numpy as np,pandas as pd
from scipy import stats
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.expanduser('~/fedmse/src'))
import dataset_adapter as DA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
def client_auc(R_,ks):  # name -> mean R31~40 AUC (over runs)
    names=R_[ks[0]][0]['devices']
    m=np.mean([[np.mean([x['global_auc'][i] for x in R_[r][30:]]) for i in range(len(names))] for r in ks],axis=0)*100
    return dict(zip(names,m))
def props(Xn,Xa,pool_mu,pool_sd,cap):
    n=len(Xn);ntr=int(.4*min(n,cap))
    sc=StandardScaler().fit(Xn[:ntr] if False else Xn[np.random.default_rng(0).permutation(n)[:max(ntr,2)]])
    sep=float(np.median(np.abs(sc.transform(Xa)).mean(1)))
    shift=float(np.linalg.norm(((Xn-pool_mu)/pool_sd).mean(0)))
    return dict(n_raw=n,n_train=ntr,shift=shift,sep=sep,n_att=len(Xa))
# ---- data per dataset
P={}
X,y,typ,ip=DA._ton_features()
hosts=[h for h,c in pd.Series(ip[y==0]).value_counts().items() if c>=1500]
pool=X[y==0];mu=pool.mean(0);sd=pool.std(0)+1e-9
att=X[y==1]
rng=np.random.default_rng(0);aid=np.flatnonzero(y==1);attrows=np.concatenate([aid[typ[aid]==t][:1000] for t in sorted(set(typ[aid]))]);Xa=X[attrows]
P['ton']={str(h):props(X[(y==0)&(ip==h)],Xa,mu,sd,DA.TON_PER_CLIENT) for h in hosts}
# N-BaIoT
from DataLoader import load_data
base=os.path.expanduser('~/fedmse/Data/N-BaIoT/noniid-10-Client_Data');nb={};allN=[]
for k in range(1,11):
    nb[k]=(load_data(f'{base}/Client-{k}/normal').to_numpy(float),load_data(f'{base}/Client-{k}/abnormal').to_numpy(float))
pool=np.concatenate([v[0] for v in nb.values()]);mu=pool.mean(0);sd=pool.std(0)+1e-9
P['nb']={f'NBa-Scen2-Client-{k}':props(v[0],v[1],mu,sd,10**9) for k,v in nb.items()}
# CIC
ben,attc,_=DA._cic_cache()
def cic_props(seed):
    sc=StandardScaler().fit(ben[:200000]);z=sc.transform(ben)
    km=KMeans(10,n_init=3,random_state=int(seed)).fit(z[:200000]);lab=km.predict(z)
    mu=ben.mean(0);sd=ben.std(0)+1e-9
    return {f'cic{k}':props(ben[lab==k][:100000],attc,mu,sd,DA.CIC_PER_CLIENT) for k in range(10)}
cells=[('ton',1234,'D0_ton_s1234','D1_ton_s1234','D2_ton_s1234','ton'),('ton',2024,'D0_ton_s2024','D1_ton_s2024','D2_ton_s2024','ton'),
('cic',1234,'D0_cic_km_s1234','D1_cic_km_s1234','D2_cic_km_s1234','cic'),
('nb',1234,'D0_mse_par','D1_msem_par','D2_mse_par','nb'),('nb',2024,'D0_s2024_lr1e-5','D1_s2024_lr1e-5','D2_s2024_lr1e-5','nb'),
('nb',4321,'D0_s4321_lr1e-5','D1_s4321_lr1e-5','D2_s4321_lr1e-5','nb'),('nb',7777,'D0_s7777_lr1e-5','D1_s7777_lr1e-5','D2_s7777_lr1e-5','nb')]
out=['# 손해/이득 클라이언트의 데이터 특성 (탐색·사후; CRITERIA_why2.md)','','P1 n_raw/n_train=정상 행 수, P2 shift=풀 평균에서 클라이언트 평균 거리, P3 sep=공격 분리도(클수록 쉬움), P4 n_att=공격 행 수(ToN·CIC는 모든 클라이언트 공통 풀), P0=D0 AUC','']
keys=['n_raw','shift','sep','n_att','P0']
summary={}
for ds,s,c0,c1,c2,pk in cells:
    A=runs('mse',c0);B=runs('mse_m',c1);C=runs('mse_dm',c2,[10]);ks=sorted(set(A)&set(B))
    a=client_auc(A,ks);b=client_auc(B,ks);d1={k:b[k]-a[k] for k in a}
    d2=None
    if 10 in C and 10 in A: a10=client_auc(A,[10]);c10=client_auc(C,[10]);d2={k:c10[k]-a10[k] for k in a10}
    pr=cic_props(s) if ds=='cic' else P[pk]
    names=[k for k in a if k in pr]
    if len(names)<len(a): out.append(f'(경고: {ds} {s} 특성 매핑 {len(names)}/{len(a)})')
    order=sorted(names,key=lambda k:d1[k])
    lose=order[:3];gain=order[-3:]
    out+=[f'## {ds} 시드 {s}','',f'D1−D0 손해군(하위3): {", ".join(f"{k}({d1[k]:+.1f})" for k in lose)}',f'이득군(상위3): {", ".join(f"{k}({d1[k]:+.1f})" for k in gain)}','']
    out.append('|클라이언트|D1−D0|D2−D0(run10)|D0 AUC|n_raw|n_train|shift|sep|n_att|군|');out.append('|---|---|---|---|---|---|---|---|---|---|')
    for k in order:
        p=pr[k];grp='손해' if k in lose else '이득' if k in gain else ''
        out.append(f"|{k}|{d1[k]:+.2f}|{'-' if not d2 else f'{d2[k]:+.2f}'}|{a[k]:.1f}|{p['n_raw']}|{p['n_train']}|{p['shift']:.2f}|{p['sep']:.2f}|{p['n_att']}|{grp}|")
    rho={}
    for key in keys:
        v=[a[k] if key=='P0' else pr[k][key] for k in names]
        rho[key]=stats.spearmanr(v,[d1[k] for k in names])[0] if len(set(v))>1 else float('nan')
    ov={}
    for key in keys[:3]:
        lv=[pr[k][key] for k in lose];gv=[pr[k][key] for k in gain]
        ov[key]='겹침' if not(max(lv)<min(gv) or max(gv)<min(lv)) else '안 겹침'
    out+=['','스피어만 ρ(특성, D1−D0): '+', '.join(f'{k} {v:+.2f}' for k,v in rho.items()),'손해군·이득군 범위: '+', '.join(f'{k} {v}' for k,v in ov.items()),'']
    summary.setdefault(ds,[]).append((s,rho,ov))
out.append('## 데이터셋별 요약(사전 패턴: 모든 셀 같은 부호 |ρ|≥0.7 + 범위 안 겹침이면 지지)')
for ds,v in summary.items():
    out.append(f'- {ds}: '+'; '.join(f"시드{s}: "+', '.join(f"{k}{r:+.2f}" for k,r in rho.items() if k in ('n_raw','shift','sep','P0')) for s,rho,ov in v))
open(R+'/../RESULTS_why2.md','w').write('\n'.join(out)+'\n');print('\n'.join(out))
