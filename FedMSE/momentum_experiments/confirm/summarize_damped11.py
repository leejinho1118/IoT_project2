import json,os,numpy as np
D=os.path.dirname(os.path.abspath(__file__))+'/runs'
def load(v,c):
    f=f'{D}/{v}/{c}/Run_10.jsonl'
    if not os.path.exists(f): return None
    L=[list(json.loads(l).values())[0] for l in open(f)]
    return np.array([np.mean(x['global_auc'])*100 for x in L]), L
cells=[('N-BaIoT',1234,'mse','D0_mse_par','D2_mse_par'),('N-BaIoT',2024,'mse','D0_s2024_lr1e-5','D2_s2024_lr1e-5'),
('N-BaIoT',4321,'mse','D0_s4321_lr1e-5','D2_s4321_lr1e-5'),('N-BaIoT',7777,'mse','D0_s7777_lr1e-5','D2_s7777_lr1e-5'),
('ToN',1234,'mse','D0_ton_s1234','D2_ton_s1234'),('ToN',2024,'mse','D0_ton_s2024','D2_ton_s2024'),('ToN',4321,'mse','D0_ton_s4321','D2_ton_s4321'),
('CIC-km',1234,'mse','D0_cic_km_s1234','D2_cic_km_s1234'),('CIC-km',4321,'mse','D0_cic_km_s4321','D2_cic_km_s4321')]
out=['# 감쇠 모멘텀 D2 탐색 결과 (run 10 한 런씩, n=1, 판정 없음; DEVIATIONS_damped.md)','','CIC-km은 인위적 비IID(KMeans 분할) 한계.','',
'|데이터셋|시드|D0 R31~40|D2 R31~40|차이(%p)|D1(기존 모멘텀) 차이|D0 R5이후 최저|D2 R5이후 최저|D2 R20/R40 AUC|비고|','|---|---|---|---|---|---|---|---|---|---|']
for ds,s,v0,c0,c2 in cells:
    a=load(v0,c0); b=load('mse_dm',c2)
    d1c={'N-BaIoT':{1234:'D1_msem_par',2024:'D1_s2024_lr1e-5',4321:'D1_s4321_lr1e-5',7777:'D1_s7777_lr1e-5'}.get(s)}.get(ds) or f"D1_{'ton' if ds=='ToN' else 'cic_km'}_s{s}"
    c=load('mse_m',d1c)
    if a is None or b is None or len(a[0])<40 or len(b[0])<40: out.append(f'|{ds}|{s}|-|-|미완/없음|-|-|-|-||'); continue
    A,B=a[0],b[0]; dd=B[30:].mean()-A[30:].mean()
    d1=f"{c[0][30:].mean()-A[30:].mean():+.2f}" if c is not None and len(c[0])==40 else '-'
    same=all(p['join_clients']==q['join_clients'] for p,q in zip(a[1],b[1]))
    out.append(f"|{ds}|{s}|{A[30:].mean():.2f}|{B[30:].mean():.2f}|{dd:+.2f}|{d1}|{A[4:].min():.1f}|{B[4:].min():.1f}|{B[19]:.1f}/{B[39]:.1f}|참여순서 {'동일' if same else '다름'}|")
open(D+'/../RESULTS_damped11.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
