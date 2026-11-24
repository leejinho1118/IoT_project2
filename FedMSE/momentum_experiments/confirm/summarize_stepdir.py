import json,os,numpy as np
D=os.path.dirname(os.path.abspath(__file__))+'/runs/mse'
cells=[('N-BaIoT',1234,'D0_mse_par',5),('N-BaIoT',2024,'D0_s2024_lr1e-5',5),('N-BaIoT',4321,'D0_s4321_lr1e-5',5),('N-BaIoT',7777,'D0_s7777_lr1e-5',5),
('ToN',1234,'D0_ton_s1234',4),('ToN',2024,'D0_ton_s2024',4),('ToN',4321,'D0_ton_s4321',4),('CIC-km',1234,'D0_cic_km_s1234',5),('CIC-km',4321,'D0_cic_km_s4321',5)]
out=['# 클라이언트 방향 일치 대리지표 E (D0, 탐색·사후; CRITERIA_stepdir.md)','','E=‖평균 델타‖/평균‖개별 델타‖. 직교 기준=1/√n(n=참가자). 클라이언트 델타 자체는 로그에 없음.','',
'|데이터셋|시드|런수|n|직교기준|E 중앙 전체|R1~5|R31~40|E/기준(전체)|패턴|','|---|---|---|---|---|---|---|---|---|---|']
for ds,s,c,n in cells:
    d=f'{D}/{c}';E=[]
    for f in os.listdir(d):
        if f.startswith('Run_'):
            L=[list(json.loads(l).values())[0] for l in open(f'{d}/{f}')]
            if len(L)==40: E.append([x['diag2']['E'] for x in L])
    E=np.array(E);ref=n**-.5;m=np.median(E)
    pat='정렬 높음(≥0.8)' if m>=.8 else ('직교 근처(≤1.3×기준)' if m<=1.3*ref else '중간')
    out.append(f"|{ds}|{s}|{len(E)}|{n}|{ref:.2f}|{m:.2f}|{np.median(E[:,:5]):.2f}|{np.median(E[:,30:]):.2f}|{m/ref:.2f}|{pat}|")
open(D+'/../../RESULTS_stepdir.md','w').write('\n'.join(out)+'\n');print('\n'.join(out))
