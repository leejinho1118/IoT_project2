import json, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); D=HERE+'/runs'
def load(c):
    return [list(json.loads(l).values())[0] for l in open(f'{D}/mse_m/{c}/Run_10.jsonl')][:40]
auc=lambda L:np.array([np.mean(x['global_auc'])*100 for x in L])
def st(v):
    v=np.array(v,float)
    if len(v)==0: return '해당 라운드 없음'
    q=np.percentile(v,[25,50,75]); return f'n={len(v)}, 평균 {v.mean():+.3f}, 사분위 {q[0]:+.3f}/{q[1]:+.3f}/{q[2]:+.3f}, 음수 {int((v<0).sum())}/{len(v)}({(v<0).mean()*100:.0f}%)'
out=['# D1 재실행 + cos(Δ_t, v_{t-1}) (data_seed 1234 run10 1런 40라운드, β=0.9 η=1; n=1)','']
for ds,new,old in (('ToN','O5c_ton_s1234','D1_ton_s1234'),('N-BaIoT','O5c_nbaiot_s1234','D1_msem_par')):
    L=load(new); O=load(old); a=auc(L); ao=auc(O)
    mx=np.abs(a-ao).max(); same=bool(np.allclose(a,ao,atol=1e-9))
    out+=[f'## {ds}',f'- 재현: 새 40라운드 AUC와 기존 D1 run10 AUC 최대 차이 {mx:.4f}%p → '+('완전 일치' if same else '불일치(아래 분석은 새 런 기준)')]
    cos=[x['guard']['cos'] for x in L]; dn=[x['guard']['delta_norm'] for x in L]; vn=[x['guard']['vprev_norm'] for x in L]
    drop=[r for r in range(1,40) if a[r]-a[r-1]<=-2]; non=[r for r in range(1,40) if r not in drop]
    out+=[f'- 낙하 라운드(AUC가 전 라운드 대비 2%p 이상 하락, R2~R40): {len(drop)}개 {[r+1 for r in drop]}',
          f'- 낙하 라운드의 cos(해당 라운드 t): {st([cos[r] for r in drop])}',
          f'- 낙하 아닌 라운드의 cos: {st([cos[r] for r in non])}',
          f'- 낙하 직전 라운드(t-1)의 cos: {st([cos[r-1] for r in drop if r-1>=1])}  (R1 직후는 cos 없음)']
    caught=sum(cos[r]<0 for r in drop); fa=sum(cos[r]<0 for r in non)
    c2=sum(cos[r-1]<0 for r in drop if r-1>=1); fa2=sum(cos[r-1]<0 for r in non if r-1>=1)
    out+=[f'- 규칙 "cos<0이면 리셋" 계산만: (같은 라운드 t의 cos 사용) 낙하 {caught}/{len(drop)}개에서 cos<0, 낙하 아닌 {fa}/{len(non)}개에서도 cos<0(잘못 잡음)',
          f'- (직전 라운드 t-1의 cos로 낙하 t를 예측) 낙하 {c2}/{len([r for r in drop if r>=2])}개 포착, 낙하 아닌 {fa2}/{len([r for r in non if r>=2])}개 오경보',
          f'- 참고: cos<0 라운드 전체 {sum(c is not None and c<0 for c in cos)}/39, ‖Δ‖ 중앙값 {np.median(dn):.3f}, ‖v_prev‖ 중앙값 {np.median(vn[1:]):.3f}','',
          '|R|AUC|cos(Δ_t,v_{t-1})|‖Δ_t‖|‖v_{t-1}‖|','|---|---|---|---|---|']
    for r in range(40): out.append(f"|{r+1}|{a[r]:.1f}|{'-' if cos[r] is None else f'{cos[r]:+.3f}'}|{dn[r]:.3f}|{vn[r]:.3f}|")
    out.append('')
open(HERE+'/RESULTS_cos.md','w').write('\n'.join(out)+'\n'); print('\n'.join(l for l in out if not l.startswith('|')))
