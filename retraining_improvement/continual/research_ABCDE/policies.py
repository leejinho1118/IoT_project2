"""리플레이 비율·다양성·채택 게이트. 최종 평가에 의존하지 않는 규칙."""
import math
import numpy as np


def memory_bank(rows,scaler,capacity,mode):
    if capacity>len(rows):raise ValueError('Requested memory exceeds old normal data')
    if mode=='uniform':return list(rows[:capacity])
    if mode!='diverse':raise ValueError('Unknown memory mode')
    x=scaler.transform(np.asarray([r['x'] for r in rows]));center=np.median(x,axis=0)
    first=int(np.argmin(np.sum((x-center)**2,axis=1)));chosen=[first];distance=np.sum((x-x[first])**2,axis=1);distance[first]=-1
    while len(chosen)<capacity:
        index=int(np.argmax(distance));chosen.append(index);distance=np.minimum(distance,np.sum((x-x[index])**2,axis=1));distance[chosen]=-1
    return [rows[i] for i in chosen]


def mix(rows,bank,ratio,seed):
    if not rows:raise ValueError('No verified normals')
    n=len(rows);k=min(int(n*ratio),len(bank),n-1);rng=np.random.default_rng(seed)
    result=[rows[int(i)] for i in rng.choice(n,n-k,replace=False)]+[bank[int(i)] for i in rng.choice(len(bank),k,replace=False)]
    rng.shuffle(result)
    if len({r['id'] for r in result})!=n or any(r['y']!=0 for r in result):raise ValueError('Contaminated or duplicate training input')
    return result,k


def development_score(metrics):
    # 정상/공격이 모두 있는 개발 자료만 사용한다. 최종 결과를 방향 선택에 넣지 않는다.
    return .5*(metrics['old']['mean_client_auc']+metrics['new']['mean_client_auc'])-.1*max(0.,max(metrics['old']['fpr'],metrics['new']['fpr'])-.1)


def adaptive_ratio(previous,initial,current):
    if current is None:return previous
    old=current['old']/max(initial['old'],1e-12);new=current['new']/max(initial['new'],1e-12)
    if old>1.25:return min(.75,previous+.25)
    if new>=.9 or old<=1.10:return max(.125,previous-.125)
    return previous


def gate_accept(candidate,previous,initial):
    # 기존 영역 AUC 3%p 저하 한계 + 균형 개발 점수 개선을 모두 요구한다.
    old_ok=candidate['old']['mean_client_auc']>=initial['old']['mean_client_auc']-.03
    gain=development_score(candidate)-development_score(previous)
    return bool(old_ok and gain>=1e-4),dict(old_auc_floor=initial['old']['mean_client_auc']-.03,old_floor_pass=bool(old_ok),development_score_gain=gain,required_gain=1e-4)


def choose_best(analyses):
    # 같은 개발 지표에서 동률이면 더 짧은 추가 학습 시간, 그 뒤 이름 순서로 결정.
    return sorted(analyses,key=lambda x:(-x['development_score_mean'],x['training_seconds_mean'],x['round']))[0]['round']
