"""개발 전용 PCA 축으로 A/B 정상 분포를 나누고 보류 평가를 격리한다."""
from pathlib import Path
import sys,json,gzip,pickle,hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'continual'))
from protocol import fingerprint,check_partition,verify_candidates,fixed_budget_mix


class BoundedStandardScaler(StandardScaler):
    """초기 A 표준화 뒤 고정 [-20,20] 포화. 미래 자료로 경계를 적합하지 않는다."""
    def transform(self,X,copy=None):
        with np.errstate(over='ignore',invalid='ignore'):
            scaled=super().transform(X,copy=copy)
        if np.isnan(scaled).any():raise ValueError('NaN standardized feature')
        return np.clip(scaled,-20.,20.)


def matrix(rows):
    return np.asarray([r['x'] for r in rows],dtype=np.float64).reshape(-1,115)


def signed_log(x):
    return np.sign(x)*np.log1p(np.abs(x))


def uncertainty_verdict(row,rate):
    # 분포 점수나 공격 탐지 결과에 의존하지 않는 사전 고정 ID 기반 보류.
    if int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)%10000<int(rate*10000):
        return 'uncertain'
    return 'normal' if row['y']==0 else 'attack'


def verified_rows(items,rate):
    lookup={r['id']:r for r in items}
    return verify_candidates(items,lambda rid:uncertainty_verdict(lookup[rid],rate))


def replay_mix(new,bank,capacity,seed):
    if capacity==0:return list(new),0
    return fixed_budget_mix(new,bank[:capacity],.25,np.random.default_rng(seed))


def load_plan(config,out):
    rng=np.random.default_rng(config['data_seed']);seen=set();raw={};removed=[];source_hashes={}
    for cid in range(1,config['clients']+1):
        pools={'normal':[],'attack':[]}
        for source in ['normal','test_normal','abnormal']:
            p=ROOT/'Data/N-BaIoT/IID-50-Client_Data'/f'Client-{cid}'/source/'data.csv'
            source_hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
            x=pd.read_csv(p,header=None).to_numpy(dtype=np.float64)
            if x.shape[1]!=115 or not np.isfinite(x).all():raise ValueError('Invalid CSV')
            indices=rng.permutation(len(x))
            if config['source_cap'] is not None:indices=indices[:config['source_cap']]
            for idx in indices:
                fp=fingerprint(x[idx]);rid=f'Client-{cid}/{source}/{idx}'
                if fp in seen:removed.append(rid);continue
                seen.add(fp);pools['attack' if source=='abnormal' else 'normal'].append(dict(id=rid,fingerprint=fp,client=cid,source=source,y=int(source=='abnormal'),x=x[idx]))
        rng.shuffle(pools['normal']);rng.shuffle(pools['attack']);normal=pools['normal'];n=len(normal);a=int(.2*n);b=int(.4*n)
        raw[cid]=dict(scenario=normal[:a],future=normal[a:b],model=normal[b:],attack=pools['attack'])
    # 시나리오 기준은 scenario 정상만 사용한다. 최종 보류 특징·정답은 fit에 사용하지 않는다.
    scenario=[r for pools in raw.values() for r in pools['scenario']]
    scaler=StandardScaler().fit(signed_log(matrix(scenario)))
    pca=PCA(n_components=1,svd_solver='full').fit(scaler.transform(signed_log(matrix(scenario))))
    if pca.components_[0,np.argmax(np.abs(pca.components_[0]))]<0:pca.components_*=-1
    def project(rows):return pca.transform(scaler.transform(signed_log(matrix(rows)))).ravel()
    cut=float(np.median(project(scenario)));groups={};clients=[];audit=[]
    for cid,pools in raw.items():
        score=project(pools['model']);a=[r for r,v in zip(pools['model'],score) if v<=cut];b=[r for r,v in zip(pools['model'],score) if v>cut]
        future_score=project(pools['future']);fa=[r for r,v in zip(pools['future'],future_score) if v<=cut];fb=[r for r,v in zip(pools['future'],future_score) if v>cut]
        # A는 초기 학습·검증·서버 개발에, B는 후보와 로컬 검증에만 배정한다.
        i=int(.5*len(a));j=int(.7*len(a));k=int(.75*len(b));attack=pools['attack'];q=int(.25*len(attack));v=q+int((len(attack)-q)/2)
        c=dict(id=cid,train=a[:i],valid=a[i:j],dev=a[j:],new_valid=b[k:],old_normal=fa,new_normal=fb,old_attack=attack[q:v],new_attack=attack[v:])
        bn=b[:k];ba=attack[:q]
        ns=[list(x) for x in np.array_split(np.asarray(bn,dtype=object),config['stages'])];ats=[list(x) for x in np.array_split(np.asarray(ba,dtype=object),config['stages'])]
        c['candidate_stages']=[]
        for normal,attack_stage in zip(ns,ats):
            rows=normal+attack_stage;rng.shuffle(rows);c['candidate_stages'].append(rows)
        for key in ['train','valid','dev','new_valid','old_normal','new_normal','old_attack','new_attack']:
            if len(c[key])<2:raise ValueError(f'Insufficient {cid}/{key}; scenario not altered to force an effect')
            groups[f'{cid}/{key}']=c[key]
        groups[f'{cid}/scenario']=pools['scenario']
        for stage,rows in enumerate(c['candidate_stages']):groups[f'{cid}/candidate_{stage}']=rows
        c['scaler']=BoundedStandardScaler().fit(matrix(c['train']));c['replay']=c['train'][:config['max_capacity']]
        if len(c['replay'])<config['max_capacity']:raise ValueError(f'Capacity not available in client {cid}')
        # 테스트에 의존하지 않는 유입/개발 분리도만 기록한다.
        audit.append(dict(client=cid,initial_a=len(a),arrival_b=len(b),projection_a_mean=float(np.mean(score[score<=cut])),projection_b_mean=float(np.mean(score[score>cut])),train=len(c['train']),arrivals=[len(x) for x in ns]))
        clients.append(c)
    counts=check_partition(groups)
    with gzip.open(out/'split_manifest.json.gz','wt',encoding='utf-8') as f:json.dump({k:[{a:b for a,b in r.items() if a!='x'} for r in rows] for k,rows in groups.items()},f)
    with (out/'scenario_mapper.pkl').open('wb') as f:pickle.dump(dict(scaler=scaler,pca=pca,cut=cut,source_ids=[r['id'] for r in scenario],transform='signed_log1p -> StandardScaler -> PC1'),f)
    result=dict(counts=counts,total_unique_rows=sum(counts.values()),duplicates_removed=len(removed),removed_ids=removed,source_csv_sha256=source_hashes,scenario='Development-only signed-log PCA median A/B; simulated feature-distribution change, not chronological drift',scenario_fit_ids=[r['id'] for r in scenario],scenario_cut=cut,scenario_explained_variance=float(pca.explained_variance_ratio_[0]),clients=audit,evaluation_policy='Holdout reserved before fitting mapper; reused across stages, never used to tune; same underlying dataset as V1, not external replication')
    (out/'split_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return clients
