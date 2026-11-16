"""A~E 공통 개발 분할. 최종 보류 행은 E 선택이 고정되기 전 materialize하지 않는다."""
from pathlib import Path
import sys,json,gzip,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'continual/drift_v2'))
import backend as B
import scenario  # BoundedStandardScaler pickle 경로
sys.path.insert(0,str(ROOT/'continual'))
from protocol import check_partition,fingerprint


def split_validation(rows):
    # 초기 학습·입력 scaler는 기존 A 초기 모델을 공통 사용한다.
    # 후속 로컬 조기 종료 / 임계값 / 방향 선택 자료는 서로 분리한다.
    i=max(2,int(.4*len(rows)));j=i+max(2,int(.3*len(rows)))
    if len(rows)-j<2:raise ValueError('Not enough validation data')
    return rows[:i],rows[i:j],rows[j:]


def load_development(source,out,config):
    with gzip.open(source/'split_manifest.json.gz','rt',encoding='utf-8') as f:manifest=json.load(f)
    cache={};groups={};clients=[];final_meta={};source_csv={}
    def hydrate(rows):
        items=[]
        for row in rows:
            key=(row['client'],row['source'])
            if key not in cache:
                p=ROOT/'Data/N-BaIoT/IID-50-Client_Data'/f'Client-{key[0]}'/key[1]/'data.csv'
                cache[key]=pd.read_csv(p,header=None).to_numpy(dtype=np.float64)
                source_csv[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
            x=cache[key][int(row['id'].split('/')[-1])]
            if fingerprint(x)!=row['fingerprint']:raise ValueError('Source feature mismatch')
            items.append(dict(row,x=x))
        return items
    for cid in range(1,config['clients']+1):
        c={'id':cid}
        for key in ['train','valid','dev','new_valid']:c[key]=hydrate(manifest[f'{cid}/{key}'])
        c['valid'],c['calib_old'],c['decision_old']=split_validation(c['valid'])
        c['new_valid'],c['calib_new'],c['decision_new']=split_validation(c['new_valid'])
        c['candidate_stages']=[];c['decision_attack']=[]
        for stage in range(config['stages']):
            items=hydrate(manifest[f'{cid}/candidate_{stage}']);att=[r for r in items if r['y']==1]
            k=max(1,int(.2*len(att)));held={r['id'] for r in att[:k]};c['decision_attack']+=att[:k]
            c['candidate_stages'].append([r for r in items if r['id'] not in held])
        for key in ['train','valid','dev','new_valid','calib_old','calib_new','decision_old','decision_new','decision_attack']:groups[f'{cid}/{key}']=c[key]
        for stage,rows in enumerate(c['candidate_stages']):groups[f'{cid}/candidate_{stage}']=rows
        for key in ['old_normal','new_normal','old_attack','new_attack']:
            # ID/라벨/해시만 보존한다. 이 함수에서 최종 특징은 추출하지 않는다.
            final_meta[f'{cid}/{key}']=manifest[f'{cid}/{key}']
        clients.append(c)
    audit_groups={**groups,**final_meta};counts=check_partition(audit_groups)
    with gzip.open(out/'partition_manifest.json.gz','wt',encoding='utf-8') as f:json.dump({k:B.metadata(v) for k,v in audit_groups.items()},f)
    B.atomic(out/'data_audit.json',dict(counts=counts,source_csv_sha256=source_csv,source=str(source),final_feature_rows_materialized=0,limits='Development rows reuse initial-model early-stop provenance; final holdouts were seen in the earlier V2 study. This is exploratory adaptive follow-up, not an untouched external replication.'))
    return clients,final_meta


def load_final_rows(final_meta,selection_file):
    # 최종 평가를 여는 유일한 경계. 선택 파일이 없으면 실행 불가.
    if not selection_file.exists():raise RuntimeError('Selection must be frozen before final features')
    cache={};rows={}
    for role,meta in final_meta.items():
        rs=[]
        for row in meta:
            key=(row['client'],row['source'])
            if key not in cache:cache[key]=pd.read_csv(ROOT/'Data/N-BaIoT/IID-50-Client_Data'/f'Client-{key[0]}'/key[1]/'data.csv',header=None).to_numpy(dtype=np.float64)
            x=cache[key][int(row['id'].split('/')[-1])]
            if fingerprint(x)!=row['fingerprint']:raise ValueError('Final feature changed')
            rs.append(dict(row,x=x))
        rows[role]=rs
    return rows
