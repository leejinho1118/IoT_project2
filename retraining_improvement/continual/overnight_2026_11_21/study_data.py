"""Load prepared development roles separately from a locked final NPZ."""
from pathlib import Path
import json,gzip,numpy as np,hashlib,datetime

def load_development(data):
    data=Path(data)
    with gzip.open(data/'roles.json.gz','rt',encoding='utf-8') as f:roles=json.load(f)
    ids=sorted({int(k.split('/')[0]) for k in roles})
    clients=[]
    with np.load(data/'development.npz',allow_pickle=False) as arrays:
        for cid in ids:
            c={'id':cid,'_data_directory':str(data),'_roles_metadata':roles,'candidate_stages':[None,None,None]}
            for key in ['train','valid','dev','calib_old','decision_old','decision_attack']:
                meta=roles[f'{cid}/{key}'];a=arrays[f'{cid}__{key}'];assert len(a)==len(meta)
                c[key]=[dict(r,x=np.asarray(x,dtype=np.float64)) for r,x in zip(meta,a)]
            c['device']=c['train'][0].get('device',f'Virtual-client-{cid}')
            for name in ['new_valid','calib_new','decision_new']:c[name]=[]
            clients.append(c)
    return clients,roles

def activate(clients,stage):
    # Initial loading does not decompress any future-stage feature arrays.
    for c in clients:
        if stage==0:
            for name in ['new_valid','calib_new','decision_new']:c[name]=[]
            continue
        cid=c['id'];meta=c['_roles_metadata'];data=Path(c['_data_directory'])
        with np.load(data/'development.npz',allow_pickle=False) as arrays:
            for name in ['candidate','new_valid','calib_new','decision_new']:
                key=f'{name}_{stage-1}';a=arrays[f'{cid}__{key}'];rows=meta[f'{cid}/{key}']
                assert len(a)==len(rows)
                materialized=[dict(r,x=np.asarray(x,dtype=np.float64)) for r,x in zip(rows,a)]
                if name=='candidate':c['candidate_stages'][stage-1]=materialized
                else:c[name]=materialized
        c['_released_stage']=stage

def load_final(data,freeze_file,access_log):
    data=Path(data);freeze_file=Path(freeze_file)
    if not freeze_file.exists():raise RuntimeError('Final evaluation requires a frozen study selection')
    with gzip.open(data/'roles.json.gz','rt',encoding='utf-8') as f:roles=json.load(f)
    result={}
    access_log=Path(access_log)
    if access_log.exists():raise RuntimeError('Existing final access record preserved')
    access_log.write_text(json.dumps(dict(opened_at=datetime.datetime.now().astimezone().isoformat(),selection_sha256=hashlib.sha256(freeze_file.read_bytes()).hexdigest(),file=str(data/'final.npz'),scope='Final features opened only after study selection was frozen'),indent=2),encoding='utf-8')
    with np.load(data/'final.npz',allow_pickle=False) as arrays:
        for role,meta in roles.items():
            if role.split('/')[1].startswith('final_'):
                a=arrays[role.replace('/','__')];assert len(a)==len(meta)
                result[role]=[dict(r,x=np.asarray(x,dtype=np.float64)) for r,x in zip(meta,a)]
    return result
