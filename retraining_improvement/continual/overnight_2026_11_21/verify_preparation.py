"""Independent split-integrity checks without opening the locked final NPZ."""
from pathlib import Path
import json,gzip,hashlib,collections,csv
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def fp(x):
 a=np.asarray(x,dtype='<f8').copy();a[a==0]=0;return hashlib.sha256(a.tobytes()).hexdigest()
def coarse(x):
 a=np.round(np.sign(x)*np.log1p(np.abs(x)),6).astype('<f8');a[a==0]=0;return hashlib.sha256(a.tobytes()).hexdigest()
reports=[]
for name in ['raw_blocks_v2','rt_thing_to_mqtt_v1','rt_mqtt_to_thing_v1']:
 data=H/name;protocol=json.loads((data/'protocol_frozen.json').read_text(encoding='utf-8-sig'));audit=json.loads((data/'preparation_audit.json').read_text(encoding='utf-8-sig'))
 assert hashlib.file_digest((data/'development.npz').open('rb'),'sha256').hexdigest()==audit['development_npz_sha256']
 assert hashlib.file_digest((data/'final.npz').open('rb'),'sha256').hexdigest()==audit['final_npz_sha256']
 with gzip.open(data/'roles.json.gz','rt',encoding='utf-8') as f:roles=json.load(f)
 seen_ids={};owners=collections.defaultdict(list);qowners=collections.defaultdict(list);development_checks=0;final_conflicts=0
 for role,rows in roles.items():
  cid,key=role.split('/');assert int(cid)>0
  for row in rows:
   assert row['id'] not in seen_ids,('Duplicate source row ID',name,row['id']);seen_ids[row['id']]=role;owners[row['fingerprint']].append((role,row))
   if row.get('coarse_fingerprint'):qowners[row['coarse_fingerprint']].append(role)
   if row.get('label_conflict'):assert key.startswith('final_');final_conflicts+=1
   if row['family'] in ['mirai','heldout_attack_type']:assert key.startswith('final_'),('Unseen attack entered development',row['id'])
   if key in ['train','valid','dev','calib_old','decision_old'] or key.startswith(('new_valid_','calib_new_','decision_new_')):assert row['y']==0
 for h,groups in owners.items():
  if len(groups)>1:assert all(role.split('/')[1].startswith('final_') and row.get('label_conflict') for role,row in groups),('Feature cross-role leakage',name,h)
 for h,groups in qowners.items():assert len(groups)==1,('Coarse feature role collision',name,h)
 with np.load(data/'development.npz',allow_pickle=False) as arrays:
  for role,rows in roles.items():
   if role.split('/')[1].startswith('final_'):continue
   a=arrays[role.replace('/','__')];assert len(a)==len(rows) and a.shape[1]==(115 if name=='raw_blocks_v2' else 81)
   for r,x in zip(rows,a):
    assert np.isfinite(x).all() and fp(x)==r['fingerprint']
    if r.get('coarse_fingerprint'):assert coarse(x)==r['coarse_fingerprint']
    development_checks+=1
 report=dict(state='passed',dataset=name,source_row_IDs=len(seen_ids),development_feature_rows_recomputed=development_checks,locked_final_npz_opened=False,all_stage_roles_disjoint=True,heldout_attack_types_final_only=True,retained_label_conflict_final_rows=final_conflicts)
 if name=='raw_blocks_v2':
  files=json.loads((data/'source_files.json').read_text(encoding='utf-8-sig'));raw=ROOT/'external_data/n_baiot_raw/unpacked_v1'
  for file in files:assert hashlib.file_digest((raw/file['path']).open('rb'),'sha256').hexdigest()==file['sha256']
  report['official_source_csv_hashes_verified']=len(files)
 else:
  source=ROOT/protocol['source'];assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==protocol['source_sha256']
  d=pd.read_csv(source);cols=protocol['feature_names'];assert 'Unnamed: 0' not in cols and 'Attack_type' not in cols and 'proto' not in cols and 'service' not in cols
  source_values=d[cols].to_numpy(dtype=np.float64)
  with np.load(data/'development.npz',allow_pickle=False) as arrays:
   for role,rows in roles.items():
    if role.split('/')[1].startswith('final_'):continue
    a=arrays[role.replace('/','__')]
    if len(rows):assert np.array_equal(a,source_values[[r['row'] for r in rows]])
  report['raw_source_row_values_checked']=True
 (data/'independent_preparation_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');reports.append(report);print(report,flush=True)
(H/'preparation_validation.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
