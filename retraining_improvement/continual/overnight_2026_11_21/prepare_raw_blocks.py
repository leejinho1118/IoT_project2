"""Prepare device/type-preserving raw N-BaIoT blocks; no model fitting or test metrics."""
from pathlib import Path
import json,csv,gzip,hashlib,datetime,collections,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];RAW=ROOT/'external_data/n_baiot_raw';SOURCE=RAW/'unpacked_v1';OUT=ROOT/'continual/overnight_2026_11_21/raw_blocks_v1'
if OUT.exists():raise SystemExit('Existing preparation preserved')
OUT.mkdir();started=time.perf_counter()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def status(**kw):save('status.json',dict(updated=datetime.datetime.now().astimezone().isoformat(),**kw))
def fp(x):
 a=np.asarray(x,dtype='<f8').copy();a[a==0]=0;return hashlib.sha256(a.tobytes()).hexdigest()
def coarse(x):
 a=np.round(np.sign(x)*np.log1p(np.abs(x)),6).astype('<f8');a[a==0]=0;return hashlib.sha256(a.tobytes()).hexdigest()
def count_hash(path):
 h=hashlib.sha256();n=0;last=b''
 with path.open('rb') as f:
  while True:
   block=f.read(8*1024*1024)
   if not block:break
   h.update(block);n+=block.count(b'\n');last=block[-1:]
 return n+(last!=b'\n')-1,h.hexdigest()
normal_defs=[('train',0.,.20,1024),('valid',.22,.26,256),('dev',.28,.32,256),('calib_old',.34,.38,512),('decision_old',.40,.44,256),('final_old_normal',.46,.50,512),('candidate_0',.52,.58,512),('candidate_1',.60,.66,512),('candidate_2',.68,.74,512),('calib_new',.76,.80,512),('decision_new',.82,.86,256),('final_new_normal',.90,.98,1024)]
gaf_defs=[('decision_attack',.02,.20,256),('candidate_0',.30,.42,64),('candidate_1',.44,.56,64),('candidate_2',.58,.70,64),('final_old_attack',.74,.98,512)]
mirai_defs=[('final_new_attack',.05,.95,512)]
save('protocol_frozen.json',dict(author='20222038 이진호',created=datetime.datetime.now().astimezone().isoformat(),normal_blocks=normal_defs,gafgyt_blocks=gaf_defs,mirai_blocks=mirai_defs,sample_seed=4422026,oversample_multiplier=4,feature_count=115,original_row_order='File-row proxy; no timestamp column is asserted',attack_type_holdout='Mirai features/labels are reserved for final evaluation; development selection uses Gafgyt only',prior_prepared_filter='Exclude exact canonical float64 SHA-256 and six-decimal signed-log fingerprints matching earlier IID50 prepared data',cross_role_filter='Training-first priority, then development/calibration/candidates; final last. Exact and coarse fingerprints must be disjoint.',coarse_filter_limits='Conservative approximate duplicate exclusion, not proof all near duplicates or temporal dependence eliminated',source_license='UCI CC BY 4.0; DOI 10.24432/C5RC8J'))
status(state='hashing_prior_prepared')
prior={};priorq=set();prior_count=0
for path in sorted((ROOT/'Data/N-BaIoT/IID-50-Client_Data').glob('Client-*/*/data.csv')):
 y=0 if path.parent.name in ['normal','test_normal'] else 1
 a=pd.read_csv(path,header=None).to_numpy(dtype=np.float64)
 for x in a:
  prior[fp(x)]=y;priorq.add(coarse(x));prior_count+=1
save('prior_prepared_audit.json',dict(rows=prior_count,unique_exact=len(prior),unique_coarse=len(priorq),definition='pandas training runtime float64; signed-log round 6 decimals for coarse exclusion'))
inventory=json.loads((RAW/'csv_inventory.json').read_text(encoding='utf-8'))
excluded=[x for x in inventory if x['family']=='unknown' or '.csv' in x['device']]
entries=[x for x in inventory if x not in excluded];devices=sorted({x['device'] for x in entries});assert len(devices)==9
headers=entries[0]['header'];assert len(headers)==115 and len(set(headers))==115
assert all(x['header']==headers for x in entries),'Raw feature header/order mismatch'
device_id={name:i+1 for i,name in enumerate(devices)}
buckets=collections.defaultdict(list);source_info=[];audit=collections.Counter();old_overlap=[]
for fi,entry in enumerate(entries):
 path=SOURCE/entry['path'];n,h=count_hash(path);assert n>0
 source_info.append(dict(path=entry['path'],device=entry['device'],family=entry['family'],attack_type=entry['attack_type'],rows=n,bytes=entry['bytes'],sha256=h))
 defs=normal_defs if entry['family']=='benign' else gaf_defs if entry['family']=='gafgyt' else mirai_defs
 cid=device_id[entry['device']];selected={};caps={}
 for role,lo,hi,cap in defs:
  left=int(lo*n);right=int(hi*n);amount=min(max(0,right-left),cap*4)
  if not amount:continue
  # A deterministic label-independent random sample within a prespecified source block.
  seed=int(hashlib.sha256((entry['path']+'/'+role+'/4422026').encode()).hexdigest()[:16],16)
  indexes=np.sort(np.random.default_rng(seed).choice(np.arange(left,right),amount,replace=False))
  selected[role]=indexes;caps[role]=cap
 offset=0;status(state='reading_raw',file=entry['path'],files_done=fi,total_files=len(entries))
 for chunk in pd.read_csv(path,chunksize=16384,skip_blank_lines=False):
  end=offset+len(chunk);windows={role:idx[(idx>=offset)&(idx<end)]-offset for role,idx in selected.items()}
  for role,idx in windows.items():
   if not len(idx):continue
   x=chunk.iloc[idx].to_numpy(dtype=np.float64)
   for local,row in zip(idx,x):
    audit['sampled_rows']+=1
    if not np.isfinite(row).all():audit['nonfinite_excluded']+=1;continue
    exact=fp(row);q=coarse(row);y=0 if entry['family']=='benign' else 1
    if exact in prior:
     audit['prior_exact_excluded']+=1
     if prior[exact]!=y:audit['prior_label_disagreement']+=1
     if len(old_overlap)<30:old_overlap.append(dict(source=entry['path'],row=int(offset+local),old_label=prior[exact],raw_label=y))
     continue
    if q in priorq:audit['prior_coarse_excluded']+=1;continue
    rid=f"{entry['path'].replace(chr(92),'/')}::{int(offset+local)}"
    meta=dict(id=rid,client=cid,device=entry['device'],source=entry['path'],row=int(offset+local),family=entry['family'],attack_type=entry['attack_type'],y=y,fingerprint=exact,coarse_fingerprint=q)
    priority=int(hashlib.sha256((rid+'/priority').encode()).hexdigest()[:16],16)
    buckets[(role,cid,entry['path'])].append((priority,meta,row.copy(),caps[role]))
  offset=end
 assert offset==n,('Physical/data row count mismatch',entry['path'],offset,n)
roles={};arrays={};seen=set();seenq=set();counts=[]
order=['train','valid','dev','calib_old','calib_new','decision_old','decision_new','decision_attack','candidate_0','candidate_1','candidate_2','final_old_normal','final_old_attack','final_new_normal','final_new_attack']
for role in order:
 for cid in range(1,10):
  selected=[];vectors=[]
  for key in sorted([k for k in buckets if k[0]==role and k[1]==cid]):
   kept=0
   for priority,m,x,cap in sorted(buckets[key],key=lambda z:z[0]):
    if kept>=cap:break
    if m['fingerprint'] in seen or m['coarse_fingerprint'] in seenq:audit['cross_role_or_within_role_duplicate_excluded']+=1;continue
    seen.add(m['fingerprint']);seenq.add(m['coarse_fingerprint']);selected.append(m);vectors.append(x);kept+=1
  name=f'{cid}/{role}';roles[name]=selected;arrays[name]=np.asarray(vectors,dtype=np.float64).reshape(-1,115)
  counts.append(dict(client=cid,device=devices[cid-1],role=role,rows=len(selected),normal=sum(m['y']==0 for m in selected),attack=sum(m['y']!=0 for m in selected)))
for cid in range(1,10):
 for role in ['train','valid','dev','calib_old','calib_new','decision_old','decision_new','final_old_normal','final_new_normal']:
  minimum=128 if role=='train' else 32
  assert len(roles[f'{cid}/{role}'])>=minimum,('Insufficient integrity-filtered normal rows',cid,role,len(roles[f'{cid}/{role}']))
 for stage in range(3):assert sum(m['y']==0 for m in roles[f'{cid}/candidate_{stage}'])>=32
 assert len(roles[f'{cid}/decision_attack'])>=2 and len(roles[f'{cid}/final_old_attack'])>=2
new_devices=[cid for cid in range(1,10) if len(roles[f'{cid}/final_new_attack'])>=2]
assert len(new_devices)>=2
with gzip.open(OUT/'roles.json.gz','wt',encoding='utf-8') as f:json.dump(roles,f)
np.savez_compressed(OUT/'development.npz',**{k.replace('/','__'):v for k,v in arrays.items() if not k.split('/')[1].startswith('final_')})
np.savez_compressed(OUT/'final.npz',**{k.replace('/','__'):v for k,v in arrays.items() if k.split('/')[1].startswith('final_')})
save('source_files.json',source_info);save('feature_names.json',headers);save('devices.json',device_id);save('role_counts.json',counts)
save('preparation_audit.json',dict(state='passed',source_files=len(entries),metadata_files_excluded=[x['path'] for x in excluded],devices=devices,new_mirai_evaluation_clients=new_devices,audit=dict(audit),prior_exact_match_examples=old_overlap,rows_kept=sum(len(x) for x in roles.values()),all_exact_and_coarse_fingerprints_disjoint=True,development_npz_sha256=hashlib.file_digest((OUT/'development.npz').open('rb'),'sha256').hexdigest(),final_npz_sha256=hashlib.file_digest((OUT/'final.npz').open('rb'),'sha256').hexdigest(),seconds=time.perf_counter()-started,limits='Same public dataset/captures as the source of earlier prepared CSV; selected exact/coarse overlaps removed, not an independent new capture or external dataset. Raw row order is not verified wall-clock chronology.'))
status(state='complete',devices=devices,new_mirai_evaluation_clients=new_devices,rows_kept=sum(len(x) for x in roles.values()),seconds=time.perf_counter()-started)
print(json.dumps(dict(state='complete',devices=9,new_mirai_clients=len(new_devices),rows_kept=sum(len(x) for x in roles.values()),audit=dict(audit)),indent=2))
