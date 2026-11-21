"""Prospective RT-IoT2022 normal-pattern shift splits, two fixed orderings."""
from pathlib import Path
import json,gzip,hashlib,datetime,collections
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];HOME=Path(__file__).resolve().parent;SOURCE=ROOT/'external_data/rt_iot2022/unpacked_v1/RT_IOT2022'
df=pd.read_csv(SOURCE);normal_types={'Thing_Speak','MQTT_Publish','Wipro_bulb'};known_attacks={'DOS_SYN_Hping','ARP_poisioning'}
columns=[c for c in df.select_dtypes(include='number').columns if c!='Unnamed: 0'];assert len(columns)==81
x=df[columns].to_numpy(dtype=np.float64);assert np.isfinite(x).all()
y=np.array([0 if s in normal_types else 1 for s in df.Attack_type]);fingerprints=[];labelsets=collections.defaultdict(set)
for a,label in zip(x,y):
 a=a.astype('<f8',copy=True);a[a==0]=0;h=hashlib.sha256(a.tobytes()).hexdigest();fingerprints.append(h);labelsets[h].add(int(label))
ambiguous={h for h,labels in labelsets.items() if len(labels)>1};amb_indices=[i for i,h in enumerate(fingerprints) if h in ambiguous]
old_defs=[('train',0.,.35,2048),('valid',.37,.44,512),('dev',.46,.53,512),('calib_old',.55,.63,512),('decision_old',.65,.73,512),('final_old_normal',.80,.999,1536)]
new_defs=[]
for st,(lo,hi) in enumerate([(0.,.26),(.28,.54),(.56,.82)]):
 span=hi-lo;new_defs += [(f'candidate_{st}',lo,lo+.50*span,512),(f'new_valid_{st}',lo+.52*span,lo+.68*span,256),(f'calib_new_{st}',lo+.70*span,lo+.84*span,256),(f'decision_new_{st}',lo+.86*span,hi,256)]
new_defs += [('final_new_normal',.86,.999,1024)]
attack_defs=[('decision_attack',0.,.15,1024),('candidate_0',.20,.30,256),('candidate_1',.35,.45,256),('candidate_2',.50,.60,256),('final_old_attack',.70,.999,1024)]
order=['train','valid','dev','calib_old','decision_old','decision_attack']+[r for st in range(3) for r in [f'candidate_{st}',f'new_valid_{st}',f'calib_new_{st}',f'decision_new_{st}']]+['final_old_normal','final_old_attack','final_new_normal','final_new_attack']
for old_type,new_type,dirname in [('Thing_Speak','MQTT_Publish','rt_thing_to_mqtt_v1'),('MQTT_Publish','Thing_Speak','rt_mqtt_to_thing_v1')]:
 out=HOME/dirname
 if out.exists():raise SystemExit('Existing split preserved: '+dirname)
 out.mkdir();buckets=collections.defaultdict(list);audit=collections.Counter()
 protocol=dict(author='20222038 이진호',created=datetime.datetime.now().astimezone().isoformat(),source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.file_digest(SOURCE.open('rb'),'sha256').hexdigest(),source_page='https://archive.ics.uci.edu/dataset/942/rt-iot2022',license='CC BY 4.0; DOI 10.24432/C5P338',feature_count=81,feature_names=columns,excluded_feature_columns=['Unnamed: 0','proto','service','Attack_type'],old_normal_pattern=old_type,new_normal_pattern=new_type,additional_final_only_normal_pattern='Wipro_bulb',normal_labels=sorted(normal_types),known_development_attacks=sorted(known_attacks),final_only_attacks=sorted(set(df.Attack_type)-normal_types-known_attacks),normal_old_blocks=old_defs,normal_new_stage_blocks=new_defs,known_attack_blocks=attack_defs,clients='5 simulated clients, not identified physical hosts; deterministic role/type-balanced assignment',row_order='Within-label CSV row blocks, not verified timestamps or new captures',label_conflicts='All contradictory numeric feature groups excluded from all fit/selection roles and retained exclusively in final evaluation, including duplicate opposite labels',ambiguous_rows=len(amb_indices),ambiguous_feature_groups=len(ambiguous),actual_label_counts=df.Attack_type.value_counts().to_dict(),official_metadata_discrepancy='Use actual CSV labels/counts; official prose lists mismatched MQTT/Thing counts and an Alexa class absent from this file',selection_boundary='Final NPZ must stay unopened until cross-dataset method and feature recipe are frozen')
 (out/'protocol_frozen.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
 for label in sorted(set(df.Attack_type)):
  indexes=np.flatnonzero(df.Attack_type.to_numpy()==label)
  if label==old_type:defs=old_defs
  elif label==new_type:defs=new_defs
  elif label=='Wipro_bulb':defs=[('final_new_normal',0.,1.,10000)]
  elif label in known_attacks:defs=attack_defs
  else:defs=[('final_new_attack',.05,.999,1024)]
  for role,lo,hi,cap in defs:
   pool=indexes[int(lo*len(indexes)):int(hi*len(indexes))]
   pool=[int(i) for i in pool if fingerprints[int(i)] not in ambiguous]
   pool.sort(key=lambda i:hashlib.sha256(f'{label}/{role}/{i}/9422026'.encode()).digest())
   buckets[(role,label)]=[(i,cap) for i in pool[:cap*4]]
 roles={f'{cid}/{role}':[] for cid in range(1,6) for role in order};vectors={k:[] for k in roles};seen=set()
 def append(cid,role,i,conflict=False):
  label=str(df.iloc[i].Attack_type);meta=dict(id=f'RT_IOT2022::{i}',client=cid,device=f'Virtual-client-{cid}',source='RT_IOT2022',row=i,family='benign' if y[i]==0 else 'known_development_attack' if label in known_attacks else 'heldout_attack_type',attack_type=label,normal_pattern=label if y[i]==0 else None,y=int(y[i]),fingerprint=fingerprints[i],label_conflict=bool(conflict),novel_normal=label=='Wipro_bulb')
  key=f'{cid}/{role}';roles[key].append(meta);vectors[key].append(x[i].copy())
 for role in order:
  for key in sorted(k for k in buckets if k[0]==role):
   count=0
   for i,cap in buckets[key]:
    if count>=cap:break
    h=fingerprints[i]
    if h in seen:audit['same_feature_duplicate_excluded']+=1;continue
    seen.add(h);append(count%5+1,role,i);count+=1
 # Contradictory groups stay out of fitting/selection, but difficult cases remain in the final metrics.
 for i in amb_indices:
  label=str(df.iloc[i].Attack_type)
  role=('final_old_normal' if label==old_type else 'final_new_normal') if y[i]==0 else ('final_old_attack' if label in known_attacks else 'final_new_attack')
  cid=int(hashlib.sha256(f'RT_IOT2022::{i}'.encode()).hexdigest()[:8],16)%5+1;append(cid,role,i,True)
 for cid in range(1,6):
  for role in ['train','valid','dev','calib_old','decision_old','decision_attack','final_old_normal','final_old_attack','final_new_normal','final_new_attack']+[r for st in range(3) for r in [f'new_valid_{st}',f'calib_new_{st}',f'decision_new_{st}']]:
   minimum=128 if role=='train' else 12
   assert len(roles[f'{cid}/{role}'])>=minimum,('Insufficient predeclared role',dirname,cid,role,len(roles[f'{cid}/{role}']))
  for st in range(3):assert sum(r['y']==0 for r in roles[f'{cid}/candidate_{st}'])>=24
 arrays={k:np.asarray(vectors[k],dtype=np.float64).reshape(-1,81) for k in roles}
 with gzip.open(out/'roles.json.gz','wt',encoding='utf-8') as f:json.dump(roles,f)
 np.savez_compressed(out/'development.npz',**{k.replace('/','__'):v for k,v in arrays.items() if not k.split('/')[1].startswith('final_')})
 np.savez_compressed(out/'final.npz',**{k.replace('/','__'):v for k,v in arrays.items() if k.split('/')[1].startswith('final_')})
 (out/'feature_names.json').write_text(json.dumps(columns,indent=2),encoding='utf-8')
 counts={k:dict(rows=len(rs),normal=sum(r['y']==0 for r in rs),attack=sum(r['y']!=0 for r in rs)) for k,rs in roles.items()}
 (out/'role_counts.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
 (out/'preparation_audit.json').write_text(json.dumps(dict(state='passed',rows_kept=sum(len(rs) for rs in roles.values()),numeric_features=81,virtual_clients=5,audit=dict(audit),all_contradictory_rows_retained_in_final=len(amb_indices),fitting_fingerprints_exclude_all_contradictions=True,source_sha256=protocol['source_sha256'],development_npz_sha256=hashlib.file_digest((out/'development.npz').open('rb'),'sha256').hexdigest(),final_npz_sha256=hashlib.file_digest((out/'final.npz').open('rb'),'sha256').hexdigest(),limits='Independent dataset from N-BaIoT, but two reciprocal scenarios share source rows and are not independent datasets. Application normal patterns are not proof of distinct hosts or time drift.'),indent=2),encoding='utf-8')
 print(dirname,sum(len(rs) for rs in roles.values()),'rows, ambiguous final rows',len(amb_indices),flush=True)
