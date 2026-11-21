"""Package reviewable code and evidence; exclude raw CSV/archives and model tensors."""
from pathlib import Path
import json,zipfile,hashlib,datetime
H=Path(__file__).resolve().parent;ROOT=H.parents[1];OUT=H/'output';OUT.mkdir(exist_ok=True);target=OUT/'FedMSE_Overnight_Research_Package.zip';assert not target.exists()
files=[]
for p in H.glob('*.py'):files.append((p,'continual/overnight_2026_11_21/'+p.name))
for p in H.glob('*.txt'):files.append((p,'continual/overnight_2026_11_21/'+p.name))
for p in H.glob('*.json'):
 if p.stat().st_size<8_000_000:files.append((p,'continual/overnight_2026_11_21/'+p.name))
for sub in ['analysis_v1','output/pdf']:
 for p in (H/sub).rglob('*'):
  if p.is_file():files.append((p,'continual/overnight_2026_11_21/'+str(p.relative_to(H)).replace('\\','/')))
for sub in ['raw_blocks_v2','rt_thing_to_mqtt_v1','rt_mqtt_to_thing_v1']:
 for p in (H/sub).glob('*.json'):files.append((p,'continual/overnight_2026_11_21/'+str(p.relative_to(H)).replace('\\','/')))
 p=H/sub/'roles.json.gz';files.append((p,'continual/overnight_2026_11_21/'+str(p.relative_to(H)).replace('\\','/')))
for base in ['runs','normal_runs','control_runs','objective_runs']:
 if not (H/base).exists():continue
 for folder in (H/base).iterdir():
  if not (folder/'final_evaluation_v1/independent_validation.json').exists():continue
  for name in ['protocol_frozen.json','status.json','environment.json','initial_sources.json','runtime_hashes.json','upstream_hashes.json','independent_development_validation.json','independent_approval_validation.json','independent_bundle_validation.json','independent_approval_bundle_validation.json','independent_guard_validation.json','independent_normal_guard_validation.json','label_and_payload_audit.json','matched_main_control_validation.json','independent_participation_validation.json','independent_routing_and_reference_validation.json','final_evaluation_v1/results.json','final_evaluation_v1/complete.json','final_evaluation_v1/final_access.json','final_evaluation_v1/independent_validation.json']:
   p=folder/name
   if p.exists():files.append((p,'continual/overnight_2026_11_21/'+str(p.relative_to(H)).replace('\\','/')))
for sub in ['src']:
 for p in (ROOT/sub).rglob('*.py'):files.append((p,str(p.relative_to(ROOT)).replace('\\','/')))
for rel in ['continual/drift_v2/backend.py','continual/retraining_v3/losses.py','continual/retraining_v3/checkpoint_guard.py','continual/research_ABCDE/policies.py']:
 p=ROOT/rel;files.append((p,rel))
seen=set();manifest=[]
with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
 for p,name in files:
  assert name not in seen;seen.add(name);assert p.suffix not in ['.pt','.npz'];z.write(p,name);manifest.append(dict(name=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 z.writestr('REPRODUCE.txt','Review package with original directory layout. Raw feature arrays, downloaded archives, model tensors and row-level prediction CSVs are excluded. Reproduction requires the official UCI 442/942 source data and the original FedMSE prepared data at the recorded source commit. See dataset protocol_frozen.json, source hashes, roles.json.gz, environment versions and Python preparation/training scripts. Do not present this archive as containing the entire training data or final models. Reports use a separate bundled Python/ReportLab runtime. Full evidence remains under C:/Users/dlwls/LocalProjects/fedmse. Author: 20222038 Lee Jinho.\n')
 z.writestr('MANIFEST.json',json.dumps(dict(created=datetime.datetime.now().astimezone().isoformat(),author='20222038 이진호',scope='Code/protocol/aggregate evidence/report. Raw data, weights and row-level prediction files remain in original LocalProjects directory and are intentionally excluded from this portable review package.',files=manifest),ensure_ascii=False,indent=2))
with zipfile.ZipFile(target) as z:assert z.testzip() is None
(H/'output/package_validation.json').write_text(json.dumps(dict(state='passed',files=len(manifest)+2,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),raw_data_or_weights_included=False),indent=2),encoding='utf-8');print(str(target))

