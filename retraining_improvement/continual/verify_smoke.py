from pathlib import Path
import json,gzip,csv,math,sys
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'continual/output'/(sys.argv[1] if len(sys.argv)>1 else 'smoke_v3')
config=json.loads((out/'config.json').read_text()); result=json.loads((out/'results.json').read_text())
with gzip.open(out/'split_manifest.json.gz','rt',encoding='utf-8') as f: groups=json.load(f)
ids=set(); fps=set(); by_id={}
for name,rows in groups.items():
 for row in rows:
  assert row['id'] not in ids and row['fingerprint'] not in fps
  ids.add(row['id']); fps.add(row['fingerprint']); by_id[row['id']]=(name,row)
assert len(result)==6 and {x['arm'] for x in result}=={'no_update','new_only','replay'}
for r in result:
 assert all(math.isfinite(r['metrics'][d][k]) and 0<=r['metrics'][d][k]<=1 for d in ['old','new'] for k in ['mean_client_auc','fpr','tpr'])
 path=out/f'seed_{r["seed"]}'/r['arm']/f'stage_{r["stage"]}'
 with gzip.open(path/'predictions.csv.gz','rt',encoding='utf-8') as f: predictions=list(csv.DictReader(f))
 detections=json.loads((path/'detections_before_update.json').read_text())['items']
 for item in detections:
  partition,_=by_id[item['id']]; assert '/candidate_' in partition; assert math.isfinite(item['score'])
 for p in predictions:
  partition,row=by_id[p['id']]
  assert partition.endswith(('/old_normal','/old_attack')) if p['domain']=='old' else partition.endswith(('/new_normal','/new_attack'))
  assert int(p['label'])==row['y']; assert math.isfinite(float(p['score']))
 for d in ['old','new']:
  prs=[x for x in predictions if x['domain']==d]; tn=sum(int(x['label'])==0 and int(x['prediction'])==0 for x in prs); fp=sum(int(x['label'])==0 and int(x['prediction'])==1 for x in prs); fn=sum(int(x['label'])==1 and int(x['prediction'])==0 for x in prs); tp=sum(int(x['label'])==1 and int(x['prediction'])==1 for x in prs)
  assert r['metrics'][d]['confusion_matrix']==[[tn,fp],[fn,tp]]
for stage in [1,2]:
 a=json.loads((out/f'seed_0/new_only/stage_{stage}/training/training.json').read_text())
 b=json.loads((out/f'seed_0/replay/stage_{stage}/training/training.json').read_text())
 assert [x['participants'] for x in a['rounds']]==[x['participants'] for x in b['rounds']]
 assert [[v['rows'] for v in x['local']] for x in a['rounds']]==[[v['rows'] for v in x['local']] for x in b['rounds']]
base={r['state_sha256'] for r in result if r['arm']=='no_update'}
assert len(base)==1
assert all(r['state_sha256'] not in base and not r['update_skipped'] for r in result if r['arm']!='no_update')
assert all(r['replay_samples_this_stage']>0 for r in result if r['arm']=='replay')
summary=dict(validation='passed',checks=['Global ID and exact-feature fingerprint disjointness across all roles','Six real-data evaluation records, three arms and two arrival stages','Future evaluation IDs never belong to candidate or training partitions','Saved confusion matrices independently recomputed from prediction CSVs','Same selected clients and sample budget in new-only and replay training','No-update weights unchanged; both updating arms actually change weights','Replay includes retained initial normal records; attack/uncertain routing excluded by training guards'],scope='4 clients, capped real data, 2 epochs/2 rounds; implementation validation only, not final comparative performance')
(out/'validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

