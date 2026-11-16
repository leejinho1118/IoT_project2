"""Read-only validation and summary of all original 6 x 5 experiments."""
from pathlib import Path
import json,statistics,re,math,datetime
ROOT=Path(__file__).resolve().parents[1]
s=json.loads((ROOT/'original_output/status.json').read_text(encoding='utf-8-sig'))
if s.get('state')!='complete' or s.get('exit_code')!=0: raise SystemExit('Original run is not complete; no final summary produced.')
expected={(m,model,run) for m in ['avg','fedprox','mse_avg'] for model in ['hybrid','autoencoder'] for run in range(5)}
found={}
for p in (ROOT/'src/Checkpoint/Results').rglob('*_results.json'):
 m=re.search(r'_(avg|fedprox|mse_avg)_results\.json$',p.name)
 if not m: raise ValueError('Unexpected result filename '+str(p))
 model='autoencoder' if '_autoencoder_' in p.name else 'hybrid'; run=int(re.search(r'Run_(\d+)',str(p)).group(1)); key=(m[1],model,run)
 if key in found: raise ValueError('Duplicate experiment '+str(key))
 rows=[]
 for line in p.read_text(encoding='utf-8').splitlines():
  obj=json.loads(line)
  if len(obj)!=1: raise ValueError('Unexpected result structure')
  round_name,value=next(iter(obj.items())); n=int(round_name.split('_')[-1])
  if n!=len(rows)+1 or n>20: raise ValueError('Round sequence')
  scores={k:v for k,v in value.items() if k not in ('global_loss','join_clients')}
  if len(scores)!=50 or not all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in scores.values()): raise ValueError('Invalid client AUC')
  chosen=value['join_clients']
  if len(chosen)!=25 or len(set(chosen))!=25 or any(not isinstance(i,int) or not 0<=i<50 for i in chosen): raise ValueError('Invalid participation')
  if not math.isfinite(value['global_loss']): raise ValueError('Nonfinite loss')
  rows.append(dict(round=n,mean_auc=statistics.mean(scores.values()),std_across_clients=statistics.stdev(scores.values()),min_client_auc=min(scores.values()),global_loss=value['global_loss']))
 if not rows: raise ValueError('Empty result')
 found[key]=dict(method=key[0],model=model,run=run,rounds=rows,path=str(p.relative_to(ROOT)))
if set(found)!=expected: raise ValueError('Missing or unexpected experiments '+str(expected-set(found)))
summary=[]
for m in ['avg','fedprox','mse_avg']:
 for model in ['hybrid','autoencoder']:
  runs=[found[m,model,i] for i in range(5)]; values=[x['rounds'][-1]['mean_auc'] for x in runs]
  summary.append(dict(method=m,model=model,final_mean_client_auc_mean=statistics.mean(values),std_across_runs=statistics.stdev(values),runs=runs))
result=dict(validated=datetime.datetime.now().astimezone().isoformat(),commit='d1268850696ffa7e9ca5469c0fc9276e966eafd7',experiments=30,policy='Last actual round, mean of 50 client AUC values; no best-test-round selection.',summary=summary)
p=ROOT/'continual/original_baseline_summary.json'; p.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8'); print(p)
