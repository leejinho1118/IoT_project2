import csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t
R=Path(__file__).resolve().parent
OLD=R.parent.parent/'src/Checkpoint/Results/EXP/H12/final_mse_avg'
def data(cell,method,run,rounds=20):
 p=R/f'runs/{method}/{cell}/Run_{run}.jsonl'; x=[list(json.loads(z).values())[0] for z in p.open()]
 if len(x)!=rounds: raise ValueError(f'incomplete {p}: {len(x)}/{rounds}')
 return x
def meanlast(x): return float(np.mean(x[-1]['global_auc']))
def calc(d):
 a=np.asarray(d); m=float(a.mean()); h=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/math.sqrt(len(a)))
 return m,m-h,m+h
out=[]; csvrows=[]
for cell,n in [('noniid',10),('iid',5)]:
 methods=['mse','fedavg','fedavgm','eta10']; vals={m:[] for m in methods}; joins=True
 for r in range(n):
  ds={m:data(cell,m,r) for m in methods}
  for m,x in ds.items(): vals[m].append(meanlast(x))
  joins &= all(all(ds['mse'][i]['join_clients']==ds[m][i]['join_clients'] for i in range(20)) for m in methods)
  for m,x in ds.items():
   for i,z in enumerate(x,1): csvrows.append({'cell':cell,'method':m,'run':r,'round':i,'mean_client_auc':meanlast([z]),'global_auc_clients':json.dumps(z['global_auc']),'join_clients':json.dumps(z['join_clients'])})
 for m in methods:
  avg=float(np.mean(vals[m])); out.append({'cell':cell,'method':m,'runs':n,'mean_final_auc':avg,'mean_final_auc_pct':avg*100,'join_sequences_match':joins})
 for c in ['mse','fedavg','eta10']:
  dif=(np.array(vals['fedavgm'])-np.array(vals[c]))*100; m,lo,hi=calc(dif)
  out.append({'cell':cell,'comparison':f'fedavgm_minus_{c}','runs':n,'paired_diff_pp':m,'ci95_low_pp':lo,'ci95_high_pp':hi,'df':n-1,'join_sequences_match':joins})
with (R/'RESULTS_confirm.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=sorted({k for x in csvrows for k in x}));w.writeheader();w.writerows(csvrows)
with (R/'SUMMARY_confirm.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=sorted({k for x in out for k in x}));w.writeheader();w.writerows(out)
# H12 reproduction reference and C2 40-round reference
h12=[]
for r in range(10):
 x=[list(json.loads(z).values())[0] for z in (OLD/f'Run_{r}.jsonl').open()];h12.append(meanlast(x))
base=np.mean([meanlast(data('noniid','mse',r)) for r in range(10)])
hdelta=(base-float(np.mean(h12)))*100
c2=[meanlast(data('noniid','mse',r,40)) for r in range(10)]
repro={'h12_mean_final_auc':float(np.mean(h12)),'new_b1_mean_final_auc':base,'difference_pp':hdelta,'within_0_3pp':abs(hdelta)<=.3}
(R/'REPRO_confirm.json').write_text(json.dumps(repro,indent=2)+'\n')
(R/'C2_confirm.json').write_text(json.dumps({'rounds':40,'runs':10,'mean_final_auc':float(np.mean(c2)),'mean_final_auc_pct':float(np.mean(c2)*100),'note':'40라운드 결과; 20라운드 주 대조와 라운드 수가 다름'},indent=2)+'\n')
print(json.dumps({'summary':out,'repro':repro,'c2':float(np.mean(c2)*100)},indent=2))
