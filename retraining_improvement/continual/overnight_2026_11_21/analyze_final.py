"""Summarize locked results without selecting or retuning models."""
from pathlib import Path
import json,csv,itertools,collections
import numpy as np
H=Path(__file__).resolve().parent;OUT=H/'analysis_v1'
def mean(x):return float(np.mean(x))
def sd(x):return float(np.std(x,ddof=1)) if len(x)>1 else None
def exact_signflip(values):
 x=np.array(values);observed=abs(x.mean());return sum(abs(np.mean(x*np.array(s)))>=observed-1e-15 for s in itertools.product([-1,1],repeat=len(x)))/(2**len(x))
def main():
 assert not OUT.exists(),'Existing analysis preserved';OUT.mkdir();F=json.loads((H/'family_selection_frozen.json').read_text());rows=[];runs=[]
 for folder in sorted(list((H/'runs').iterdir())+list((H/'normal_runs').iterdir() if (H/'normal_runs').exists() else [])+list((H/'control_runs').iterdir() if (H/'control_runs').exists() else [])+list((H/'objective_runs').iterdir() if (H/'objective_runs').exists() else [])):
  final=folder/'final_evaluation_v1'
  if not (final/'independent_validation.json').exists():continue
  pr=json.loads((folder/'protocol_frozen.json').read_text());cfg=pr['config'];kind=cfg.get('selection_ablation') or cfg.get('approval_stress') or ('budget100' if cfg['local_epochs']==100 else 'main30');data=Path(cfg['data']).name
  source=json.loads((folder/'initial_sources.json').read_text());runs.append(dict(run=folder.name,data=data,mode=cfg['mode'],loss=cfg['shrink_kind'],kind=kind,initial_sources=source,status=json.loads((folder/'status.json').read_text())))
  for r in json.loads((final/'results.json').read_text()):
   row=dict(run=folder.name,data=data,mode=cfg['mode'],loss=cfg['shrink_kind'],kind=kind,method=r['method'],seed=r['seed'],stage=r['stage'],seconds=r['cumulative_training_seconds'],steps=r['cumulative_optimizer_steps'],accepted=r['checkpoint'].get('accepted'),trigger_skipped=r['checkpoint'].get('trigger_skipped',False),trained=not (r['method'] in ['no_update','calibration_only','reference_only','reference_only_new']) and not r['checkpoint'].get('trigger_skipped',False))
   for d in ['old','new']:
    for key in ['mean_client_auc','mean_client_pr_auc','fpr','tpr','auc_clients','normal_clients','attack_clients']:row[f'{d}_{key}']=r['metrics'][d][key]
    row[f'{d}_worst_client_auc']=min(c['auc'] for c in r['metrics'][d]['clients'] if c['auc'] is not None);row[f'{d}_worst_client_fpr']=max(c['fpr'] for c in r['metrics'][d]['clients'] if c['fpr'] is not None)
   rows.append(row)
 assert rows,'No independently validated final results'
 key=lambda r:(r['data'],r['mode'],r['loss'],r['kind'],r['method'],r['seed'],r['stage']);assert len({key(r) for r in rows})==len(rows)
 numeric=['old_mean_client_auc','new_mean_client_auc','old_mean_client_pr_auc','new_mean_client_pr_auc','old_fpr','new_fpr','old_tpr','new_tpr','seconds','steps','old_worst_client_auc','new_worst_client_auc','new_worst_client_fpr']
 groups=collections.defaultdict(list)
 for r in rows:groups[tuple(r[k] for k in ['data','mode','loss','kind','method','stage'])].append(r)
 summary=[]
 for k,rr in groups.items():
  obj=dict(zip(['data','mode','loss','kind','method','stage'],k));obj['n_seeds']=len(rr);obj['seeds']=sorted(r['seed'] for r in rr)
  for metric in numeric:values=[r[metric] for r in rr];obj[metric+'_mean']=mean(values);obj[metric+'_sd']=sd(values)
  obj['update_accept_count']=sum(r['accepted'] is True for r in rr if r['method'] in ['safe_new','safe_replay','trigger_safe']);obj['trigger_skip_count']=sum(r['trigger_skipped'] for r in rr);summary.append(obj)
 paired=[]
 for group in sorted({tuple(r[k] for k in ['data','mode','loss','kind']) for r in rows}):
  subset=[r for r in rows if tuple(r[k] for k in ['data','mode','loss','kind'])==group and r['stage']==3];primary={r['seed']:r for r in subset if r['method']=='safe_new'}
  for baseline in ['no_update','calibration_only','reference_only','new_only','replay','safe_replay','trigger_safe']:
   base={r['seed']:r for r in subset if r['method']==baseline};seeds=sorted(primary.keys()&base.keys())
   if not seeds:continue
   for metric in ['old_mean_client_auc','new_mean_client_auc','old_fpr','new_fpr','old_tpr','new_tpr']:
    delta=[primary[s][metric]-base[s][metric] for s in seeds];paired.append(dict(zip(['data','mode','loss','kind'],group))|dict(primary='safe_new',baseline=baseline,metric=metric,seeds=seeds,n=len(seeds),mean_delta=mean(delta),sd_delta=sd(delta),min_delta=min(delta),max_delta=max(delta),individual_deltas=delta,exact_two_sided_signflip_p=exact_signflip(delta),interpretation='Exploratory paired learning-seed variability on fixed source blocks, not independent capture evidence'))
 # Cross-kind contrasts isolate neural updates from exactly matched reference-only controls.
 for data,mode,loss in sorted({(r['data'],r['mode'],r['loss']) for r in rows}):
  for pk,pm,bk,bm in [('main30','new_only','reference_only_new','reference_only_new'),('main30','safe_new','reference_only_new','reference_only_new'),('normal_only','normal_gate_new','main30','new_only'),('normal_only','normal_gate_new','main30','safe_new'),('normal_only','normal_gate_replay','main30','replay'),('gpu_objectives','distill_replay','main30','replay'),('gpu_objectives','align_distill_replay','main30','replay'),('gpu_objectives','safe_align_distill','main30','safe_replay'),('gpu_objectives','safe_align_distill','main30','safe_new'),('gpu_objectives','align_distill_replay','gpu_objectives','distill_replay'),('gpu_objectives','safe_align_distill','gpu_objectives','align_distill_replay')]:
   aa={r['seed']:r for r in rows if (r['data'],r['mode'],r['loss'],r['kind'],r['method'],r['stage'])==(data,mode,loss,pk,pm,3)};bb={r['seed']:r for r in rows if (r['data'],r['mode'],r['loss'],r['kind'],r['method'],r['stage'])==(data,mode,loss,bk,bm,3)};seeds=sorted(aa.keys()&bb.keys())
   if not seeds:continue
   for metric in ['old_mean_client_auc','new_mean_client_auc','old_fpr','new_fpr','old_tpr','new_tpr']:
    delta=[aa[z][metric]-bb[z][metric] for z in seeds];paired.append(dict(data=data,mode=mode,loss=loss,kind=pk+'_vs_'+bk,primary=pm,baseline=bm,metric=metric,seeds=seeds,n=len(seeds),mean_delta=mean(delta),sd_delta=sd(delta),min_delta=min(delta),max_delta=max(delta),individual_deltas=delta,exact_two_sided_signflip_p=exact_signflip(delta),interpretation='Matched learning seeds; reference-only controls isolate weight updates for ordinary new_only; guarded rollback retains a joint-system interpretation'))
 for name,rr in [('records.csv',rows),('summary.csv',summary),('paired_differences.csv',paired)]:
  with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 result=dict(author='20222038 이진호',selection=F,runs=runs,records=rows,summary=summary,paired=paired,limits=['RT reciprocal scenarios share source rows and are not independent datasets','Repeated seeds measure optimizer variability on fixed evaluation blocks','With five nonzero pairs, the smallest exact two-sided sign-flip p-value is 0.0625','N-BaIoT raw captures are the same public acquisition, with prior prepared exact/coarse matches filtered','File-row order is a proxy, not verified capture timestamps','No test-set tuning; all negative outcomes retained'])
 (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(runs=len(runs),records=len(rows),summary_rows=len(summary),paired_rows=len(paired)),indent=2))
if __name__=='__main__':main()
