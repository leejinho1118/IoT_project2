"""개발/최종 예측, 입력 ID, 조기 종료 예산, 실제 모델 버전과 게이트 독립 감사."""
from pathlib import Path
import json,gzip,csv,math,hashlib,random,statistics
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from policies import memory_bank,mix,adaptive_ratio,gate_accept,development_score
from data import B


def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def read_manifest(out):
    with gzip.open(out/'partition_manifest.json.gz','rt',encoding='utf-8') as f:groups=json.load(f)
    seen={};fps=set()
    for role,rows in groups.items():
        for r in rows:
            if r['id'] in seen or r['fingerprint'] in fps:raise AssertionError('Partition leakage')
            seen[r['id']]=(role,r);fps.add(r['fingerprint'])
    return groups,seen


def verify_predictions(path,expected,by,clients,final=False):
    with gzip.open(path,'rt',encoding='utf-8') as f:pred=list(csv.DictReader(f))
    for domain in ['old','new']:
        rows=[r for r in pred if r['domain']==domain];assert len({r['id'] for r in rows})==len(rows)
        cm=[[0,0],[0,0]];aucs=[]
        normal_role=('old_normal' if domain=='old' else 'new_normal') if final else ('decision_old' if domain=='old' else 'decision_new')
        attack_role=('old_attack' if domain=='old' else 'new_attack') if final else 'decision_attack'
        for r in rows:
            role,meta=by[r['id']];assert role.endswith('/'+(normal_role if r['label']=='0' else attack_role));assert int(r['label'])==meta['y']
            assert math.isfinite(float(r['score'])) and math.isfinite(float(r['threshold']))
            assert int(r['prediction'])==int(float(r['score'])>float(r['threshold']));cm[int(r['label'])][int(r['prediction'])]+=1
        for cid in range(1,clients+1):
            q=[r for r in rows if int(r['client'])==cid];aucs.append(roc_auc_score([int(r['label']) for r in q],[float(r['score']) for r in q]))
        m=expected[domain];assert cm==m['confusion_matrix'];assert abs(statistics.mean(aucs)-m['mean_client_auc'])<1e-12
        assert abs(cm[0][1]/sum(cm[0])-m['fpr'])<1e-12 and abs(cm[1][1]/sum(cm[1])-m['tpr'])<1e-12


def verify_round(out,letter,config):
    groups,by=read_manifest(out);records=load(out/letter/'development_results.json');policy=load(out/letter/'policy.json')
    assert len(records)==len(config['seeds'])*config['stages']
    assert {(r['seed'],r['stage']) for r in records}=={(s,k) for s in config['seeds'] for k in range(1,config['stages']+1)}
    for seed in config['seeds']:
        base=load(out/'initial'/f'seed_{seed}'/'metrics.json');previous=base;ratio=policy['ratio']
        source=torch.load(out/'initial'/f'seed_{seed}'/'detector_bundle.pt',map_location='cpu',weights_only=False)
        previous_ref={cid:c['reference_ids'] for cid,c in source['clients'].items()}
        bank=load(out/letter/f'seed_{seed}'/'memory_ids.json')
        initial_sha=hashlib.sha256(b''.join(v.numpy().tobytes() for v in source['model_state'].values())).hexdigest()
        assert load(out/letter/f'seed_{seed}'/'start.json')['initial_weights_sha256']==initial_sha
        for cid,ids in bank.items():assert len(ids)==config['capacity'] and len(set(ids))==len(ids) and set(ids)<={r['id'] for r in groups[f'{cid}/train']}
        for stage in range(1,config['stages']+1):
            folder=out/letter/f'seed_{seed}'/f'stage_{stage}';r=load(folder/'record.json')
            if policy['adaptive']:ratio=adaptive_ratio(ratio,base['normal_mse'],None if stage==1 else previous['normal_mse'])
            assert r['ratio']==ratio
            for which,key in [('candidate','candidate_metrics'),('published','metrics')]:verify_predictions(folder/which/'predictions.csv.gz',r[key],by,config['clients'])
            if policy['gate']:
                accept,reason=gate_accept(r['candidate_metrics'],previous,base);assert accept==r['accepted'] and reason==r['gate_reason']
            else:assert r['accepted']
            ids=load(folder/'training_input_ids.json');replayed=0
            for cid,items in ids.items():
                candidates={x['id'] for x in groups[f'{cid}/candidate_{stage-1}'] if x['y']==0};memory=set(bank[cid])
                assert len(items)==len(candidates) and len(set(items))==len(items)
                assert set(items)<=candidates|memory and all(by[i][1]['y']==0 for i in items)
                count=min(int(len(candidates)*ratio),len(memory),len(candidates)-1)
                assert len(set(items)&memory)==count;replayed+=count
            assert replayed==r['replay_samples']
            history=load(folder/'training/training.json');assert len(history['rounds'])<=config['update_rounds'];step_count=0;rng=random.Random(seed+100000+stage-1)
            for rr in history['rounds']:
                expected=rng.sample(range(config['clients']),int(.5*config['clients']))
                assert rr['participants']==[x+1 for x in expected] and math.isfinite(rr['global_loss'])
                for x in rr['local']:
                    assert x['rows']==len(ids[str(x['client'])]);assert 1<=x['epochs']<=config['local_epochs'];assert x['optimizer_steps']==x['epochs']*math.ceil(x['rows']/12);step_count+=x['optimizer_steps']
            assert step_count==r['optimizer_steps']
            bundle=torch.load(folder/'published/detector_bundle.pt',map_location='cpu',weights_only=False)
            assert hashlib.sha256(b''.join(v.numpy().tobytes() for v in bundle['model_state'].values())).hexdigest()==r['state_sha256']
            if r['accepted']:previous_ref={int(cid):items for cid,items in ids.items()}
            for cid,c in bundle['clients'].items():
                assert c['reference_ids']==previous_ref[cid]
                assert set(c['threshold_source_ids'])=={x['id'] for x in groups[f'{cid}/calib_old']}
                assert np.array_equal(c['input_scaler'].mean_,source['clients'][cid]['input_scaler'].mean_)
                assert np.array_equal(c['input_scaler'].scale_,source['clients'][cid]['input_scaler'].scale_)
                assert all(by[i][1]['y']==0 for i in c['reference_ids'])
            if not r['accepted']:assert r['metrics']==previous
            previous=r['metrics']
    report=dict(validation='passed',records=len(records),checks=['development predictions/confusion/AUC independently recomputed','development/final global ID and feature disjointness','normal-only candidate and replay IDs, fixed sample budgets','exact original participant sequence and max epoch/round budgets','actual optimizer step counts','adaptive ratio from prior decision normals only','gate and rejection rollback','published bundle hash, reference IDs, calibration IDs and fixed input scaler'])
    B.atomic(out/letter/'validation.json',report);return report


def verify_final(out,config):
    groups,by=read_manifest(out);selection=out/'selection_frozen.json';sha=hashlib.sha256(selection.read_bytes()).hexdigest();records=load(out/'final_results.json')
    assert len(records)==len(config['seeds'])*(1+5*config['stages'])
    assert len({(r['round'],r['seed'],r['stage']) for r in records})==len(records)
    for r in records:
        assert r['selection_sha256']==sha
        p=out/'final'/r['round']/f'seed_{r["seed"]}'/f'stage_{r["stage"]}'/'predictions.csv.gz'
        verify_predictions(p,r['metrics'],by,config['clients'],final=True)
    summary=[]
    for letter in ['initial']+list('ABCDE'):
        rows=[r for r in records if r['round']==letter and (r['stage']==0 if letter=='initial' else r['stage']==config['stages'])]
        item={'round':letter}
        for domain in ['old','new']:
            item[domain]={k:dict(mean=statistics.mean(r['metrics'][domain][k] for r in rows),std=statistics.stdev(r['metrics'][domain][k] for r in rows) if len(rows)>1 else None) for k in ['mean_client_auc','fpr','tpr']}
        summary.append(item)
    B.atomic(out/'final_summary.json',dict(validation='passed',records=len(records),selected_by_development=load(selection)['selected_round'],summary=summary,scope='Final holdout locked during this campaign but previously seen in V2; exploratory, no external-replication claim'))
    return dict(validation='passed',records=len(records),prediction_client_auc_checks=len(records)*config['clients']*2)
