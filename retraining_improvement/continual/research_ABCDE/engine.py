"""5회 순차 연구의 학습·개발 평가·최종 평가. 원본 src/V2 파일은 수정하지 않는다."""
from pathlib import Path
import sys,json,csv,gzip,hashlib,copy,statistics,time,random,shutil
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'continual/drift_v2'))
import backend as B
from policies import memory_bank,mix,development_score,adaptive_ratio,gate_accept,choose_best


def weights_hash(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def normal_mse(model,clients):
    result={};device=next(model.parameters()).device;model.eval()
    for domain,key in [('old','decision_old'),('new','decision_new')]:
        error=0.;count=0
        for c in clients:
            x=c['scaler'].transform(B.payload(c[key]))
            with torch.no_grad():
                for start in range(0,len(x),256):
                    batch=torch.tensor(x[start:start+256],dtype=torch.float32,device=device);_,out,_=model(batch)
                    error+=float(torch.sum((batch-out)**2).item());count+=batch.numel()
        result[domain]=error/count
    return result


def evaluate_development(model,clients,reference,folder,config):
    folder.mkdir(parents=True,exist_ok=True);result={};detectors={};prediction=[]
    for c in clients:
        if any(r['y']!=0 for r in reference[c['id']]):raise ValueError('Attack in reference')
        cen=B.CentroidBasedOneClassClassifier();cen.fit(B.encode(model,c,reference[c['id']]))
        threshold=float(np.quantile(cen.get_density(B.encode(model,c,c['calib_old'])),.95))
        detectors[c['id']]=(cen,threshold)
    for domain,key in [('old','decision_old'),('new','decision_new')]:
        cm=np.zeros((2,2),dtype=np.int64);aucs=[]
        for c in clients:
            items=c[key]+c['decision_attack'];y=np.array([r['y'] for r in items]);cen,threshold=detectors[c['id']]
            score=cen.get_density(B.encode(model,c,items))
            if not np.isfinite(score).all():raise ValueError('Nonfinite development score')
            pred=(score>threshold).astype(int);cm+=confusion_matrix(y,pred,labels=[0,1]);aucs.append(float(roc_auc_score(y,score)))
            prediction.extend(dict(id=r['id'],domain=domain,client=c['id'],label=int(yy),score=float(ss),prediction=int(pp),threshold=threshold) for r,yy,ss,pp in zip(items,y,score,pred))
        tn,fp,fn,tp=cm.ravel();result[domain]=dict(mean_client_auc=float(np.mean(aucs)),fpr=float(fp/(tn+fp)),tpr=float(tp/(tp+fn)),confusion_matrix=cm.tolist())
    result['normal_mse']=normal_mse(model,clients)
    with gzip.open(folder/'predictions.csv.gz','wt',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(prediction[0]));writer.writeheader();writer.writerows(prediction)
    bundle=dict(stage=int(folder.parent.name.split('_')[-1]) if folder.parent.name.startswith('stage_') else 0,architecture=dict(input_dim=115,output_dim=115,hidden_neus=50,latent_dim=11,shrink_lambda=10),model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},clients={c['id']:dict(input_scaler=c['scaler'],cen=detectors[c['id']][0],threshold=detectors[c['id']][1],reference_ids=[r['id'] for r in reference[c['id']]],threshold_source_ids=[r['id'] for r in c['calib_old']]) for c in clients},config=config,state_sha256=weights_hash(model))
    torch.save(bundle,folder/'detector_bundle.pt');B.atomic(folder/'metrics.json',result)
    return result


def decide_policy(letter,analyses,policies):
    if letter=='A':return dict(round='A',parent=None,memory='uniform',ratio=.25,adaptive=False,gate=False,direction='고정 25% 리플레이 기준 실험')
    if letter=='B':return dict(round='B',parent='A',memory='uniform',ratio=.50,adaptive=False,gate=False,direction='기존 성능 저하 대응: 리플레이 50%로 강화')
    best=choose_best(analyses);policy=copy.deepcopy(policies[best]);policy.update(round=letter,parent=best)
    if letter=='C':policy.update(memory='diverse',adaptive=False,gate=False,direction='고정 용량의 정상 패턴 다양성 확보: farthest-first 보관')
    if letter=='D':policy.update(adaptive=True,gate=False,direction='개발 정상 복원 오차에 따른 단계별 리플레이 비율 조정')
    if letter=='E':policy.update(gate=True,direction='개발 성능 검증 뒤 갱신 채택 또는 이전 버전으로 되돌리기')
    return policy


def train_round(policy,clients,source,out,config,status):
    B.configure(config,config['device']);records=[];begin=time.perf_counter()
    for seed in config['seeds']:
        B.seeds(seed)
        initial=torch.load(source/f'seed_{seed}/initial_evaluation/detector_bundle.pt',map_location='cpu',weights_only=False)
        device=torch.device(config['device']);model=B.Shrink_Autoencoder(**initial['architecture']).to(device);model.load_state_dict(initial['model_state'])
        for c in clients:c['scaler']=initial['clients'][c['id']]['input_scaler']
        reference={c['id']:c['train'] for c in clients}
        memory_begin=time.perf_counter()
        memory={c['id']:memory_bank(c['train'],c['scaler'],config['capacity'],policy['memory']) for c in clients}
        seed_dir=out/policy['round']/f'seed_{seed}';seed_dir.mkdir(parents=True,exist_ok=True)
        B.atomic(seed_dir/'start.json',dict(initial_weights_sha256=weights_hash(model),source_bundle_sha256=hashlib.sha256((source/f'seed_{seed}/initial_evaluation/detector_bundle.pt').read_bytes()).hexdigest(),memory_selection_seconds=time.perf_counter()-memory_begin))
        B.atomic(seed_dir/'memory_ids.json',{str(k):[r['id'] for r in v] for k,v in memory.items()})
        initial_folder=out/'initial'/f'seed_{seed}'
        if not (initial_folder/'metrics.json').exists():initial_metrics=evaluate_development(model,clients,reference,initial_folder,config)
        else:initial_metrics=json.loads((initial_folder/'metrics.json').read_text(encoding='utf-8'))
        previous=initial_metrics;ratio=policy['ratio'];cumulative=0.;steps=0
        for stage in range(config['stages']):
            folder=seed_dir/f'stage_{stage+1}';folder.mkdir(parents=True,exist_ok=True)
            if policy['adaptive']:ratio=adaptive_ratio(ratio,initial_metrics['normal_mse'],None if stage==0 else previous['normal_mse'])
            status('training',round=policy['round'],seed=seed,stage=stage+1,replay_ratio=ratio)
            accepted={};routing=[]
            for c in clients:
                items=c['candidate_stages'][stage]
                accepted[c['id']]=[r for r in items if r['y']==0]
                routing.extend(dict(id=r['id'],client=c['id'],verified='normal' if r['y']==0 else 'attack') for r in items)
            B.atomic(folder/'routing.json',dict(environment='정상 여부를 확인할 수 있는 검증자 보조 환경',items=routing))
            if min(len(x) for x in accepted.values())<6:raise ValueError('Insufficient verified normal stream; no forced effect')
            train={};replay_count=0
            for c in clients:
                train[c['id']],k=mix(accepted[c['id']],memory[c['id']],ratio,seed+stage*1000+c['id']);replay_count+=k
            B.atomic(folder/'training_input_ids.json',{str(k):[r['id'] for r in rs] for k,rs in train.items()})
            previous_state=copy.deepcopy(model.state_dict());previous_ref=reference
            model,seconds,history=B.fit(model,clients,train,'new_valid',config['update_rounds'],seed+100000+stage,folder/'training')
            cumulative+=seconds;actual=sum(r['optimizer_steps'] for rr in history for r in rr['local']);steps+=actual
            candidate=evaluate_development(model,clients,train,folder/'candidate',config)
            if policy['gate']:accept,reason=gate_accept(candidate,previous,initial_metrics)
            else:accept,reason=True,dict(policy='always update')
            if accept:
                reference=train;published=candidate
                (folder/'published').mkdir(exist_ok=True)
                for name in ['predictions.csv.gz','detector_bundle.pt','metrics.json']:shutil.copyfile(folder/'candidate'/name,folder/'published'/name)
            else:
                model.load_state_dict(previous_state);reference=previous_ref
                published=evaluate_development(model,clients,reference,folder/'published',config)
            record=dict(round=policy['round'],seed=seed,stage=stage+1,ratio=ratio,policy=policy,accepted=accept,gate_reason=reason,metrics=published,candidate_metrics=candidate,development_score=development_score(published),initial_development_score=development_score(initial_metrics),old_auc_change=published['old']['mean_client_auc']-initial_metrics['old']['mean_client_auc'],old_mse_change=published['normal_mse']['old']-initial_metrics['normal_mse']['old'],training_seconds=seconds,cumulative_training_seconds=cumulative,optimizer_steps=actual,cumulative_optimizer_steps=steps,replay_samples=replay_count,replay_feature_bytes=config['clients']*config['capacity']*115*8,state_sha256=weights_hash(model))
            B.atomic(folder/'record.json',record);records.append(record);B.atomic(out/policy['round']/'development_results.json',records)
            previous=published
    summary=analyze(policy,records,config);summary['wall_seconds']=time.perf_counter()-begin
    B.atomic(out/policy['round']/'analysis.json',summary)
    return summary


def analyze(policy,records,config):
    last=[r for r in records if r['stage']==config['stages']]
    result=dict(round=policy['round'],policy=policy,records=len(records),development_score_mean=statistics.mean(r['development_score'] for r in last),training_seconds_mean=statistics.mean(r['cumulative_training_seconds'] for r in last),accepted_updates=sum(r['accepted'] for r in records),total_updates=len(records),metrics={})
    for domain in ['old','new']:
        result['metrics'][domain]={k:dict(mean=statistics.mean(r['metrics'][domain][k] for r in last),std=statistics.stdev(r['metrics'][domain][k] for r in last) if len(last)>1 else None) for k in ['mean_client_auc','fpr','tpr']}
    result['old_auc_change_mean']=statistics.mean(r['old_auc_change'] for r in last);result['old_mse_change_mean']=statistics.mean(r['old_mse_change'] for r in last)
    result['interpretation']=dict(adaptation='Development-only; final comparison is still locked',retention='Existing mean AUC decreased' if result['old_auc_change_mean']<0 else 'No mean old AUC decrease',memory='Logical feature payload only, not total RSS',causality='CEN reference is refit per current normal training set; model/reference effects coexist')
    return result


def final_evaluation(final_rows,out,config,selection_sha,status):
    # 저장된 scaler·모델·CEN·임계값을 그대로 사용한다. 최종 자료로 fit하지 않는다.
    final=[]
    for label in ['initial']+list('ABCDE'):
        for seed in config['seeds']:
            stages=[0] if label=='initial' else range(1,config['stages']+1)
            for stage in stages:
                status('final_evaluation',round=label,seed=seed,stage=stage)
                src=out/'initial'/f'seed_{seed}' if label=='initial' else out/label/f'seed_{seed}'/f'stage_{stage}'/'published'
                bundle=torch.load(src/'detector_bundle.pt',map_location='cpu',weights_only=False)
                model=B.Shrink_Autoencoder(**bundle['architecture']).to(config['device']);model.load_state_dict(bundle['model_state']);model.eval()
                result={};predictions=[];reconstruction={}
                for domain,nkey,akey in [('old','old_normal','old_attack'),('new','new_normal','new_attack')]:
                    cm=np.zeros((2,2),dtype=np.int64);aucs=[];err=0.;nvalues=0
                    for cid in range(1,config['clients']+1):
                        c=bundle['clients'][cid];normal=final_rows[f'{cid}/{nkey}'];items=normal+final_rows[f'{cid}/{akey}'];x=c['input_scaler'].transform(B.payload(items));lat=[]
                        with torch.no_grad():
                            for i in range(0,len(x),256):
                                batch=torch.tensor(x[i:i+256],dtype=torch.float32,device=config['device']);z,recon,_=model(batch);lat.append(z.cpu().numpy())
                                take=max(0,min(len(batch),len(normal)-i))
                                if take:err+=float(torch.sum((batch[:take]-recon[:take])**2).item());nvalues+=batch[:take].numel()
                        score=c['cen'].get_density(np.concatenate(lat));threshold=c['threshold'];y=np.array([r['y'] for r in items]);prediction=(score>threshold).astype(int)
                        if not np.isfinite(score).all():raise ValueError('Nonfinite final scores')
                        cm+=confusion_matrix(y,prediction,labels=[0,1]);aucs.append(float(roc_auc_score(y,score)))
                        predictions.extend(dict(id=r['id'],domain=domain,client=cid,label=int(yy),score=float(ss),prediction=int(pp),threshold=float(threshold)) for r,yy,ss,pp in zip(items,y,score,prediction))
                    tn,fp,fn,tp=cm.ravel();result[domain]=dict(mean_client_auc=float(np.mean(aucs)),fpr=float(fp/(tn+fp)),tpr=float(tp/(tp+fn)),confusion_matrix=cm.tolist());reconstruction[domain]=err/nvalues
                folder=out/'final'/label/f'seed_{seed}'/f'stage_{stage}';folder.mkdir(parents=True,exist_ok=True)
                with gzip.open(folder/'predictions.csv.gz','wt',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
                record=dict(round=label,seed=seed,stage=stage,metrics=result,reconstruction_mse=reconstruction,selection_sha256=selection_sha,bundle_sha256=hashlib.sha256((src/'detector_bundle.pt').read_bytes()).hexdigest())
                B.atomic(folder/'evaluation.json',record);final.append(record);B.atomic(out/'final_results.json',final)
    return final
