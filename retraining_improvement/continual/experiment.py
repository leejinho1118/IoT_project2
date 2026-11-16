"""Verifier-assisted continual SAE-CEN experiment; upstream files never edited."""
import os,sys,argparse,json,time,copy,random,hashlib,datetime,logging,pickle,csv,gzip,traceback,subprocess
from pathlib import Path
P=argparse.ArgumentParser()
P.add_argument('--profile',choices=['smoke','full'],default='smoke')
P.add_argument('--device',choices=['cpu','cuda'],default='cpu')
P.add_argument('--run-id',default='smoke_v1')
P.add_argument('--prepare-only',action='store_true')
ARGS=P.parse_args()
if ARGS.device=='cpu': os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,confusion_matrix
from torch.utils.data import DataLoader
from Model import Shrink_Autoencoder
from Model.Centroid import CentroidBasedOneClassClassifier
from DataLoader import IoTDataset
from Trainer import ClientTrainer,GlobalAggregator
from protocol import fingerprint,verify_candidates,fixed_budget_mix,check_partition

OUT=ROOT/'continual/output'/ARGS.run_id
if any(x in ARGS.run_id for x in ['..','/','\\']): raise ValueError('run-id must be a single directory name')
OUT.mkdir(parents=True,exist_ok=True)
import msvcrt
LOCK=open(OUT/'run.lock','a+b'); LOCK.seek(0); LOCK.write(b'1'); LOCK.flush(); LOCK.seek(0); msvcrt.locking(LOCK.fileno(),msvcrt.LK_NBLCK,1)
if (OUT/'complete.json').exists(): print('Already complete:',OUT); sys.exit(0)
if (OUT/'status.json').exists(): raise RuntimeError('Existing unfinished run preserved. Choose a new run-id, do not overwrite.')
SMOKE=ARGS.profile=='smoke'
CFG=dict(profile=ARGS.profile,device=ARGS.device,clients=4 if SMOKE else 50,normal_cap=120 if SMOKE else None,new_cap=120 if SMOKE else None,attack_cap=120 if SMOKE else None,local_epochs=2 if SMOKE else 100,initial_rounds=2 if SMOKE else 20,update_rounds=2 if SMOKE else 20,stages=2 if SMOKE else 3,seeds=[0] if SMOKE else [0,10000,20000,30000,40000],batch_size=12,lr=1e-5,shrink_lambda=10,replay_capacity=24 if SMOKE else 128,replay_fraction=.25,normal_validation_quantile=.95,data_seed=1234,min_verified_normal=6,verifier_uncertainty_rate=.1 if SMOKE else 0)
STATUS=dict(state='initializing',pid=os.getpid(),started=datetime.datetime.now().astimezone().isoformat(),config=CFG)

def atomic(path,obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    for i in range(100):
        try: os.replace(tmp,path); return
        except PermissionError:
            if i==99: raise
            time.sleep(.05)
def status(**kwargs):
    STATUS.update(kwargs); STATUS['updated']=datetime.datetime.now().astimezone().isoformat(); atomic(OUT/'status.json',STATUS)
def seeds(n): random.seed(n); np.random.seed(n); torch.manual_seed(n)
def source_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))}
def base_complete():
    p=ROOT/'original_output/status.json'
    if not p.exists(): return False
    s=json.loads(p.read_text(encoding='utf-8-sig'))
    if s.get('state')!='complete' or s.get('exit_code')!=0: return False
    files=list((ROOT/'src/Checkpoint/Results').rglob('*_results.json'))
    return len(files)==30 and all(p.stat().st_size>0 for p in files)
def payload(rows): return np.asarray([r['x'] for r in rows],dtype=np.float64).reshape(-1,115)
def metadata(rows): return [{k:v for k,v in r.items() if k!='x'} for r in rows]
def load_plan():
    clients=[]; groups={}; seen=set(); removed=[]; rng=np.random.default_rng(CFG['data_seed'])
    for cid in range(1,CFG['clients']+1):
        pools={}
        for kind,cap in [('normal',CFG['normal_cap']),('test_normal',CFG['new_cap']),('abnormal',CFG['attack_cap'])]:
            df=pd.read_csv(ROOT/'Data/N-BaIoT/IID-50-Client_Data'/f'Client-{cid}'/kind/'data.csv',header=None)
            if df.shape[1]!=115: raise ValueError('Expected 115 features')
            indices=rng.permutation(len(df))
            if cap is not None: indices=indices[:cap]
            rows=[]
            for idx in indices:
                x=df.iloc[int(idx)].to_numpy(dtype=np.float64); fp=fingerprint(x); rid=f'Client-{cid}/{kind}/{idx}'
                if fp in seen: removed.append(rid); continue
                seen.add(fp); rows.append(dict(id=rid,fingerprint=fp,client=cid,source=kind,y=1 if kind=='abnormal' else 0,x=x))
            pools[kind]=rows
        normal=pools['normal']; new=pools['test_normal']; attack=pools['abnormal']
        n=len(normal); a=int(.4*n); b=int(.1*n); d=int(.4*n)
        nn=len(new); q=int(.4*nn); v=int(.2*nn)
        na=len(attack); qa=int(.2*na); va=int(.4*na)
        c=dict(id=cid,train=normal[:a],valid=normal[a:a+b],dev=normal[a+b:a+b+d],old_normal=normal[a+b+d:],new_valid=new[q:q+v],new_normal=new[q+v:],old_attack=attack[qa:qa+va],new_attack=attack[qa+va:])
        candidates=new[:q]+attack[:qa]; rng.shuffle(candidates)
        c['candidate_stages']=[list(x) for x in np.array_split(np.asarray(candidates,dtype=object),CFG['stages'])]
        if min(len(c[k]) for k in ['train','valid','dev','old_normal','new_valid','new_normal','old_attack','new_attack'])<2: raise ValueError('Insufficient partition after deduplication')
        for key in ['train','valid','dev','old_normal','new_valid','new_normal','old_attack','new_attack']: groups[f'{cid}/{key}']=c[key]
        for j,rs in enumerate(c['candidate_stages']): groups[f'{cid}/candidate_{j}']=rs
        scaler=StandardScaler().fit(payload(c['train'])); c['scaler']=scaler
        c['replay']=c['train'][:CFG['replay_capacity']]
        clients.append(c)
    counts=check_partition(groups)
    with gzip.open(OUT/'split_manifest.json.gz','wt',encoding='utf-8') as f: json.dump({k:metadata(v) for k,v in groups.items()},f,ensure_ascii=False)
    report=dict(counts=counts,total_unique_rows=sum(counts.values()),duplicates_removed=len(removed),removed_ids=removed,scenario='Simulated arrivals from prepared IID-50 CSV; physical device and chronology not recoverable.',selection='All arrival candidates are verified; anomaly flag is recorded but is not treated as truth.',leakage_policy='Global SHA-256 feature-vector deduplication before partition, first occurrence retained. Normalized -0; all features finite.',baseline_difference='New continual-learning partition removes duplicates and reserves arrival/validation/future data. This is separate from unmodified original main.py results.')
    atomic(OUT/'split_audit.json',report)
    return clients

def loader(c,rows):
    if any(x['y']!=0 for x in rows): raise ValueError('Attack entered normal training')
    x=c['scaler'].transform(payload(rows)); return DataLoader(IoTDataset(x,np.zeros(len(x))),batch_size=CFG['batch_size'],pin_memory=ARGS.device=='cuda')

def fit(model,clients,train_rows,validation_key,rounds,seed,folder):
    seeds(seed)
    agg=GlobalAggregator(model,update_type='mse_avg')
    minimum=min(len(c['dev']) for c in clients)
    rr=np.random.default_rng(seed)
    dev=np.concatenate([payload(c['dev'])[rr.choice(len(c['dev']),minimum,replace=False)] for c in clients])
    agg.create_dev_dataset({'dataset':dev})
    selection=random.Random(seed); history=[]; best=float('inf'); worse=0; begin=time.perf_counter()
    for r in range(rounds):
        chosen=selection.sample(range(len(clients)),max(1,int(.5*len(clients))))
        weights=[]; trained=[]
        total=sum(len(train_rows[clients[i]['id']]) for i in chosen)
        for i in chosen:
            c=clients[i]; rows=train_rows[c['id']]
            # Deliberately retain upstream shared-model sequential semantics.
            trainer=ClientTrainer(model=agg.model,save_dir=str(folder/f'round_{r+1}'/f'client_{c["id"]}'),epoch=CFG['local_epochs'],lr_rate=CFG['lr'],update_type='mse_avg')
            trainer.run(loader(c,rows),loader(c,c[validation_key]))
            weights.append((copy.deepcopy(trainer.model.state_dict()),total,len(rows)))
            tracking=Path(trainer.save_dir)/'training_tracking.pkl'
            with tracking.open('rb') as f: epochs=len(pickle.load(f))
            trained.append(dict(client=c['id'],rows=len(rows),epochs=epochs,optimizer_steps=epochs*len(loader(c,rows))))
        agg.update(weights)
        history.append(dict(round=r+1,participants=[clients[i]['id'] for i in chosen],global_loss=agg.val_loss,local=trained))
        # Preserve upstream two-if early-stop behavior exactly.
        if agg.val_loss<best: best=agg.val_loss; worse=0
        if agg.val_loss>=best:
            worse+=1
            if worse>1: break
    elapsed=time.perf_counter()-begin
    atomic(folder/'training.json',dict(seconds=elapsed,rounds=history,original_model_sharing=True,original_global_early_stop=True,max_rounds=rounds))
    return agg.model,elapsed,history

def encode(model,c,rows):
    x=c['scaler'].transform(payload(rows)); values=[]; device=next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for start in range(0,len(x),256): values.append(model(torch.tensor(x[start:start+256],dtype=torch.float32,device=device))[0].cpu().numpy())
    return np.concatenate(values)
def detector(model,c,reference):
    if any(r['y']!=0 for r in reference): raise ValueError('Attack entered CEN reference')
    cen=CentroidBasedOneClassClassifier(); cen.fit(encode(model,c,reference))
    threshold=float(np.quantile(cen.get_density(encode(model,c,c['valid'])),CFG['normal_validation_quantile']))
    return cen,threshold

def evaluate(model,clients,reference,folder,stage):
    folder.mkdir(parents=True,exist_ok=True); result={}; detectors={}; prediction_rows=[]
    for c in clients: detectors[c['id']]=detector(model,c,reference[c['id']])
    for domain,normal_key,attack_key in [('old','old_normal','old_attack'),('new','new_normal','new_attack')]:
        rows=[]; cms=np.zeros((2,2),dtype=int)
        for c in clients:
            cen,threshold=detectors[c['id']]; items=c[normal_key]+c[attack_key]
            y=np.array([x['y'] for x in items]); score=cen.get_density(encode(model,c,items)); pred=(score>threshold).astype(int)
            if not np.isfinite(score).all(): raise ValueError('Nonfinite anomaly score')
            cm=confusion_matrix(y,pred,labels=[0,1]); cms+=cm; tn,fp,fn,tp=cm.ravel()
            rows.append(dict(client=c['id'],auc=float(roc_auc_score(y,score)),fpr=float(fp/(tn+fp)),tpr=float(tp/(tp+fn)),threshold=threshold,normal=int(tn+fp),attack=int(tp+fn)))
            prediction_rows.extend(dict(id=x['id'],domain=domain,client=c['id'],label=int(yy),score=float(ss),prediction=int(pp),threshold=threshold) for x,yy,ss,pp in zip(items,y,score,pred))
        tn,fp,fn,tp=cms.ravel(); result[domain]=dict(mean_client_auc=float(np.mean([x['auc'] for x in rows])),fpr=float(fp/(tn+fp)),tpr=float(tp/(tp+fn)),confusion_matrix=cms.tolist(),clients=rows)
    atomic(folder/'metrics.json',result)
    with gzip.open(folder/'predictions.csv.gz','wt',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    bundle=dict(stage=stage,model_state={k:v.cpu().clone() for k,v in model.state_dict().items()},architecture=dict(input_dim=115,output_dim=115,hidden_neus=50,latent_dim=11,shrink_lambda=10),clients={c['id']:dict(input_scaler=c['scaler'],cen=detectors[c['id']][0],threshold=detectors[c['id']][1],threshold_source_ids=[x['id'] for x in c['valid']],reference_ids=[x['id'] for x in reference[c['id']]]) for c in clients},config=CFG)
    torch.save(bundle,folder/'detector_bundle.pt')
    return result,detectors

def oracle(labels):
    def verify(rid):
        # Uncertainty simulator depends only on ID, never on a test-set metric.
        if int(hashlib.sha256(rid.encode()).hexdigest()[:8],16)%10000<int(CFG['verifier_uncertainty_rate']*10000): return 'uncertain'
        return 'normal' if labels[rid]==0 else 'attack'
    return verify

def select_stage(model,clients,reference,stage,folder):
    accepted={}; routes=[]; audit=[]
    for c in clients:
        items=c['candidate_stages'][stage]
        cen,threshold=detector(model,c,reference[c['id']]); scores=cen.get_density(encode(model,c,items))
        lookup={r['id']:r['y'] for r in items}
        normal,attacks,uncertain=verify_candidates(items,oracle(lookup)); accepted[c['id']]=normal
        categories={r['id']:kind for kind,group in [('normal',normal),('attack',attacks),('uncertain',uncertain)] for r in group}
        for item,score in zip(items,scores): routes.append(dict(id=item['id'],client=c['id'],anomaly_flag=bool(score>threshold),score=float(score),threshold=threshold,verified=categories[item['id']]))
        audit.append(dict(client=c['id'],arrivals=len(items),accepted_normal=len(normal),quarantined_attack=len(attacks),uncertain=len(uncertain)))
    atomic(folder/'routing.json',dict(environment='정상 여부를 확인할 수 있는 검증자 보조 환경',policy='All candidates verified; detection score is not ground truth',clients=audit,items=routes))
    return accepted

def log_current_detection(model,clients,reference,stage,folder):
    rows=[]
    for c in clients:
        items=c['candidate_stages'][stage]
        cen,threshold=detector(model,c,reference[c['id']])
        score=cen.get_density(encode(model,c,items))
        rows.extend(dict(id=x['id'],client=c['id'],score=float(v),anomaly_flag=bool(v>threshold),threshold=threshold) for x,v in zip(items,score))
    atomic(folder/'detections_before_update.json',dict(policy='Current arm model detects arrivals before this update. Shared verifier outcomes are fixed across arms; no prediction is treated as a true label.',items=rows))

def main():
    if not SMOKE and not ARGS.prepare_only and not base_complete(): raise RuntimeError('Full comparison waits for normal completion of all 30 original experiments; GPU baseline is not disturbed.')
    if ARGS.device=='cuda' and not base_complete(): raise RuntimeError('No concurrent GPU experiment while original baseline runs.')
    if not SMOKE and not ARGS.prepare_only: subprocess.run([sys.executable,str(ROOT/'continual/summarize_original.py')],check=True)
    torch.set_num_threads(1)
    if not SMOKE and not ARGS.prepare_only:
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if ARGS.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if ARGS.device=='cpu' and torch.cuda.is_available(): raise RuntimeError('CPU isolation failed')
    rootlog=logging.getLogger(); rootlog.handlers.clear(); rootlog.setLevel(logging.INFO); rootlog.addHandler(logging.FileHandler(OUT/'training.log',encoding='utf-8'))
    before=source_hashes(); status(state='preparing',source_hashes=before,extension_source_hashes={name:hashlib.sha256((ROOT/'continual'/name).read_bytes()).hexdigest() for name in ['experiment.py','protocol.py','infer.py']},environment=dict(python=sys.version,torch=torch.__version__,pandas=pd.__version__,numpy=np.__version__,cuda=torch.version.cuda))
    clients=load_plan(); atomic(OUT/'config.json',CFG)
    if ARGS.prepare_only: status(state='prepared'); return
    all_metrics=[]
    for seed in CFG['seeds']:
        seeds(seed); status(state='initial_training',seed=seed)
        fresh=Shrink_Autoencoder(input_dim=115,output_dim=115,hidden_neus=50,latent_dim=11,shrink_lambda=10)
        initial,initial_seconds,history=fit(fresh,clients,{c['id']:c['train'] for c in clients},'valid',CFG['initial_rounds'],seed,OUT/f'seed_{seed}'/'initial_training')
        initial_state=copy.deepcopy(initial.state_dict()); initial_device=next(initial.parameters()).device; initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in initial_state.values())).hexdigest(); initial_reference={c['id']:c['train'] for c in clients}
        base_result,_=evaluate(initial,clients,initial_reference,OUT/f'seed_{seed}'/'initial_evaluation',0)
        # One verified stream shared across all arms; verification does not use future labels.
        verified=[]
        for stage in range(CFG['stages']):
            folder=OUT/f'seed_{seed}'/f'verified_stage_{stage+1}'; folder.mkdir(parents=True,exist_ok=True)
            verified.append(select_stage(initial,clients,initial_reference,stage,folder))
        for arm in ['no_update','new_only','replay']:
            model=Shrink_Autoencoder(input_dim=115,output_dim=115,hidden_neus=50,latent_dim=11,shrink_lambda=10).to(initial_device); model.load_state_dict(initial_state)
            if next(model.parameters()).device!=initial_device: raise RuntimeError('Arm device differs from common initial model')
            reference=initial_reference; cumulative=0.
            for stage,new in enumerate(verified):
                status(state='updating',seed=seed,arm=arm,stage=stage+1)
                folder=OUT/f'seed_{seed}'/arm/f'stage_{stage+1}'; folder.mkdir(parents=True,exist_ok=True)
                log_current_detection(model,clients,reference,stage,folder)
                accepted_min=min(len(x) for x in new.values())
                replay_count=0; rows={}; skipped=False
                if arm!='no_update' and accepted_min>=CFG['min_verified_normal']:
                    for c in clients:
                        if arm=='replay': mixed,count=fixed_budget_mix(new[c['id']],c['replay'],CFG['replay_fraction'],np.random.default_rng(seed+stage*1000+c['id'])); rows[c['id']]=mixed; replay_count+=count
                        else: rows[c['id']]=new[c['id']]
                    model,seconds,history=fit(model,clients,rows,'new_valid',CFG['update_rounds'],seed+100000+stage,folder/'training'); cumulative+=seconds; reference=rows
                else: seconds=0.; skipped=arm!='no_update'
                result,_=evaluate(model,clients,reference,folder,stage+1)
                record=dict(seed=seed,arm=arm,stage=stage+1,metrics=result,initial_training_seconds=initial_seconds,additional_training_seconds=seconds,cumulative_additional_seconds=cumulative,replay_samples_this_stage=replay_count,replay_bank_samples=sum(len(c['replay']) for c in clients) if arm=='replay' else 0,replay_bank_feature_bytes=sum(len(c['replay'])*115*8 for c in clients) if arm=='replay' else 0,minimum_verified_new_normal=accepted_min,update_skipped=skipped,old_auc_change=result['old']['mean_client_auc']-base_result['old']['mean_client_auc'],old_fpr_change=result['old']['fpr']-base_result['old']['fpr'],state_sha256=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest())
                if arm=='no_update' and (record['state_sha256']!=initial_hash or result!=base_result): raise RuntimeError('No-update model or evaluation differs from initial baseline')
                record['model_device']=str(next(model.parameters()).device)
                atomic(folder/'evaluation.json',record); all_metrics.append(record); atomic(OUT/'results.json',all_metrics)
    after=source_hashes()
    if before!=after: raise RuntimeError('Upstream source changed during extension test')
    checks=dict(upstream_sources_unchanged=True,cpu_isolation=not torch.cuda.is_available() if ARGS.device=='cpu' else None,three_arms=sorted(set(x['arm'] for x in all_metrics)),baseline_unchanged_across_stages=all(len({x['state_sha256'] for x in all_metrics if x['seed']==s and x['arm']=='no_update'})==1 for s in CFG['seeds']),normal_training_only=True,partition_audit='split_audit.json',threshold='95th percentile of fixed old normal validation scores; CEN refit using each arm normal reference; no test-label tuning',limitations=['Simulated arrivals; no physical-device or chronological metadata','Separate initial training cost; original global model not exported','Smoke profile is an implementation check, not a full performance result'] if SMOKE else ['Simulated arrivals; no physical-device or chronological metadata','Separate initial training and split; do not pool numerically with original baseline'])
    if not checks['baseline_unchanged_across_stages']: raise RuntimeError('No-update baseline mutated')
    atomic(OUT/'checks.json',checks); atomic(OUT/'complete.json',dict(completed=datetime.datetime.now().astimezone().isoformat(),config=CFG,records=len(all_metrics),checks=checks)); status(state='complete')
    print(json.dumps(dict(output=str(OUT),records=len(all_metrics),checks=checks),ensure_ascii=False),flush=True)
if __name__=='__main__':
    try: main()
    except BaseException:
        status(state='failed',error=traceback.format_exc()); raise





