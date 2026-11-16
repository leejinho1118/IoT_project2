import sys,json,gzip,math
from pathlib import Path
sys.path.insert(0,str(Path('src').resolve()))
import torch
p=Path('continual/output/full_v1')
with gzip.open(p/'split_manifest.json.gz','rt',encoding='utf-8') as f:g=json.load(f)
by={r['id']:(k,r) for k,rows in g.items() for r in rows};count=0
for seed in [0,10000,20000,30000,40000]:
 initial=torch.load(p/f'seed_{seed}/initial_evaluation/detector_bundle.pt',map_location='cpu',weights_only=False)
 for arm in ['no_update','new_only','replay']:
  for stage in [1,2,3]:
   b=torch.load(p/f'seed_{seed}/{arm}/stage_{stage}/detector_bundle.pt',map_location='cpu',weights_only=False)
   assert len(b['clients'])==50
   for cid,c in b['clients'].items():
    assert math.isfinite(c['threshold'])
    assert c['threshold_source_ids']==initial['clients'][cid]['threshold_source_ids']
    assert all(by[i][0]==f'{cid}/valid' and by[i][1]['y']==0 for i in c['threshold_source_ids'])
    assert all(by[i][1]['y']==0 and by[i][0].endswith(('/train',f'/candidate_{stage-1}')) for i in c['reference_ids'])
    if arm=='no_update':assert c['reference_ids']==initial['clients'][cid]['reference_ids']
    if arm=='new_only':assert all(by[i][0]==f'{cid}/candidate_{stage-1}' for i in c['reference_ids'])
    if arm=='replay':assert any(by[i][0]==f'{cid}/train' for i in c['reference_ids'])
   count+=1
v=json.loads((p/'validation.json').read_text());v['bundle_reference_validation']={'passed':True,'bundles':count,'checks':['Only normal training/candidate reference IDs','No final evaluation IDs in references or threshold calibration','Fixed initial validation IDs across all arms','Replay includes initial normal; new_only only current verified normal']};(p/'validation.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');print('45 bundles and 2250 client reference/threshold states passed')
