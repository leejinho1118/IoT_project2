"""A 선택의 민감도·시드별 차이·논문/코드 불일치 근거를 보존한다."""
from pathlib import Path
import json,itertools,statistics,hashlib
import numpy as np
ROOT=Path.cwd();O=ROOT/'continual/research_ABCDE/output/campaign_full_v1';D=ROOT/'continual/head_audit'
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
analyses=load(O/'analyses.json');final=load(O/'final_results.json');ranking=[]
for old_weight in [.25,.5,.75]:
 for penalty in [0,.1,.5,1.]:
  scores={r['round']:old_weight*r['metrics']['old']['mean_client_auc']['mean']+(1-old_weight)*r['metrics']['new']['mean_client_auc']['mean']-penalty*max(0,max(r['metrics']['old']['fpr']['mean'],r['metrics']['new']['fpr']['mean'])-.1) for r in analyses}
  ranking.append(dict(old_weight=old_weight,fpr_penalty=penalty,winner=max(scores,key=scores.get),scores=scores))
paired=[];indices=np.array(list(itertools.product(range(5),repeat=5)))
for label in 'BCDE':
 item={'candidate':label,'baseline':'A','scope':'Paired five-seed descriptive sensitivity; resampled seed interval is not an external/general-population guarantee'}
 for domain in ['old','new']:
  vals=[]
  for seed in [0,10000,20000,30000,40000]:
   a=next(r for r in final if (r['round'],r['seed'],r['stage'])==('A',seed,3));b=next(r for r in final if (r['round'],r['seed'],r['stage'])==(label,seed,3));vals.append(b['metrics'][domain]['mean_client_auc']-a['metrics'][domain]['mean_client_auc'])
  boot=np.asarray(vals)[indices].mean(axis=1);item[domain]=dict(differences=vals,mean=statistics.mean(vals),improved_seeds=sum(x>0 for x in vals),descriptive_seed_resample_interval=np.quantile(boot,[.025,.975]).tolist())
 paired.append(item)
risks=[
 dict(id='objective',severity='high',finding='A wins one composite development objective, not every metric. Ranking changes with old/new priority and FPR penalty.',evidence='objective_sensitivity in this file'),
 dict(id='threshold_shift',severity='high',finding='Threshold_source_ids only calib_old; calib_new normals are not used for current threshold. New FPR around58% is a deployment failure.',evidence='research_ABCDE/engine.py evaluate_development'),
 dict(id='one_center',severity='hypothesis',finding='Single standardized centroid can misrepresent multiple normal modes. Must test alternative heads on exactly frozen embeddings before blaming training.'),
 dict(id='test_reuse',severity='high',finding='Same heldout pools were opened in V2 and A-E. Further comparison is exploratory, not a new untouched confirmation.'),
 dict(id='calibration_provenance',severity='high',finding='Initial models used original validation pools for early stopping before those pools were divided into local/calibration/decision. No exact split-conformal coverage guarantee is claimed.'),
 dict(id='paper_code_shrink',severity='high',finding='Paper Eq4/Algorithm1 uses squared latent L2. Upstream code uses mean unsquared L2 norm. Repository-faithful reproduction is not proof of paper-faithful mathematical implementation.',paper='https://arxiv.org/html/2410.14121v2',code='src/Model/Shrink_Autoencoder.py:152',numeric_example='latent=(3,4): unsquared norm=5; squared norm=25'),
 dict(id='paper_code_sharing',severity='high',finding='Paper GatewayUpdate receives initial encoder/decoder weights; upstream main passes shared aggregator model sequentially. Preserved, not silently fixed.'),
 dict(id='early_stop',severity='medium',finding='Two independent global if statements increment patience even on improvement. Preserved upstream behavior changes actual training budget.'),
 dict(id='development_attack_reuse',severity='medium',finding='Old/new development domains use same attack pool, so development TPRs are identical by design. This is not unseen-attack validation.'),
 dict(id='synthetic_severity',severity='high',finding='A/B split is defined by development PC1 median, not actual chronology. Initial new AUC<0.5 can be driven by constructed feature shift. Need independent real chronological/device data for generalization.'),
 dict(id='clip',severity='hypothesis',finding='Fixed input clip may erase shifted-feature distinctions. Quantify saturation separately for old/new normal and attacks and compare latent vs input-space heads.'),
 dict(id='five_seeds',severity='medium',finding='Five seeds share a single split. Client/seed results are correlated; do not claim statistical significance or general deployment performance.'),
]
sources=[
 dict(title='FedMSE original paper, Section4.1 and Algorithm1',url='https://arxiv.org/html/2410.14121v2',supports='SAE-CEN, squared mathematical shrink term, receive initial gateway weights; not proof of our implementation'),
 dict(title='PatchCore, CVPR2022 official paper',url='https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf',supports='Normal feature memory and nearest-neighbor anomaly score as a transferable hypothesis. Image benchmark performance is not IoT evidence.'),
 dict(title='scikit-learn novelty/outlier detection',url='https://scikit-learn.org/stable/modules/outlier_detection.html',supports='Normal-only novelty setting; LOF novelty=True and evaluate only unseen queries'),
 dict(title='NearestNeighbors official API',url='https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html',supports='Nearest normal reference distances'),
 dict(title='LedoitWolf official API',url='https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html',supports='Regularized covariance estimate; anisotropic latent-distance diagnostic'),
 dict(title='Conformal Prediction Beyond Exchangeability',url='https://www.stat.berkeley.edu/~ryantibs/papers/nexcp.pdf',supports='Exchangeability/model symmetry assumptions and problems under shift. Empirical quantile here is not a distribution-free guarantee.'),
]
(D/'risk_audit.json').write_text(json.dumps(dict(risks=risks,objective_sensitivity=ranking,paired_seed_sensitivity=paired,sources=sources),ensure_ascii=False,indent=2),encoding='utf-8')
(D/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
print('Sensitivity winners:',sorted({r['winner'] for r in ranking}));print('Risks recorded',len(risks))
