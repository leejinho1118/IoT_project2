"""완료된 비교를 검증한 뒤 과장 없는 최종 요약과 보존 검사를 기록한다."""
import json,hashlib,csv,statistics,datetime,argparse
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('--run-id',required=True);a=P.parse_args()
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'continual/drift_v2';O=D/'output'/a.run_id
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
v=load(O/'validation.json');cfg=load(O/'config.json');records=load(O/'results.json');complete=load(O/'complete.json')
assert v['validation']=='passed'
protected=load(D/'integrity_before.json')
for path,h in protected.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
source=load(O/'split_audit.json')['source_csv_sha256']
for path,h in source.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
summary=v['summary'];rows=[]
for x in summary:
 rows.append(dict(arm=x['arm'],old_auc_mean=x['old']['mean_client_auc']['mean'],old_auc_std=x['old']['mean_client_auc']['std'],new_auc_mean=x['new']['mean_client_auc']['mean'],new_auc_std=x['new']['mean_client_auc']['std'],old_fpr=x['old']['fpr']['mean'],new_fpr=x['new']['fpr']['mean'],old_tpr=x['old']['tpr']['mean'],new_tpr=x['new']['tpr']['mean'],old_auc_change=x['old_auc_change']['mean'],old_reconstruction_change=x['old_reconstruction_change']['mean'],extra_seconds=x['cumulative_additional_seconds']['mean'],optimizer_steps=x['cumulative_optimizer_steps']['mean'],replay_feature_bytes=x['replay_feature_bytes']['mean']))
with (O/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
initial_times=[next(x['initial_training_seconds'] for x in records if x['seed']==seed) for seed in cfg['seeds']]
lines=['# FedMSE 分布 변화 지속학습 비교 결과'.replace('分布','분포'),'20222038 이진호 / '+datetime.datetime.now().astimezone().isoformat(),'독립 검증 완료: '+str(v['records'])+'개 평가 / '+cfg['profile']+' 프로필.','| 방식 | 기존 AUC | 신규 AUC | 기존 FPR | 신규 FPR | 신규 TPR |','|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['arm']} | {r['old_auc_mean']:.5f} | {r['new_auc_mean']:.5f} | {r['old_fpr']:.5f} | {r['new_fpr']:.5f} | {r['new_tpr']:.5f} |")
lines+=['','## 과거 성능 유지와 해석','양수 old_auc_change는 초기보다 AUC가 상승한 것이며, 음수면 저하다. old_reconstruction_change는 양수일수록 과거 정상 복원 오차가 커진 것이다. 임계값 재보정 FPR과 frozen_initial_threshold를 함께 읽는다.','CEN 기준 자료도 방식별로 달라지므로 순수 가중치 변화만 분리한 인과 효과로 해석하지 않는다.','','| 방식 | 기존 AUC 변화 | 기존 정상 MSE 변화 | 추가 학습 초 | 실제 스텝 | 보관 특징 바이트 |','|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['arm']} | {r['old_auc_change']:+.6f} | {r['old_reconstruction_change']:+.6f} | {r['extra_seconds']:.1f} | {r['optimizer_steps']:.0f} | {r['replay_feature_bytes']:.0f} |")
lines+=['','공통 초기 학습 총 시간(시드당 한 번): '+str(sum(initial_times))+'초. 전체 벽시계 '+str(complete['wall_seconds'])+'초.','입력 전처리: 초기 A 표준화 + 고정 [-20,20] 포화. CPU 개발 실패를 교정하기 위한 수치 안정화이며 모든 비교 방식에 동일하다. 원본/V1과 수치를 직접 빼서 개선 폭으로 주장하지 않는다.','무작위 시드 반복이지 독립 실제 환경 표본은 아니다. 통계적 유의성은 주장하지 않는다. 데이터는 모의 특징 분포 변화이며 실제 시간 드리프트가 아니다. 후보 정답을 검증자로 쓰는 정상 여부를 확인할 수 있는 검증자 보조 환경이다.','보관량32/128 비교는 25% 대체율 상한과 한 단계 후보 수 때문에 실제 리플레이 투입 수가 같을 수 있다. 모델 설정이 같아도 동일한 결과는 가능한 유효 결과다.','기존 코드·결과 '+str(len(protected))+'개 및 원본 CSV '+str(len(source))+'개 해시 보존 확인.','근거: config.json, split_audit.json, split_manifest.json.gz, results.json, validation.json, 시드/방식/단계별 예측·학습 입력 ID·번들.']
(O/'FINAL_RESULTS_KO.md').write_text('\n\n'.join(lines),encoding='utf-8')
(O/'finalized.json').write_text(json.dumps(dict(state='complete',validated_records=v['records'],protected_artifacts=len(protected),source_csv=len(source),report='FINAL_RESULTS_KO.md',summary='summary.csv',initial_training_seconds_total=sum(initial_times)),indent=2),encoding='utf-8')
print('Final report and protected integrity audit complete:',O)
