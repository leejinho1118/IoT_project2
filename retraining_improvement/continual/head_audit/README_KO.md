# FedMSE A안 재검토와 후속 개선
20222038 이진호

A는 기존 개발 점수에서 선택된 안이며 절대 최적은 아닙니다. 신규 정상 분포 오탐, 잠재 표현 병목, 검증자 오류를 분리해 조사했습니다. 원본 src 및 기존 실험은 수정하지 않았습니다.

## 실제 결과 (50클라이언트, 5시드, 마지막 단계 평균)
|구성|신규 AUC|오탐률|탐지율|
|---|---|---|---|
|동일 A 가중치 원래 CEN|0.60752|58.28%|88.12%|
|입력+SAE 정상 기억, 현재 단계|0.98061|6.94%|89.69%|
|누적 우선순위 정상 기억, 사후 탐색|0.98386|7.67%|93.73%|

마지막 정책은 탐지율과 오탐률 절충입니다. 원시 입력 kNN 기준도 거의 같은 성능이라 SAE의 추가 효용은 입증되지 않았습니다. SAE/MSEAvg 가중치는 동결하고 정상 자료로 탐지 헤드만 적합했습니다. 신경망 재훈련 결과가 아닙니다.

## 구조와 실행 순서
```
보존된 A 단계별 SAE + 초기 scaler
  -> head_models.py        중심/거리/정상 보정 비교
  -> residual_models.py    입력115 + SAE11 블록 결합, 정상 kNN
  -> stream_memory.py      최대128개 과거/현재 정상 우선순위 기억
  -> run*.py               개발 평가 -> 설정 동결 -> 탐색적 보류 평가
  -> verify*.py            독립 AUC/혼동행렬/정상 참조/분할/임계값 검사
  -> stress_verifier.py    잘못 승인된 공격의 기억 오염
  -> stress_guard.py       신뢰 정상 앵커 기반 부분 방어
  -> infer_residual.py     정답/도메인 없는 CPU 추론
```
run.py=9가지 원인 분해, run_residual.py=입력 우회 후보, run_trajectory.py=1·2단계 확인, run_priority.py=과거 정상 누적 기억 비교입니다. 기존 경로 재실행/덮어쓰기를 거부합니다. main.py를 고치거나 원래 결과와 섞지 않습니다.

## 단독 추론
프로젝트 루트에서 실행합니다. 헤더 없는 원본과 같은 순서의 유한 숫자115열 CSV가 필요합니다. 정답 열은 넣지 않습니다. 클라이언트 번호는 그 모델에 존재해야 합니다.
```
.venv/Scripts/python.exe continual/head_audit/infer_residual.py --bundle continual/head_audit/output/residual_full_v1/models/seed_0/residual_knn_w025.pt --client 1 --input INPUT.csv --out NEW_predictions.csv
```
누적 기억 후보는 bundle 경로를 output/priority_stage3_v1/models/seed_0/residual_knn_w025.pt로 바꿉니다. 두 모델 모두 검증자 정상 판정을 전제로 하며 실서비스 추천 확정 모델은 아닙니다. 신뢰할 수 있는 자기 모델 번들만 로드합니다.

## 한계와 실패도 근거로 보존
- A/B는 모의 특징 분할이며 실제 시간·기기·미지 공격 검증이 아닙니다.
- 이전에 본 보류 풀을 재사용했으므로 탐색적 결과입니다. 설정을 고정한 새 자료 검증은 미완료입니다.
- 후보 적합/보정에는 정상만 쓰지만 후보 선택에는 개발 공격 라벨을 사용했습니다.
- 검증자 공격 오승인 약4.79%에서 탐지율89.30%->19.32%. 구현한 신뢰 앵커 가드는65.02%로 부분 회복했습니다. 해결된 문제가 아닙니다.
- 누적 기억 정책은 사후 추가 비교입니다. 오염 스트레스는 현재 단계 기억 후보에서 수행했으며 누적 정책에 그대로 보장되지 않습니다.
- 논문 제곱 L2와 공개 코드 norm 손실 불일치 직접 검사 완료. 손실 교정 재훈련은 아직 하지 않았습니다.
- head_full_v1은 미세 수치 경계의 엄격 판정 일치 검사 실패 기록입니다. 숨기거나 성공으로 합산하지 않습니다.
- 원시 정상 특징 보관으로 개인정보 저장 비용이 생깁니다. DP/안전 집계를 구현하지 않았습니다.

## 결과 위치
output/head_full_v2, residual_full_v1, trajectory_stage1_v1, trajectory_stage2_v1, priority_stage1_v1~3_v1의 validation.json 참조. 새 완료 평가360개, 독립 클라이언트 AUC36,000개 검사를 통과했습니다. 별도 스트레스 결과와13개 단위 검사도 보존했습니다. 학습/평가 샘플 수나 독립 데이터셋 수라는 뜻은 아닙니다.

evidence_summary.json, memory_policy_comparison.json, risk_audit.json, sources.json, comparison.csv와9쪽 최종 PDF를 함께 보세요. 실행 환경은 output/residual_full_v1/environment.json입니다.

연구 묶음 ZIP에는 코드와 요약 근거를 넣었으며 원시 CSV·대형 예측·모델은 제외했습니다. 모델은 원래 로컬 경로에 보존돼 있습니다. 동일 프로젝트의 continual/protocol.py, drift_v2, research_ABCDE 데이터 도우미가 필요합니다.
