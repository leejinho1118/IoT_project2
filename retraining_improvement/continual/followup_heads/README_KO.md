# FedMSE 후속 탐색: 거리·밀도·저오탐 비교
20222038 이진호

별도 continual/followup_heads에 구현했습니다. 원본 src 및 과거 결과를 수정하지 않았습니다. 모든 대형 결과는 LocalProjects에 보존합니다.

## 이번 결과
신규 입력 전용 Chebyshev kNN: AUC0.98485 / 오탐6.60% / 탐지92.49%. 기존 입력+SAE 현재 정상 기억 후보의0.98061 / 6.94% / 89.69%보다 좋았습니다. SAE를 사용하지 않아 FedMSE 신경망 개선 효과라고 해석하면 안 됩니다.

8개 새 방식과4개 공통 대조군을 두 묶음으로 비교했습니다. 입력 전용 방법의 시드별 결과가 동일한 이유는 분할/참조/전처리가 같고 신경망 난수를 사용하지 않기 때문입니다. 독립5회 성공이나 외부 자료 일반화가 아닙니다. 기존 누적 기억 방법과 새 거리의 결합은 실행하지 않았습니다.

## 코드 구조
- models.py: whitening kNN, 밀도 정규화 kNN, IsolationForest, 두 정상 공분산 거리.
- metric_models.py: Manhattan, cosine, Chebyshev, 입력+SAE Manhattan.
- run.py / run_metrics.py: 공통 A 인코더·자료를 보존한 개발 선택/동결/보류 평가.
- verify.py / verify_metrics.py: 원본해시, 정상 참조, 분할, AUC/혼동행렬, 정상 임계값을 독립 재계산.
- operating_points.py / operating_metrics.py: 정상 보정 q95/q97.5/q99/q99.5/q100 비교. 오탐1% 목표 통과 구성 없음.
- stress_metric.py: 정상 기억에 후보 공격을 잘못 승인한 상황. 평가 공격은 별도 ID.
- infer.py: 115열, 정답/도메인 없는 CPU 추론. 입력 전용 방법에서는 SAE를 실행하지 않음.
- test*.py:8개 단위 검사.

run 경로는 기존 결과가 있으면 거부합니다. 재실행 시 새 run-id를 사용해야 합니다. 기존 프로젝트의 head_audit, drift_v2, research_ABCDE와 원본 데이터가 필요합니다. 이 ZIP은 해당 프로젝트에 추가할 코드·근거 묶음이며 원본 데이터/학습 번들을 담지 않았습니다.

## 실행 예
프로젝트 루트에서:
```
.venv/Scripts/python.exe continual/followup_heads/run_metrics.py --profile full --device cuda --run-id NEW_RUN
.venv/Scripts/python.exe continual/followup_heads/verify_metrics.py --run-id NEW_RUN
.venv/Scripts/python.exe continual/followup_heads/infer.py --bundle continual/followup_heads/output/metrics_v1/models/seed_0/input_chebyshev.pt --client 1 --input INPUT.csv --out NEW_predictions.csv
.venv/Scripts/python.exe -c "import torch,unittest; s=unittest.defaultTestLoader.discover('continual/followup_heads',pattern='test*.py'); r=unittest.TextTestRunner().run(s); raise SystemExit(not r.wasSuccessful())"
```
입력은 헤더 없이 원래와 동일한 특징 순서의 유한 숫자115열 CSV입니다. 모델이 가진 클라이언트 ID를 지정해야 합니다. 자신의 신뢰할 수 있는 번들만 로드합니다. 기존 출력은 덮어쓰지 않습니다.

Windows 단위 검사에서 torch 이전에 sklearn ensemble을 단독 로딩하자 DLL 로딩 오류가 한 번 발생했습니다. 실제 runner는 torch를 먼저 로드해 정상 완료했고 위 동일 순서의 단위 검사8개도 통과했습니다. 보안 설정이나 패키지는 변경하지 않았습니다.

## 증거와 남은 위험
- 본 비교160개 평가 기록 / 독립 클라이언트 AUC16,000개 검사. 저오탐700개 기록 / 혼동행렬1,400개 재검산. 숫자는 반복 평가 수이며 독립 데이터셋 수가 아닙니다.
- 정상 보정 자료는 클라이언트·영역별26~39개. 1% 꼬리 확률을 보장할 충분한 근거가 아니며 실제 교환가능성도 미확인입니다.
- Chebyshev 오염 스트레스: 공격 약4.79% 혼입 시 개발 TPR92.15%->17.75%. 주요 번들은 변경하지 않았습니다.
- 현재 모의 A/B 및 이전에 본 보류 자료 재사용. 실제 기기·시간 드리프트/미지 공격 검증이 아닙니다.
- 초기 scaler·정상 참조·보정 역할은 기존 연구 프로토콜을 따릅니다. 공개 원본 코드와 논문 손실의 차이를 교정한 재훈련은 이번 범위에 없습니다.
- 원시 정상 특징의 보관에 DP나 안전 집계를 적용한 결과는 아닙니다.

4쪽 PDF에 실패한 방향까지 기록했습니다. 결과는 output/full_v1, metrics_v1, operating_v1, operating_metrics_v1, chebyshev_stress_v1에 분리돼 있습니다. completion.json과 각 validation.json을 확인할 수 있습니다.
