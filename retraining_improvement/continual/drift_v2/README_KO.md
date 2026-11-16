# FedMSE 지속학습 개선 V2

20222038 이진호 / 2026-11-13

기존 src, continual/experiment.py와 full_v1 결과는 보존한다. 이 폴더는 분포 변화·보관량·검증 불확실성을 비교하는 별도 확장이다.

## 새로 구현한 것

- 개발 전용 정상 자료의 signed-log1p 표준화와 PC1 중앙값으로 A/B 정상 패턴을 구분한다. 보류 평가는 매퍼 적합 전에 격리한다.
- 초기 SAE는 A만 학습하고 이후 B 정상·공격 후보가 유입된다. 정상 검증자 보조 환경이며 공격과 불확실 후보는 학습에서 제외한다.
- 무갱신, 신규 정상만, 기존 정상 32개·128개 리플레이를 비교한다. 후속으로 동일한 세 갱신 방식을 후보20% 보류 조건에서도 비교한다.
- 원본 SAE·MSEAvg·공유 모델 순차 학습·두 if 조기 종료를 유지한다. 새 분포의 입력 폭주로 NaN이 발생한 개발 실패 기록은 cpu_smoke_v1에 보존한다. 모든 방식에 동일한 초기 A StandardScaler 및 고정 [-20,20] 포화를 적용했다.
- 초기 A 검증 자료로 재보정한 FPR/TPR 외에 초기 수치 임계값을 고정한 지표와 정상 복원 MSE도 기록한다. 임계값 재보정이 망각을 가리는지 구분해 읽는다.
- CEN은 해당 방식 정상 학습 기준으로 다시 적합하므로 이 비교에는 표현 학습과 CEN 기준 변화 효과가 함께 포함된다. 순수 가중치 망각 효과만 분리했다고 주장하지 않는다.

## 파일 구조

- scenario.py: A/B 매퍼·행 ID/특징 해시 분할·정상 검증·보관량별 혼합·고정 범위 입력 scaler
- backend.py: V1 학습 함수 10개를 그대로 복사, 원본 src 모듈 import. backend_provenance.json에 출처 기록
- run.py: 실험 잠금, CPU/GPU 실행, 공통 초기 모델, 7방식 비교, 비용·복원 오차·초기 임계값 지표
- infer.py: 저장된 V2 번들로 정답 없이 CPU 추론
- verify.py: 분할, 실제 A/B 배정, 검증 라우팅, 학습 ID, 예측·AUC, 예산·번들 독립 검증
- test_protocol.py: 분포 변환·포화·보류·공격 제외·고정 예산·재현성 테스트 7개
- PROTOCOL_FROZEN.json: 성능 결과와 무관하게 고정한 전체 비교 계획
- integrity_before.json: 기존 코드와 결과의 보존 검사용 해시

## 실행

프로젝트 루트 C:/Users/dlwls/LocalProjects/fedmse에서 실행한다. 기존 run-id는 덮어쓰지 않는다.

```powershell
$py = '.venv/Scripts/python.exe'
& $py -m unittest discover -s continual/drift_v2 -p test_protocol.py -v
& $py continual/drift_v2/run.py --profile smoke --device cpu --run-id cpu_smoke_NEW
& $py continual/drift_v2/verify.py --run-id cpu_smoke_NEW
```

full 프로필은 50클라이언트·5시드·3단계·7방식의105개 평가다. pilot은10클라이언트·2시드·최대10에포크의 개발 검증이며 최종 성능이 아니다. 모델 초기 학습 비용은 시드별 한 번만 집계한다. 동일 데이터셋을 다시 분할한 모의 분포 변화이므로 실제 시간 드리프트·미지 공격·실제 검증자·외부 일반화는 검증하지 않았다. 결과가 좋아진다는 가정이나 리플레이 우월성 주장은 하지 않는다.
