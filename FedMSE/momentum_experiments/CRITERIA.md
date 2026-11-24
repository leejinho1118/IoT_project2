# 집계 변형 사전 기준 (2026-11-21)

## 설정 및 기준선

기존 H12 `final_mse_avg` run 0–9를 MSEAvg 기준선으로 재사용한다. 기존 결과는 재실행하지 않는다. 새 복사 실행 코드의 기준 MSEAvg 경로는 run 0 한 개를 수행해 기존 H12 run 0의 참여자 시퀀스와 모든 라운드 `global_auc`가 일치하는지 확인한다(부동소수 비교 허용오차 1e-9). 결과는 `RESULTS.md`에 기록한다.

H12 설정 확인 출처는 `analysis/PRE_RULES_2.md` §0–1, `analysis/verify_cells.py`의 `n_*`/`i_*`, `src/main_exp.py` 상수 및 `src/main_exp3.py` 실행 경로다. 비IID: `Configuration/repro-noniid-10clients.json`, `data_seed=1234`; IID: `Configuration/repro-noniid-10clients-iidified.json`(verify 셀 `i_S`/`i_P1`와 같은 방식), `data_seed=1234`. 각 run은 run 번호×10000 시드, run 0–9를 짝지어 사용한다. 10개 클라이언트 중 라운드마다 5개 참여, 20라운드, hybrid, 로컬 최대 100 epoch, patience=3 조기종료, lr=1e-5, batch=12. 참여자 선택은 기준선과 동일한 Python 난수 경로로 생성하며 각 run/라운드의 `join_clients` 일치를 확인한다.

## 사전 지정 변형

모두 같은 비IID 분할, 학습 절차, 20라운드 및 run별 참여자 시퀀스를 사용한다.

- A1-p2: 다섯 참여 클라이언트의 dev MSE에 따라 가중치 `w_i ∝ MSE_i^-2`로 모델 상태 좌표별 혼합.
- A1-p4: `w_i ∝ MSE_i^-4`로 혼합.
- A2: 서버 모멘텀 FedAvgM, sample-count 가중 FedAvg pseudo-gradient, 서버 학습률 1.0, 모멘텀 0.9. 초기 속도 0이며 매 라운드 `v_t = 0.9 v_(t-1) + (FedAvg_t - global_t)`, `global_(t+1) = global_t + 1.0 v_t`.
- A3: 좌표별 trimmed mean. 참여 클라이언트 5개 중 각 파라미터 좌표에서 최솟값·최댓값을 하나씩 버리고 나머지 세 값의 평균.

## 지표 및 통계

주 지표는 20번째 라운드의 10개 클라이언트 AUC 평균이다. 각 방법의 런 평균 AUC와 기준선 대비 run별 짝 차이(변형−MSEAvg, AUC percentage point)를 보고한다. 짝 차이의 양측 95% t 구간은 표본 표준편차(ddof=1), df=9의 t 임계값으로 계산한다. 변형 4개 간 다중 비교 보정은 하지 않으며 그 사실을 보고한다.

## 판정 및 후속 IID 확인

비IID에서 런 평균 차이가 가장 큰 변형 하나를 선택한다(선택 기준은 평균 이득 하나뿐). 선택된 변형이 MSEAvg 대비 평균 +0.5%p 이상이고 짝 차이 95% 구간 하한이 0보다 크면 통과한다. 그 외는 '접음'이다. 통과한 경우에만 해당 변형 하나를 IID 설정에서 run 0–4로 추가 실행하고, IID MSEAvg 기존 기준선과의 차이가 −0.3%p보다 나쁘지 않은지 확인한다. IID에서 기준선으로 사용할 기존 5개 run이 없으면 동일 IID split/seed의 MSEAvg run 0–4도 추가해 짝 기준선을 만든다. 비교, 설정, 예외는 결과를 보기 전에 본 문서 기준으로 고정한다.

수치만으로 성능 차이의 원인을 설명하지 않는다. 수렴/학습 역학과 데이터 특성 같은 원인 해석은 별도 증거 없이는 하지 않는다.
