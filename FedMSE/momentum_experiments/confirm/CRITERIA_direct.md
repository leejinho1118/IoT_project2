# MSEAvg 위 모멘텀 직접 대조 사전 기준 (2026-11-22)
배경: 기존 +1.96%p(FedAvgM R20 98.35 vs MSEAvg 96.40)는 표본수 가중 FedAvgM과 역MSE 가중 MSEAvg의 비교라 가중과 모멘텀이 같이 바뀌었다(run_confirm.py combine_variant: mse=1/mse 가중, fedavgm=n 가중). 이 실험은 가중을 동일하게(역MSE) 두고 모멘텀만 바꾼다.

## 셀 (각 10런, 40라운드)
- D0: MSEAvg(역MSE 가중, 모멘텀 없음) — variant `mse`
- D1: MSEAvg+모멘텀 — variant `mse_m`: 역MSE 가중 평균 W̄_t로 Δ_t=W̄_t−W_t, v_t=0.9·v_{t−1}+Δ_t, W_{t+1}=W_t+1.0·v_t
고정: topology=parallel(모든 참가자가 같은 글로벌 모델에서 시작), lr=1e-5, patience=3, batch=12, 최대 100 epoch, 비IID config, data_seed=1234, run 10–19(시드 run×10000), 라운드당 5/10 참여, 두 셀의 참여 순서 동일(사후 검증). 기존 serial 결과와는 비교하지 않는다.

## 주 지표·판정
주 지표: 런별 R31~R40 클라이언트 평균 AUC의 평균(%). 짝(같은 run) 차 D1−D0, 양측 95% t 구간(df=n−1). 하한 > +0.3%p면 통과, 아니면 불통과(FedAdam·Yogi 탐색 보류). 발산(R5 이후 평균 AUC<0.90)은 제외하지 않고 포함하며 개수를 별도 집계.
보조(판정 아님): R20, R40 AUC, 목표 AUC=0.97 도달 라운드(이후 모든 라운드가 0.97 이상 유지되는 첫 라운드; 미도달 런 수 별도), 로컬 학습 스텝 총합.
결과 후 설정·기준 변경 금지, 필요 시 DEVIATIONS_direct.md에 기록.

## 한계
같은 분할(data_seed=1234)만 사용하므로 run 10–19는 시드(초기화·참여 순서)만 새롭고 데이터 분할은 기존과 같다. 새 분할 확인은 2단계에서 한다. N-BaIoT 단일 데이터셋.
