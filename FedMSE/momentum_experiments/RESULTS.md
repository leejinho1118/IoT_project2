# 집계 변형 사전 점검 결과

## 설정과 재현 확인

비IID는 `verify_cells.py`의 `n_*`와 같은 `data_seed=1234`, `Configuration/repro-noniid-10clients.json`, run 0–9를 사용했다. 이는 `analysis/PRE_RULES_2.md` §0–1의 주 분할 및 H12 하이브리드 비교(20라운드, 10 clients, 논문 설정)다. `src/main_exp.py`의 100 epoch, lr 1e-5, batch 12와 `src/main_exp3.py`의 patience=3 로컬 조기종료를 적용했다. 참여율 0.5(매 라운드 5명)이며 run 번호×10000 난수 시드다. IID 후속은 `verify_cells.py`의 `i_*`처럼 IIDified 설정 파일을 썼고, 같은 data seed 및 run 0–4를 MSEAvg/FedAvgM에 짝지었다.

기존 H12 `final_mse_avg` 10런은 재사용했다. 새 실행기의 MSEAvg 방식 run 0은 H12 run 0과 참여자 시퀀스가 전 라운드 일치했으나, `global_auc`의 라운드·클라이언트 최대 절대차가 **0.036959**로 재현 허용오차 1e-9를 넘었다. 따라서 요청한 재현 확인은 **실패**다. 이 사실은 변경하지 않은 사전 기준에 따라 별도 기록하며, 아래 결과는 수치상 paired 실행 조건을 맞춘 탐색 결과로 해석해야 한다. MSEAvg와 네 변형 모두 기존 기준선과의 학습 궤적 동등성이 확인된 것은 아니다.

## 비IID 결과

지표는 최종(20) 라운드 클라이언트 평균 AUC다. AUC 차이는 percentage point(%p)로 표시했다. 짝 차이는 run 0–9 변형−H12 MSEAvg이며 95% 양측 t 구간(df=9)이다. 네 변형 간 다중 비교 보정은 하지 않았다.

| 변형 | 런 평균 AUC | MSEAvg 대비 평균 차이 (%p) | 95% t 구간 (%p) | 런 수 | 참여자 일치 |
|---|---:|---:|---:|---:|---|
| A1, p=2 | 96.5836% | +0.1875 | [+0.0280, +0.3469] | 10 | 예 |
| A1, p=4 | 96.4278% | +0.0317 | [−0.1619, +0.2253] | 10 | 예 |
| A2, FedAvgM | 98.3543% | +1.9582 | [+1.2651, +2.6513] | 10 | 예 |
| A3, trimmed mean | 96.4127% | +0.0166 | [−0.2370, +0.2701] | 10 | 예 |
| 기준선, H12 MSEAvg | 96.3961% | — | — | 10 기존 런 | — |

사전 선택 규칙상 평균 이득이 가장 큰 FedAvgM을 선택했다. 평균 이득 +0.5%p 이상(+1.9582%p)이고 95% 구간 하한이 0보다 커 **수치 기준 통과**다. 변형 실행 중 AUC 발산 중단은 없었다.

## IID 후속 확인

선택된 FedAvgM만 IID 설정에서 MSEAvg와 run 0–4로 비교했다. 두 방법의 참여자 시퀀스는 모두 일치했다.

| 방법 | 런 평균 최종 AUC |
|---|---:|
| IID MSEAvg | 98.2057% |
| IID FedAvgM | 99.4454% |

FedAvgM−MSEAvg 평균 차이는 **+1.2398%p**, paired 95% t 구간(df=4)은 **[+0.4023, +2.0773]%p**다. 평균 차이가 −0.3%p보다 크므로 사전 IID 확인 기준도 **통과**다.

## 판정과 한계

사전 수치 기준으로 FedAvgM은 비IID 선택 단계와 IID 후속 확인을 모두 통과했다. 다만 run 0에서 H12 MSEAvg 궤적을 재현하지 못했으므로 이 판정은 원본 기준선 대비 재현 검증이 완료된 확증 결과가 아니다. 비IID 네 변형을 동시에 비교했으며 다중 비교를 보정하지 않았다. 데이터셋·분할·모델 한 종류, 비IID n=10 및 IID n=5이며 마지막 라운드 단일 지표만 분석했다. 관찰된 수치만으로 성능 차이의 원인을 해석하지 않는다.

## 산출물

- `CRITERIA.md`와 사전 고정 해시 `CRITERIA.sha256` (결과 후 검증: 일치)
- `RESULTS.csv`: 4개 비IID 변형의 run×round 평균 AUC 및 참여자 목록
- `SUMMARY.csv`: 비IID 런 평균과 paired t 구간
- `IID_RESULTS.csv`, `IID_SUMMARY.json`: IID 기준선과 선택 변형의 run×round 자료 및 요약
- `REPRO_CHECK.json`: H12 MSEAvg run 0 재현 검사
- `runs/`: 집계 변형의 JSONL 원자료
- `run_variants.py`, `summarize.py`, `summarize_iid.py`, 실행기 두 개의 해시는 `CODE_SHA256.txt` 참조
