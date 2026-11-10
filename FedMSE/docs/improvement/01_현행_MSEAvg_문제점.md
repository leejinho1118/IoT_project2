# 현행 MSEAvg 문제점

현행: 서버 dev set에서 각 클라이언트 모델의 MSE를 구해 α_i = 1/MSE_i, 가중평균. 샘플 수는 쓰지 않음 (`src/Trainer/global_aggregator.py` `fed_mse_avg`).

## 코드로 확인된 것
| # | 문제 | 근거 | 영향 |
|---|---|---|---|
| 1 | **로컬 학습이 직렬 체인** | `main.py:271-277`이 `ClientTrainer(model=global_aggregator.model)`로 같은 객체를 넘기고, `client_trainer.py:50`의 `.to()`가 같은 객체를 돌려줌. 클라이언트마다 모델을 다시 불러오지 않음 | 클라이언트가 직전 클라이언트 가중치에서 시작 → 뒤쪽 모델의 dev MSE가 유리해져 1/MSE에 순서 편향 가능. FedProx의 `previous_global_model`도 직전 클라이언트 기준. 원 저장소 코드 그대로이므로 재현 결과에도 포함됨 |
| 2 | 샘플 수 무시 | 튜플 `(state, total_samples, n_i)` 중 MSEAvg는 안 씀. 세 번째 값이 클라이언트별 n_i, 두 번째는 참가자 전체 합 | 클라이언트 크기 차이 약 13배(313~4283) |
| 3 | 스케일러 불일치 | 클라이언트는 자기 train에 fit한 scaler(`main.py:119`), 서버 dev는 합친 raw dev에 새로 fit(`global_aggregator.py:44-46`) | 서버가 보는 입력 스케일이 로컬 학습 때와 다름 |
| 4 | dev set이 작고 고정 | 정상만, 클라이언트마다 `min_len`개(`main.py:207-219`), run당 1회 샘플 | 이상 샘플 오염은 없으나 MSE 추정 분산이 큼 |
| 5 | 지표 불일치 | 가중치는 재구성 MSE, 탐지는 잠재 Centroid, 학습손실은 MSE+10·‖z‖, global early stop은 dev shrink loss | 목적이 서로 다름 |
| 6 | 조기종료 변동 | global patience=1 + 50% 랜덤 참가 | 라운드 수 3~16 편차 |

## 아직 미확인 (로깅으로 확인 필요)
- **α가 너무 평평한가, 너무 날카로운가**: codex는 한 클라이언트 쏠림을, Sonnet은 거의 균등 평균일 가능성을 우려. 현재 코드가 `weighted`를 만들고 버려서 값을 모름.
- **직렬 체인이 실제로 순서 편향을 만드는가**: 학습 시작 가중치와 직전 global의 L2 거리, 참가 순서별 MSE를 기록하면 확인 가능.
