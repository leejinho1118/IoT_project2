# FedMSE 서버 모멘텀(FedAvgM) 실험
MSEAvg에 서버 모멘텀(β=0.9, η=1)을 얹은 집계 실험의 재현 코드·사전 기준(CRITERIA, sha256 동결)·결과 문서·그림.
- 코드: `run_variants.py`, `confirm/run_confirm.py`, `confirm/launch_*.py`, `confirm/summarize_*.py`
- 결과: `RESULTS.md`, `confirm/RESULTS_*.md`, `runs/`(요약 JSON), `figures/`, `momentum_results_2026-11-22.pptx`
- 요약: N-BaIoT(비IID)에서 R31~40 AUC 96.05→98.65(+2.60%p); ToN-IoT·CIC-IoT에서는 불리(각 −7.5/−6.0, −2.4%p). 데이터셋 의존적.
- 제외: 원시 로그(수 GB), data_cache, scratch, 문헌 PDF
