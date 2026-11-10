# 재현 가이드: 직렬 > 병렬 격차 원인 규명 (FedMSE 후속 실험)

원 논문 코드(`src/main.py` 등)는 그대로 두고, 후속 실험용 파일만 추가했다.

## 폴더 구조
```
FedMSE/
├─ REPRODUCE.md                 ← 이 문서
├─ Data/                        원본 분할 압축(Prepared_dataset.z01/.zip). 풀면 Data/N-BaIoT/ (git 제외)
├─ reproduce/                   재현 스크립트 (아래 순서)
│   ├─ 00_prepare_data.py       분할 zip 풀기
│   ├─ 01_baseline_cells.sh     비교 기준 6셀 (직렬/병렬 × U/A/M), 약 4.4시간
│   ├─ 02_analysis_only.sh      저장된 결과로 분석만 재실행 + 저장 JSON과 일치 확인 (GPU 불필요)
│   └─ 03_cause_experiment.sh   원인 규명 본 실험 6셀, 약 1시간
├─ analysis/                    사전 규칙(동결), 분석 코드, 단위 검증, 결과 JSON, 그림
│   ├─ PRE_RULES_2.md / PRE_RULES_3.md / PRE_RULES_3_ADD1.md  (+ .sha256: 실행 전 동결 해시)
│   ├─ analyze_h1h2.py, analyze_cause0.py, analyze_cause.py
│   ├─ unit_checks.py, unit_checks_2.py, unit_checks_3.py
│   ├─ h1h2_result.json, cause0_result.json, cause_result.json, figs/
│   └─ sol_stage*_prompt.txt    독립 검증(codex)에 쓴 프롬프트
├─ docs/improvement/            분석 문서 00~05 (결과는 05_검증결과.md 의 "원인 규명" 절), figures/
└─ src/
    ├─ main_repro.py            원본 직렬 코드의 non-IID 10 재현용 사본
    ├─ main_exp.py              기준 6셀 실행기 (--topology serial|parallel, --method uniform|avg|mse_avg)
    ├─ main_exp2.py             원인 규명 실행기 (--server_eta, --config, 에폭·이동거리·드리프트 로깅)
    ├─ Trainer/global_aggregator_exp.py
    ├─ Configuration/repro-noniid-10clients.json
    └─ Checkpoint/Results/EXP/{H12,S1,CAUSE}/   라운드별 원자료(JSONL, 합쳐서 약 1.3MB)
```
모든 분석 스크립트는 `src/`에서 실행하는 것을 전제한다(`cd src && python ../analysis/...`).

## 환경
Python 3.13, torch 2.14(CUDA), scikit-learn 1.9, scipy 1.18, numpy 2.4, pandas 3.0. 실험은 RTX 3060 1장, 프로세스당 CPU 2스레드. `pip install -r requirements.txt`(버전 고정 없음)로 설치하면 되지만 버전이 다르면 수치가 미세하게 달라질 수 있다.

## 가장 빠른 확인 (결과 재현, 수 분)
```bash
bash reproduce/02_analysis_only.sh
```
동결 해시 확인 → 저장된 원자료로 분석 3종 재실행 → 저장된 JSON과 1e-9 허용오차로 비교(`일치`가 3줄 나오면 성공).

## 처음부터 재실험
```bash
python reproduce/00_prepare_data.py            # Data/N-BaIoT 생성 (CRC 검증 포함)
bash reproduce/01_baseline_cells.sh            # H12 기준 6셀 (선택: 결과가 이미 저장소에 있음)
cd src && python ../analysis/unit_checks_3.py  # 단위 검증: ALL PASS 여야 다음 단계로 (약 5분)
cd .. && bash reproduce/03_cause_experiment.sh # 원인 규명 6셀 + 분석 + 그림
```
- 실행은 시드(`run×10000`, 데이터 분할 `data_seed=1234`)로 결정적이다. 같은 환경에서 η=1 병렬/직렬 재실행은 기존 셀과 AUC 차 0으로 일치했다.
- η=5 셀은 발산하면 `DIVERGED` 파일을 남기고 스스로 멈춘다. 오류가 아니라 결과다.
- 05 문서를 쓴 뒤에 `analysis/*.py`를 고치지 말 것: 사전 규칙(`PRE_RULES_*.md`)은 sha256으로 동결되어 있다.

## 결과 요약 (탐색적, run 0~2, 확정 아님)
자세한 숫자·한계는 `docs/improvement/05_검증결과.md`의 "원인 규명" 절.
- 병렬은 직렬보다 M1 −1.1%p, M2 −4~5%p (기존 10 run 결과).
- 서버 스텝 η=2로 키우면 병렬이 직렬 수준을 넘었다(M1 97.6 대 96.0). η=5는 7라운드에서 발산.
- IID 데이터에서는 격차가 비-IID의 약 10~17%로 줄었다. 로컬 early stop 차이는 작았다(에폭 1.10배).
- 원인 확정은 아님: 단일 분할, 3 run, 균등평균 한정. 확정하려면 run 확대·early stop 끄기 대조·확인 분할(data_seed=4321) 재현이 필요하다.
