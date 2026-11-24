# CRITERIA_dataset.md 변경 기록
## 2026-11-23 — CIC 시드 2024·IID 셀 취소 (사용자 지시, 마스터 세션 경유)
- 변경: CIC KMeans data_seed=1234는 이미 시작된 런만 끝내고, 시작 전이던 CIC KMeans data_seed=2024(run 10–19×2방법=20런)와 CIC IID 참고 셀(1234, run 10–14×2방법=10런)은 실행하지 않음. ToN-IoT(두 data_seed, 40런)는 완료.
- 방법: 런처(launch_dataset.py)를 종료, 이미 실행 중이던 프로세스는 그대로 완료.
- 사유: CIC 절반 진행에서 방향이 보이므로 빠르게 다음 단계로 이동, CPU를 서버 옵티마이저 1단계에 배정.
- 영향: CIC는 data_seed 1개만 있으므로 CRITERIA_dataset.md의 "데이터셋 판정(두 data_seed 모두 통과)" 규칙을 충족할 수 없다. CIC 결과는 data_seed 1234 셀만의 결과(통과/불통과)로 보고하며 데이터셋 판정은 "미완(시드 1개)"로 쓴다. 미완 시 시작된 런(run 10–19 중 일부)이 모두 40라운드를 끝낸 짝만 사용. 원문 기준은 수정하지 않음.
