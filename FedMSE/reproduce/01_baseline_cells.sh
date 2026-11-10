#!/bin/bash
# 1) 비교 기준 6셀 재현: {직렬, 병렬} x {균등 U, FedAvg A, MSEAvg M}, 주 분할 data_seed=1234, run 0~9, 20라운드.
#    6개를 동시에 돌리면 약 4.4시간(RTX 3060 1장, 프로세스당 CPU 2스레드). 결과: src/Checkpoint/Results/EXP/H12/
#    (이 저장소에는 결과가 이미 들어 있으므로, 분석만 다시 하려면 02_analysis_only.sh 만 실행하면 된다.)
set -e; cd "$(dirname "$0")/../src"
RUNS="0 1 2 3 4 5 6 7 8 9"
for m in uniform avg mse_avg; do
  extra=""; [ "$m" = "mse_avg" ] && extra="--log_cf"       # 직렬 MSEAvg만 반사실 로그(H1 기전 점검용)
  python main_exp.py --mode final --method $m --data_seed 1234 --runs $RUNS --rounds 20 --exp H12 $extra > ../reproduce/log_H12_serial_$m.txt 2>&1 &
  python main_exp.py --mode final --method $m --data_seed 1234 --runs $RUNS --rounds 20 --exp H12 --topology parallel > ../reproduce/log_H12_parallel_$m.txt 2>&1 &
done
wait; echo "기준 6셀 완료"
