#!/bin/bash
# 2) 원인 규명 본 실험(탐색 모드: run 0~2). 6개 셀 동시 실행, 약 1시간. 결과: src/Checkpoint/Results/EXP/CAUSE/
#    선행: 00_prepare_data.py, 그리고 (처음이면) 단위 검증 PASS:  cd src && python ../analysis/unit_checks_3.py
#    η=5 셀은 발산하면 스스로 중단하고 DIVERGED 파일을 남긴다(정상 동작, 결과의 일부).
set -e; cd "$(dirname "$0")/../src"
NI=Configuration/repro-noniid-10clients.json; IID=Configuration/scen2-nba-iot-10clients.json
run(){ cell=$1; shift; python main_exp2.py --data_seed 1234 --runs 0 1 2 --rounds 20 --exp CAUSE --cell $cell "$@" > ../reproduce/log_CAUSE_$cell.txt 2>&1 & }
run n_S_log  --topology serial   --config $NI                       # 비-IID 직렬(로깅)
run n_P1_log --topology parallel --server_eta 1 --config $NI        # 비-IID 병렬 η=1(로깅)
run n_P2     --topology parallel --server_eta 2 --config $NI        # η=2
run n_P5     --topology parallel --server_eta 5 --config $NI        # η=5 (발산 예상)
run i_S      --topology serial   --config $IID                      # IID 직렬
run i_P1     --topology parallel --server_eta 1 --config $IID       # IID 병렬 η=1
wait
python ../analysis/analyze_cause.py --exp CAUSE --out ../analysis/cause_result.json --plots ../analysis/figs
