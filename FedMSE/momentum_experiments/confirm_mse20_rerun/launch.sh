#!/usr/bin/env bash
cd ~/fedmse/src
for r in 0 1 2 3 4 5 6 7 8 9; do
  nice -n 10 python3 /home/sdh2983/fedmse/agg_variants_2026-11-21/confirm_mse20_rerun/run_confirm.py --data_seed 1234 --runs $r --rounds 20 --exp CONFIRM20261121 --cell noniid --config Configuration/repro-noniid-10clients.json --patience 3 --threads 1 --variant mse > /home/sdh2983/fedmse/agg_variants_2026-11-21/confirm_mse20_rerun/logs/noniid_mse_20_run$r.log 2>&1 &
done
wait
echo MSE20_RERUN_DONE
