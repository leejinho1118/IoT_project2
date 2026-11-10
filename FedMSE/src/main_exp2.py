"""
원인 규명 실험 실행기 (main_exp.py 기반, 원본 불변; PRE_RULES_3.md).
균등평균 hybrid 전용. 추가: --server_eta(병렬 서버 스텝), --config(IID 선택), 로깅(에폭/스텝/이동거리/쌍별거리),
발산 시 해당 셀 중단. 난수 소비 순서는 main_exp.py와 동일(로깅은 난수를 쓰지 않음).
"""
import os, sys, json, copy, random, argparse, pickle, time, logging
import numpy as np
import torch

import main_exp as M
from Trainer import ClientTrainer
from Trainer.global_aggregator_exp import make_weights, mix_states

logging.getLogger().setLevel(logging.WARNING)


def load_clients_cfg(data_seed, dev, config_path):
    """main_exp.load_clients와 동일하되 config 경로만 인자로 받음."""
    old = M.config_path
    M.config_path = config_path
    try:
        return M.load_clients(data_seed, dev)
    finally:
        M.config_path = old


def pair_mean(vecs):
    ds = [float((vecs[i] - vecs[j]).norm()) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    return float(np.mean(ds)) if ds else 0.0


def n_epochs_done(save_dir):
    fn = os.path.join(save_dir, "training_tracking.pkl")
    return len(pickle.load(open(fn, "rb"))) if os.path.exists(fn) else 0


def aggregate(start_state, states, n, parallel, eta):
    mean = mix_states(states, make_weights(n, 0, None))
    if parallel and eta != 1.0:     # η=1은 mean 그대로(부동소수 재조합 안 함)
        return {k: start_state[k].float() + eta * (mean[k] - start_state[k].float()) for k in mean}
    return mean


def train_round2(agg, clients, selected_idx, epochs, save_root, parallel, log):
    total = sum(len(clients[i]['train_loader'].dataset) for i in selected_idx)
    start_state = copy.deepcopy(agg.model.state_dict())
    start = M.flat(agg.model)
    local, d = [], {"epochs": [], "steps": [], "local_disp": []}
    vecs = []
    for i in selected_idx:
        c = clients[i]
        if parallel:
            agg.model.load_state_dict(start_state)
        sd = os.path.join(save_root, c['device'])
        tr = ClientTrainer(model=agg.model, save_dir=sd, epoch=epochs, lr_rate=M.lr_rate, update_type="mse_avg")
        tr.run(c["train_loader"], c["valid_loader"])
        local.append((copy.deepcopy(tr.model.state_dict()), total, len(c["train_loader"].dataset)))
        if log:
            ep = n_epochs_done(sd)
            d["epochs"].append(ep); d["steps"].append(ep * len(c["train_loader"]))
            v = M.flat(agg.model); vecs.append(v)
            d["local_disp"].append(float((v - start).norm()))
    if log:
        d["pair_mean"] = pair_mean(vecs)
    return local, start_state, start, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_seed", type=int, required=True)
    ap.add_argument("--runs", type=int, nargs="+", required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--epoch", type=int, default=M.epoch_default)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--topology", default="serial", choices=["serial", "parallel"])
    ap.add_argument("--server_eta", type=float, default=1.0)
    ap.add_argument("--config", default="Configuration/repro-noniid-10clients.json")
    ap.add_argument("--no_log", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if args.topology == "serial" and args.server_eta != 1.0:
        sys.exit("η는 병렬에만 적용")
    torch.set_num_threads(args.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallel = args.topology == "parallel"
    log = not args.no_log

    clients = load_clients_cfg(args.data_seed, dev, args.config)
    names = [c["device"] for c in clients]
    out_dir = f"Checkpoint/Results/EXP/{args.exp}/{args.cell}"
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(f"{out_dir}/DIVERGED"):
        print(f"{args.cell}: 이미 발산 표시됨, 중단"); return

    for run in args.runs:
        fn = f"{out_dir}/Run_{run}.jsonl"
        if os.path.exists(fn) and sum(1 for _ in open(fn)) == args.rounds:
            print(f"skip {fn} (complete)"); continue
        t0 = time.time()
        M.set_seeds(run * 10000)
        agg = M.build_agg(clients, dev)
        agg.update_type = "mse_avg"
        save_root = f"Checkpoint/EXP_scratch/{args.exp}/{args.cell}/{run}"
        open(fn, "w").close()
        for rnd in range(args.rounds):
            selected_idx = random.sample(list(range(len(clients))), int(M.num_participants * len(clients)))
            local, start_state, start, d = train_round2(agg, clients, selected_idx, args.epoch, save_root, parallel, log)
            n = [m[2] for m in local]; states = [m[0] for m in local]
            agg.model.load_state_dict(aggregate(start_state, states, n, parallel, args.server_eta))
            M.set_val_loss(agg)
            rec = {"round": rnd + 1, "join_clients": selected_idx, "devices": names, "n": n, "eta": args.server_eta,
                   "global_loss": agg.val_loss, "global_auc": M.fast_auc_all(agg.model, clients)}
            if log:
                d["global_disp"] = float((M.flat(agg.model) - start).norm())
                rec["diag2"] = d
            div = rnd + 1 >= 5 and float(np.mean(rec["global_auc"])) < 0.90
            if div: rec["diverged"] = True
            with open(fn, "a") as f:
                f.write(json.dumps({f"round_{rnd+1}": rec}) + "\n")
            print(f"[{args.cell}] run {run} round {rnd+1}/{args.rounds} meanAUC={np.mean(rec['global_auc']):.4f} "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)
            if div:
                open(f"{out_dir}/DIVERGED", "w").write(f"run {run} round {rnd+1}\n")
                print(f"[{args.cell}] 발산: run {run} round {rnd+1}, 셀 중단", flush=True)
                return


if __name__ == "__main__":
    main()
