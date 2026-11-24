"""
원인 검증 실험 실행기 (main_exp2.py 복사·확장, 원본 불변; PRE_RULES_4.md).
추가: --disp_target_cell (집계 결과를 같은 run·라운드의 목표 셀 global_disp 크기로 재스케일),
      --patience (ClientTrainer patience; 기본 3은 기존과 동일), 라운드별 효율 E(재스케일 전후) 로깅.
난수 소비 순서는 main_exp2.py와 동일(재스케일·로깅은 난수를 쓰지 않음).
"""
import os, sys, json, copy, random, argparse, pickle, time, logging
sys.path.insert(0, os.getcwd())
import numpy as np
import torch

import main_exp as M
from Trainer import ClientTrainer
from Trainer.global_aggregator_exp import make_weights, mix_states

logging.getLogger().setLevel(logging.WARNING)


def load_clients_cfg(data_seed, dev, config_path):
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


def state_flat(state, keys):
    return torch.cat([state[k].float().reshape(-1) for k in keys])


def aggregate(start_state, states, n, parallel, eta):
    mean = mix_states(states, make_weights(n, 0, None))
    if parallel and eta != 1.0:
        return {k: start_state[k].float() + eta * (mean[k] - start_state[k].float()) for k in mean}
    return mean


def combine_variant(agg, start_state, states, n, variant, velocity):
    """Return updated state and next FedAvgM velocity (if applicable)."""
    if variant == "mse":
        # Match GlobalAggregator.fed_mse_avg arithmetic and model evaluation path
        # exactly for the required run-0 reproducibility check.
        weighted = []
        for st in states:
            agg.model.load_state_dict(st)
            agg.model.eval()
            with torch.no_grad():
                _, generated, _ = agg.model(torch.Tensor(agg.dev_dataset).to(agg.device))
                mse = torch.nn.MSELoss(reduction="mean")(
                    torch.Tensor(agg.dev_dataset).to(agg.device), generated)
                weighted.append((st, 1 / mse))
        out = {}
        for k in weighted[0][0]:
            out[k] = sum([st[k] * alpha for st, alpha in weighted]) / sum([alpha for _, alpha in weighted])
        return out, velocity
    if variant in ("p2", "p4"):
        mses = np.array([agg.stats(st)["mse"] for st in states], dtype=np.float64)
        p = 1 if variant == "mse" else (2 if variant == "p2" else 4)
        lw = -p * np.log(mses)
        lw -= lw.max()
        w = np.exp(lw); w /= w.sum()
        return mix_states(states, w), velocity
    if variant == "trimmed":
        # Stack per coordinate, sort across the five client tensors, discard extremes.
        out = {}
        for k in states[0]:
            stack = torch.stack([st[k].float() for st in states], dim=0).sort(dim=0).values
            out[k] = stack[1:-1].mean(dim=0)
        return out, velocity
    if variant == "fedavgm":
        w = np.asarray(n, dtype=np.float64); w /= w.sum()
        avg = mix_states(states, w)
        vel = {}
        out = {}
        for k in start_state:
            delta = avg[k].float() - start_state[k].float()
            prev = torch.zeros_like(delta) if velocity is None else velocity[k]
            vel[k] = 0.9 * prev + delta
            out[k] = start_state[k].float() + vel[k]
        return out, vel
    raise ValueError(variant)


def rescale_to_target(start_state, mean_state, target):
    """W_g = start + s*(mean-start), s = target/||mean-start||. s==1이면 mean 그대로(비트 동일).
    반환: (state, s, pre_norm)."""
    keys = list(mean_state.keys())
    delta = state_flat(mean_state, keys) - state_flat(start_state, keys)
    pre = float(delta.norm())
    if pre <= 0.0:
        raise RuntimeError("집계 이동거리 0: 재스케일 불가")
    s = float(target) / pre
    if s == 1.0:
        return mean_state, 1.0, pre
    return {k: start_state[k].float() + s * (mean_state[k] - start_state[k].float()) for k in keys}, s, pre


def read_target(cell_dir, run, rounds):
    fn = f"{cell_dir}/Run_{run}.jsonl"
    if not os.path.exists(fn):
        sys.exit(f"목표 로그 없음: {fn}")
    L = [list(json.loads(l).values())[0] for l in open(fn)]
    if len(L) < rounds:
        sys.exit(f"목표 로그 라운드 부족: {fn} ({len(L)}<{rounds})")
    return [x["diag2"]["global_disp"] for x in L], [x["join_clients"] for x in L]


def check_join(target_join, selected_idx, rnd):
    """목표 로그와 현재 run의 join_clients가 같은 라운드에서 다르면 예외."""
    if list(target_join[rnd]) != list(selected_idx):
        raise RuntimeError(f"join_clients 불일치 round {rnd+1}: target {target_join[rnd]} vs {list(selected_idx)}")


def train_round3(agg, clients, selected_idx, epochs, save_root, parallel, log, patience):
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
        tr = ClientTrainer(model=agg.model, save_dir=sd, epoch=epochs, lr_rate=M.lr_rate, update_type="mse_avg",
                           patience=patience)
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
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--disp_target_cell", default=None,
                    help="목표 셀 디렉터리(예: Checkpoint/Results/EXP/VERIFY/n_S_log); 집계 결과를 그 global_disp로 재스케일")
    ap.add_argument("--no_log", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--variant", choices=["mse", "p2", "p4", "fedavgm", "trimmed"], required=True)
    args = ap.parse_args()
    if args.topology == "serial" and args.server_eta != 1.0:
        sys.exit("η는 병렬에만 적용")
    if args.disp_target_cell and args.server_eta != 1.0:
        sys.exit("재스케일과 η는 함께 쓰지 않음")
    if args.disp_target_cell and args.no_log:
        sys.exit("재스케일 셀은 로깅 필수")
    torch.set_num_threads(args.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallel = args.topology == "parallel"
    log = not args.no_log

    clients = load_clients_cfg(args.data_seed, dev, args.config)
    names = [c["device"] for c in clients]
    out_dir = f"../agg_variants_2026-11-21/runs/{args.variant}/{args.cell}"
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(f"{out_dir}/DIVERGED"):
        print(f"{args.cell}: 이미 발산 표시됨, 중단"); return

    for run in args.runs:
        fn = f"{out_dir}/Run_{run}.jsonl"
        if os.path.exists(fn) and sum(1 for _ in open(fn)) == args.rounds:
            print(f"skip {fn} (complete)"); continue
        tgt_disp = tgt_join = None
        if args.disp_target_cell:
            tgt_disp, tgt_join = read_target(args.disp_target_cell, run, args.rounds)
        t0 = time.time()
        M.set_seeds(run * 10000)
        agg = M.build_agg(clients, dev)
        agg.update_type = "mse_avg"
        save_root = f"../agg_variants_2026-11-21/scratch/{args.variant}/{args.cell}/{run}"
        velocity = None
        open(fn, "w").close()
        for rnd in range(args.rounds):
            selected_idx = random.sample(list(range(len(clients))), int(M.num_participants * len(clients)))
            if tgt_join is not None:
                check_join(tgt_join, selected_idx, rnd)
            local, start_state, start, d = train_round3(agg, clients, selected_idx, args.epoch, save_root, parallel, log,
                                                        args.patience)
            n = [m[2] for m in local]; states = [m[0] for m in local]
            new_state, velocity = combine_variant(agg, start_state, states, n, args.variant, velocity)
            scale = pre_norm = None
            if tgt_disp is not None:
                new_state, scale, pre_norm = rescale_to_target(start_state, new_state, tgt_disp[rnd])
            agg.model.load_state_dict(new_state)
            M.set_val_loss(agg)
            rec = {"round": rnd + 1, "join_clients": selected_idx, "devices": names, "n": n, "eta": args.server_eta,
                   "global_loss": agg.val_loss, "global_auc": M.fast_auc_all(agg.model, clients)}
            if log:
                d["global_disp"] = float((M.flat(agg.model) - start).norm())
                ml = float(np.mean(d["local_disp"]))
                d["E"] = d["global_disp"] / ml
                if scale is not None:
                    d["E_pre"] = pre_norm / ml
                    d["scale"] = scale; d["target_disp"] = float(tgt_disp[rnd]); d["pre_disp"] = pre_norm
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
