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


GUARD = {}
MSE_Q = []  # mse 분기가 마지막으로 쓴 정규화 가중치(로그·mse_fx 전용, 수치 경로 불변)
FX = {}


def fedexp_eta(dnorm2, q, dbar_norm2, eps):
    """FedExP 가중 일반화: max(1, sum_i q_i*||D_i||^2 / (2(||Dbar||^2+eps))). 반환 (eta, sum_q_dnorm2)."""
    sq = float(sum(float(a) * float(b) for a, b in zip(q, dnorm2)))
    return max(1.0, sq / (2.0 * (float(dbar_norm2) + eps))), sq


def combine_variant(agg, start_state, states, n, variant, velocity, server_eta=1.0, momentum=0.9, opt=(0.9, 0.99, 1e-3), fx_eps=1e-3):
    """Return updated state and next FedAvgM velocity (if applicable)."""
    if variant == "mse_fx":
        avg, _ = combine_variant(agg, start_state, states, n, "mse", None)
        q = [float(a) for a in MSE_Q]
        keys = list(start_state.keys())
        s0 = state_flat(start_state, keys)
        dn2 = [float(((state_flat(st, keys) - s0) ** 2).sum()) for st in states]
        dbar = state_flat(avg, keys) - s0
        db2 = float((dbar ** 2).sum())
        eta, sq = fedexp_eta(dn2, q, db2, fx_eps)
        FX.clear()
        FX.update(eta_g=eta, sum_q_dnorm2=sq, dbar_norm2=db2,
                  dnorm_mean=float(np.mean([x ** 0.5 for x in dn2])), dnorm_max=float(max(dn2) ** 0.5))
        if eta == 1.0:
            return avg, velocity
        return {k: start_state[k].float() + eta * (avg[k].float() - start_state[k].float()) for k in keys}, velocity
    if variant in ("mse_adam", "mse_yogi"):
        # Reddi et al. 2021 (FedAdam/FedYogi): Delta = MSEAvg 집계 - 현재 글로벌, 편향보정 없음, v_{-1}=tau^2
        avg, _ = combine_variant(agg, start_state, states, n, "mse", None)
        b1, b2, tau = opt
        mm, vv, out = {}, {}, {}
        for k in start_state:
            delta = avg[k].float() - start_state[k].float()
            m0 = torch.zeros_like(delta) if velocity is None else velocity["m"][k]
            v0 = torch.full_like(delta, tau * tau) if velocity is None else velocity["v"][k]
            mm[k] = b1 * m0 + (1 - b1) * delta
            d2 = delta * delta
            vv[k] = (b2 * v0 + (1 - b2) * d2) if variant == "mse_adam" else (v0 - (1 - b2) * d2 * torch.sign(v0 - d2))
            out[k] = start_state[k].float() + server_eta * mm[k] / (vv[k].sqrt() + tau)
        return out, {"m": mm, "v": vv}
    if variant == "mse_mg":
        # 탐색용(DEVIATIONS_optim.md): 검증 MSE 가드. M_now>M_prev이면 v 리셋하고 MSEAvg(W̄)를 사용
        avg, _ = combine_variant(agg, start_state, states, n, "mse", None)
        vel, out = {}, {}
        for k in start_state:
            delta = avg[k].float() - start_state[k].float()
            prev = torch.zeros_like(delta) if velocity is None else velocity[k]
            vel[k] = momentum * prev + delta
            out[k] = start_state[k].float() + server_eta * vel[k]
        t0 = time.time()
        m_prev = agg.stats(start_state)["mse"]; m_now = agg.stats(out)["mse"]
        GUARD["time"] = time.time() - t0; GUARD["m_prev"] = m_prev; GUARD["m_now"] = m_now
        if m_now > m_prev:
            GUARD["reset"] = True
            return avg, None
        GUARD["reset"] = False
        return out, vel
    if variant in ("mse_md", "mse_mda", "mse_mdb"):
        # 탐색용(DEVIATIONS_optim.md): 동적 모멘텀. 고정 β로 후보를 만들어 dev MSE(M_now)를 재고,
        # beta_t = β*min(1, M_prev/M_now)로 v와 글로벌을 다시 만든다(재평가 없음, 평가 2회).
        avg, _ = combine_variant(agg, start_state, states, n, "mse", None)
        def build(b):
            vel, out = {}, {}
            for k in start_state:
                delta = avg[k].float() - start_state[k].float()
                prev = torch.zeros_like(delta) if velocity is None else velocity[k]
                vel[k] = b * prev + delta
                out[k] = start_state[k].float() + server_eta * vel[k]
            return out, vel
        cand, _ = build(momentum)
        t0 = time.time()
        m_prev = agg.stats(start_state)["mse"]; m_now = agg.stats(cand)["mse"]
        GUARD["time"] = time.time() - t0
        ratio = min(1.0, m_prev / m_now)
        bt = momentum * (ratio ** 2 if variant == "mse_mda" else ratio)
        GUARD.update(m_prev=m_prev, m_now=m_now, beta_t=bt, evals=2, halved=False)
        res = build(bt)
        if variant == "mse_mdb":
            t1 = time.time()
            m_fin = agg.stats(res[0])["mse"]
            GUARD["evals"] = 3; GUARD["m_final"] = m_fin
            if m_fin > m_prev:
                bt = bt / 2; GUARD["beta_t"] = bt; GUARD["halved"] = True
                res = build(bt)
            GUARD["time"] += time.time() - t1
        return res
    if variant == "mse_nag":
        # FedNAG형 FedGM (nu=beta=0.9, eta=1), 부호: Delta = MSEAvg 집계 - 현재 글로벌. d_0=0.
        #   d_{t+1} = 0.9 d_t + 0.1 Delta ;  h_{t+1} = 0.1 Delta + 0.9 d_{t+1} ;  W_{t+1} = W_t + h_{t+1}
        # d를 먼저 갱신. 1라운드: d=0.1D, h=0.1D+0.09D=0.19D. 상수는 고정(--momentum 무시).
        avg, _ = combine_variant(agg, start_state, states, n, "mse", None)
        vel, out = {}, {}
        for k in start_state:
            delta = avg[k].float() - start_state[k].float()
            prev = torch.zeros_like(delta) if velocity is None else velocity[k]
            vel[k] = 0.9 * prev + 0.1 * delta
            h = 0.1 * delta + 0.9 * vel[k]
            out[k] = start_state[k].float() + 1.0 * h
        return out, vel
    if variant == "mse_dm":
        # 감쇠 모멘텀(CRITERIA_damped.md): v = β v_prev + (1-β) Δ
        avg, _ = combine_variant(agg, start_state, states, n, "mse", None)
        vel, out = {}, {}
        dd = pp = dp = 0.0
        for k in start_state:
            delta = avg[k].float() - start_state[k].float()
            prev = torch.zeros_like(delta) if velocity is None else velocity[k]
            dd += float((delta * delta).sum()); pp += float((prev * prev).sum()); dp += float((delta * prev).sum())
            vel[k] = momentum * prev + (1 - momentum) * delta
            out[k] = start_state[k].float() + server_eta * vel[k]
        GUARD.clear()
        GUARD.update(delta_norm=dd ** 0.5, vprev_norm=pp ** 0.5, cos=(dp / ((dd ** 0.5) * (pp ** 0.5)) if velocity is not None and pp > 0 and dd > 0 else None))
        return out, vel
    if variant == "mse_m":
        avg, _ = combine_variant(agg, start_state, states, n, "mse", None)
        vel, out = {}, {}
        dd = pp = dp = 0.0  # 로그 전용(계산 경로 불변): Δ·v_prev, ‖Δ‖², ‖v_prev‖²
        for k in start_state:
            delta = avg[k].float() - start_state[k].float()
            prev = torch.zeros_like(delta) if velocity is None else velocity[k]
            dd += float((delta * delta).sum()); pp += float((prev * prev).sum()); dp += float((delta * prev).sum())
            vel[k] = momentum * prev + delta
            out[k] = start_state[k].float() + server_eta * vel[k]
        GUARD.clear()
        GUARD.update(delta_norm=dd ** 0.5, vprev_norm=pp ** 0.5, cos=(dp / ((dd ** 0.5) * (pp ** 0.5)) if velocity is not None and pp > 0 and dd > 0 else None))
        return out, vel
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
        tot = sum([alpha for _, alpha in weighted])
        MSE_Q[:] = [float(alpha / tot) for _, alpha in weighted]
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
    if variant in ("fedavg", "eta10", "fedavgm"):
        w = np.asarray(n, dtype=np.float64); w /= w.sum()
        avg = mix_states(states, w)
        if variant in ("fedavg", "eta10"):
            eta = server_eta if variant == "eta10" else 1.0
            return {k: start_state[k].float() + eta * (avg[k].float() - start_state[k].float()) for k in start_state}, velocity
        vel = {}
        out = {}
        for k in start_state:
            delta = avg[k].float() - start_state[k].float()
            prev = torch.zeros_like(delta) if velocity is None else velocity[k]
            vel[k] = momentum * prev + delta
            out[k] = start_state[k].float() + server_eta * vel[k]
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


def dir_stats(start_state, states, prev_mean):
    """참여 클라이언트 delta(local-start) 방향 통계. 수치 경로에 영향 없음."""
    ks = list(start_state.keys())
    ds = [torch.cat([(st[k].float() - start_state[k].float()).flatten() for k in ks]) for st in states]
    mean = torch.stack(ds).mean(0)
    cs = torch.nn.functional.cosine_similarity
    cm = float(np.mean([float(cs(x, mean, dim=0)) for x in ds]))
    pc = [float(cs(ds[i], ds[j], dim=0)) for i in range(len(ds)) for j in range(i + 1, len(ds))]
    cp = None if prev_mean is None else float(cs(mean, prev_mean, dim=0))
    return {"pair_cos": float(np.mean(pc)) if pc else None, "cos_mean": cm, "cos_prev": cp}, mean


def read_target(cell_dir, run, rounds):
    fn = f"{cell_dir}/Run_{run}.jsonl"
    _alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), fn)
    if not os.path.exists(fn) and os.path.exists(_alt):
        fn = _alt
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
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=M.lr_rate)
    ap.add_argument("--dataset", default="nbaiot", choices=["nbaiot", "ton", "cic_km", "cic_iid"])
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--disp_target_cell", default=None,
                    help="목표 셀 디렉터리(예: Checkpoint/Results/EXP/VERIFY/n_S_log); 집계 결과를 그 global_disp로 재스케일")
    ap.add_argument("--dir_log", action="store_true", help="diag2에 pair_cos/cos_mean/cos_prev 추가")
    ap.add_argument("--no_log", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--fx_eps", type=float, default=1e-3)
    ap.add_argument("--variant", choices=["mse", "mse_m", "mse_dm", "mse_nag", "mse_mg", "mse_md", "mse_mda", "mse_mdb", "mse_fx", "mse_adam", "mse_yogi", "p2", "p4", "fedavg", "eta10", "fedavgm", "trimmed"], required=True)
    args = ap.parse_args()
    
    if args.disp_target_cell and args.server_eta != 1.0:
        sys.exit("재스케일과 η는 함께 쓰지 않음")
    if args.disp_target_cell and args.no_log:
        sys.exit("재스케일 셀은 로깅 필수")
    M.lr_rate = args.lr
    torch.set_num_threads(args.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallel = args.topology == "parallel"
    log = not args.no_log

    if args.dataset == "nbaiot":
        clients = load_clients_cfg(args.data_seed, dev, args.config)
    else:
        import dataset_adapter
        clients, M.dim_features = dataset_adapter.load(args.dataset, args.data_seed, dev)
    names = [c["device"] for c in clients]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.variant, args.cell)
    os.makedirs(out_dir, exist_ok=True)

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
        save_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch", args.variant, args.cell, str(run))
        velocity = None; prev_dmean = None
        open(fn, "w").close()
        for rnd in range(args.rounds):
            selected_idx = random.sample(list(range(len(clients))), int(M.num_participants * len(clients)))
            if tgt_join is not None:
                check_join(tgt_join, selected_idx, rnd)
            local, start_state, start, d = train_round3(agg, clients, selected_idx, args.epoch, save_root, parallel, log,
                                                        args.patience)
            n = [m[2] for m in local]; states = [m[0] for m in local]
            dstat = None
            if args.dir_log:
                dstat, prev_dmean = dir_stats(start_state, states, prev_dmean)
            new_state, velocity = combine_variant(agg, start_state, states, n, args.variant, velocity, args.server_eta, args.momentum, (args.beta1, args.beta2, args.tau), args.fx_eps)
            scale = pre_norm = None
            if tgt_disp is not None:
                new_state, scale, pre_norm = rescale_to_target(start_state, new_state, tgt_disp[rnd])
            agg.model.load_state_dict(new_state)
            M.set_val_loss(agg)
            rec = {"round": rnd + 1, "join_clients": selected_idx, "devices": names, "n": n, "eta": (args.server_eta if args.variant == "eta10" else 1.0), "momentum": args.momentum, "lr": args.lr, "opt": [args.beta1, args.beta2, args.tau],
                   "global_loss": agg.val_loss, "global_auc": M.fast_auc_all(agg.model, clients)}
            if args.variant in ("mse_mg", "mse_md", "mse_mda", "mse_mdb") or (args.variant in ("mse_m", "mse_dm") and "cos" in GUARD):
                rec["guard"] = dict(GUARD)
            if log:
                d["global_disp"] = float((M.flat(agg.model) - start).norm())
                ml = float(np.mean(d["local_disp"]))
                d["E"] = d["global_disp"] / ml
                if scale is not None:
                    d["E_pre"] = pre_norm / ml
                    d["scale"] = scale; d["target_disp"] = float(tgt_disp[rnd]); d["pre_disp"] = pre_norm
                if dstat is not None:
                    d.update(dstat)
                if args.variant == "mse_fx":
                    d.update(FX)
                rec["diag2"] = d
            div = rnd + 1 >= 5 and float(np.mean(rec["global_auc"])) < 0.90
            if div: rec["diverged"] = True
            with open(fn, "a") as f:
                f.write(json.dumps({f"round_{rnd+1}": rec}) + "\n")
            print(f"[{args.cell}] run {run} round {rnd+1}/{args.rounds} meanAUC={np.mean(rec['global_auc']):.4f} "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)
            if div:
                with open(f"{out_dir}/DIVERGED", "a") as marker:
                    marker.write(f"run {run} round {rnd+1}\n")
                print(f"[{args.cell}] 발산 표시: run {run} round {rnd+1}; 기록 계속", flush=True)


if __name__ == "__main__":
    main()
