"""
후속 실험 실행기 (main_repro.py 복사·수정, 원본 불변). hybrid 전용.
mode:
  diag           : MSEAvg 궤적 + 진단 로깅 (1단계)
  counterfactual : MSEAvg 궤적 + 같은 로컬 모델로 후보 global 다수 평가 (2단계)
  final          : --method {avg, mse_avg, B, A} 엔드투엔드 (3단계), 동결 설정 사용
공통: 조기종료 없음, 지정 라운드 전부 실행. 결과는 Checkpoint/Results/EXP/<exp>/ 아래 JSONL.
"""
import os, sys, json, copy, random, argparse, hashlib, time, logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc

from Model import Shrink_Autoencoder
from DataLoader import load_data, IoTDataProccessor
from Trainer import ClientTrainer
from Trainer.global_aggregator_exp import (ExpAggregator, make_weights, mix_states)
from Model.Centroid import CentroidBasedOneClassClassifier

logging.getLogger().setLevel(logging.WARNING)

epoch_default = 100
num_participants = 0.5
lr_rate = 1e-5
shrink_lambda = 10
network_size = 10
batch_size = 12
dim_features = 115
config_path = "Configuration/repro-noniid-10clients.json"
FROZEN = "../analysis/frozen_selection.json"
FROZEN_SHA = "../analysis/frozen_selection.sha256"


def set_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_clients(data_seed, dev):
    """main_repro.py의 데이터 준비 부분을 그대로 복제(난수 소비 순서 동일). 평가용 텐서 포함."""
    random.seed(data_seed); np.random.seed(data_seed)
    with open(config_path) as f:
        config = json.load(f)
    devices_list = random.sample(config['devices_list'], network_size)
    clients = []
    for device in devices_list:
        base = config['data_path']
        normal_data = load_data(os.path.join(base, device["normal_data_path"]))
        normal_data = normal_data.sample(frac=1).reset_index(drop=True)
        abnormal_data = load_data(os.path.join(base, device["abnormal_data_path"]))
        abnormal_data = abnormal_data.sample(frac=1).reset_index(drop=True)
        new_normal_data = load_data(os.path.join(base, device["test_normal_data_path"]))
        n = len(normal_data)
        tr, va, dv = int(0.4 * n), int(0.1 * n), int(0.4 * n)
        train_n = normal_data[:tr]; valid_n = normal_data[tr:tr + va]
        dev_n = normal_data[tr + va:tr + va + dv]; test_n = normal_data[tr + va + dv:]
        proc = IoTDataProccessor(scaler="standard")
        ptr, ltr = proc.fit_transform(train_n)
        pva, lva = proc.transform(valid_n)
        pte, lte = proc.transform(test_n)
        pab, lab = proc.transform(abnormal_data, type="abnormal")
        pnew, lnew = proc.transform(new_normal_data)
        pte = np.concatenate([pte, pnew], axis=0)
        lte_all = np.concatenate([lte, lnew], axis=0)
        test_x = np.concatenate([pte, pab], axis=0).astype(np.float32)
        test_y = np.concatenate([lte_all, lab], axis=0)
        from torch.utils.data import DataLoader
        from DataLoader import IoTDataset
        train_loader = DataLoader(IoTDataset(ptr, ltr), batch_size=batch_size, pin_memory=True)
        valid_loader = DataLoader(IoTDataset(pva, lva), batch_size=batch_size, pin_memory=True)
        clients.append({
            "device": device['name'], "train_loader": train_loader, "valid_loader": valid_loader,
            "dev_normal_dataset": dev_n,
            "train_x": torch.tensor(ptr.astype(np.float32)).to(dev),
            "test_x": torch.tensor(test_x).to(dev), "test_y": test_y,
        })
    return clients


@torch.no_grad()
def _latent(model, x, bs=8192):
    return torch.cat([model.encoder(x[i:i + bs]) for i in range(0, len(x), bs)], 0).cpu().numpy()


def fast_auc_all(model, clients):
    """Evaluator(hybrid, AUC)와 같은 계산(Centroid 분류기 재적합 → roc_curve/auc)을 큰 배치로."""
    model.eval()
    out = []
    for c in clients:
        tl = _latent(model, c["train_x"]); te = _latent(model, c["test_x"])
        cen = CentroidBasedOneClassClassifier(); cen.fit(tl)
        pred = cen.get_density(te)
        if not np.all(np.isfinite(pred)):
            pred = np.nan_to_num(pred)
        fpr, tpr, _ = roc_curve(c["test_y"], pred)
        out.append(float(auc(fpr, tpr)))
    return out


def flat(model):
    return torch.cat([p.data.reshape(-1) for p in model.parameters()]).clone()


def build_agg(clients, dev):
    gm = Shrink_Autoencoder(input_dim=dim_features, output_dim=dim_features,
                            shrink_lambda=shrink_lambda, latent_dim=11, hidden_neus=50)
    agg = ExpAggregator(gm, update_type="mse_avg")
    min_len = min(len(c['dev_normal_dataset']) for c in clients)
    dd = [c['dev_normal_dataset'].sample(n=min_len) for c in clients]
    agg.create_dev_dataset({"dataset": np.concatenate(dd, axis=0)})
    return agg


def set_val_loss(agg):
    agg.model.eval()
    with torch.no_grad():
        _, _, vl = agg.model(agg.dev_t)
        agg.val_loss = vl.item()


def train_round(agg, clients, selected_idx, epochs, save_root, update_type, record_chain=False,
                parallel=False):
    total = sum(len(clients[i]['train_loader'].dataset) for i in selected_idx)
    start = flat(agg.model)
    start_state = copy.deepcopy(agg.model.state_dict()) if parallel else None
    local, chain = [], []
    for k, i in enumerate(selected_idx):
        c = clients[i]
        if parallel:   # 병렬: 모든 클라이언트가 라운드 시작 global에서 독립 학습
            agg.model.load_state_dict(start_state)
        if record_chain:
            chain.append({"order": k, "dist_before": float((flat(agg.model) - start).norm())})
        sd = os.path.join(save_root, c['device'])
        tr = ClientTrainer(model=agg.model, save_dir=sd, epoch=epochs, lr_rate=lr_rate, update_type=update_type)
        tr.run(c["train_loader"], c["valid_loader"])
        local.append((copy.deepcopy(tr.model.state_dict()), total, len(c["train_loader"].dataset)))
        if record_chain:
            chain[-1]["dist_after"] = float((flat(agg.model) - start).norm())
    return local, chain


def entropy_ratio(w):
    w = np.asarray(w, dtype=np.float64)
    return float(-(w * np.log(w + 1e-30)).sum() / np.log(len(w)))


def candidates(agg, n, stats, a_scores, rng):
    sc = agg.scores_all(stats)
    c = {"uniform": make_weights(n, 0, None), "fedavg": make_weights(n, 1, None),
         "mseavg": agg.mse_avg_weights(stats)}
    for beta in (0, 0.5):
        for T in (0.5, 1, 2):
            for lam in (0, 0.5):
                c[f"B|b{beta}|T{T}|l{lam}"] = make_weights(n, beta, T, sc[f"B_lam{lam}"])
            for a in a_scores:
                c[f"{a}|b{beta}|T{T}"] = make_weights(n, beta, T, sc[a])
    for k in range(10):
        c[f"rand{k}"] = rng.dirichlet(np.ones(len(n)))
    return c


def load_frozen():
    with open(FROZEN, "rb") as f:
        raw = f.read()
    want = open(FROZEN_SHA).read().split()[0]
    got = hashlib.sha256(raw).hexdigest()
    if want != got:
        sys.exit(f"동결 설정 sha256 불일치: {want} != {got}")
    return json.loads(raw)


def weights_for_method(method, agg, local, frozen):
    n = [m[2] for m in local]
    states = [m[0] for m in local]
    stats = [agg.stats(s) for s in states]
    sc = agg.scores_all(stats)
    if method == "B":
        p = frozen["B"]
        return make_weights(n, p["beta"], p["T"], sc[f"B_lam{p['lam']}"])
    p = frozen["A"]
    return make_weights(n, p["beta"], p["T"], sc[p["score"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["diag", "counterfactual", "final"])
    ap.add_argument("--data_seed", type=int, required=True)
    ap.add_argument("--runs", type=int, nargs="+", required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--epoch", type=int, default=epoch_default)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--method", default="mse_avg", choices=["avg", "mse_avg", "uniform", "B", "A"])
    ap.add_argument("--topology", default="serial", choices=["serial", "parallel"])
    ap.add_argument("--log_cf", action="store_true")
    ap.add_argument("--a_scores", default="")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen = load_frozen() if (args.mode == "final" and args.method in ("A", "B")) else None
    a_scores = [s for s in args.a_scores.split(",") if s]

    clients = load_clients(args.data_seed, dev)
    names = [c["device"] for c in clients]
    tag = args.mode if args.mode != "final" else f"final_{args.method}"
    if args.mode == "final" and args.topology == "parallel":
        tag += "_parallel"
    out_dir = f"Checkpoint/Results/EXP/{args.exp}/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    for run in args.runs:
        fn = f"{out_dir}/Run_{run}.jsonl"
        if os.path.exists(fn) and sum(1 for _ in open(fn)) == args.rounds:
            print(f"skip {fn} (complete)"); continue
        t0 = time.time()
        set_seeds(run * 10000)
        agg = build_agg(clients, dev)
        update_type = "avg" if (args.mode == "final" and args.method == "avg") else "mse_avg"
        agg.update_type = update_type
        save_root = f"Checkpoint/EXP_scratch/{args.exp}/{tag}/{run}"
        boot_rng = np.random.RandomState(run)
        open(fn, "w").close()
        for rnd in range(args.rounds):
            selected_idx = random.sample(list(range(len(clients))), int(num_participants * len(clients)))
            local, chain = train_round(agg, clients, selected_idx, args.epoch, save_root, update_type,
                                       record_chain=(args.mode == "diag"),
                                       parallel=(args.topology == "parallel"))
            n = [m[2] for m in local]
            states = [m[0] for m in local]
            rec = {"round": rnd + 1, "join_clients": selected_idx, "devices": names, "n": n}

            if args.mode in ("diag", "counterfactual"):
                stats = [agg.stats(s, boot_rng=boot_rng, n_boot=20 if args.mode == "diag" else 0) for s in states]
            if args.mode == "diag":
                rec["chain"] = chain
                rec["stats"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()} for s in stats]
                wm = agg.mse_avg_weights(stats)
                rec["mse_weights"] = wm.tolist(); rec["mse_entropy_ratio"] = entropy_ratio(wm)
                rec["local_auc"] = [fast_auc_all(_load(agg, s), clients) for s in states]

            if args.mode == "counterfactual":
                cands = candidates(agg, n, stats, a_scores, np.random.RandomState(run * 1000 + rnd))
                cres = {}
                for name, w in cands.items():
                    cres[name] = {"w": [float(x) for x in w],
                                  "auc": fast_auc_all(_load(agg, mix_states(states, w)), clients)}
                rec["cand"] = cres
                rec["scores"] = {k: v.tolist() for k, v in agg.scores_all(stats).items()}
                mixed_mse = mix_states(states, cands["mseavg"])

            # --- 궤적 갱신 ---
            if args.mode == "final" and args.log_cf and args.method == "mse_avg":
                st = [agg.stats(x) for x in states]
                wm = agg.mse_avg_weights(st)
                wu = make_weights(n, 0, None)
                rec["cf"] = {"w_mse": wm.tolist(), "entropy": entropy_ratio(wm),
                             "auc_uniform": fast_auc_all(_load(agg, mix_states(states, wu)), clients),
                             "auc_mse": fast_auc_all(_load(agg, mix_states(states, wm)), clients)}
            if args.mode == "final" and args.method == "uniform":
                agg.model.load_state_dict(mix_states(states, make_weights(n, 0, None)))
                set_val_loss(agg)
            elif args.mode == "final" and args.method in ("A", "B"):
                w = weights_for_method(args.method, agg, local, frozen)
                agg.model.load_state_dict(mix_states(states, w))
                set_val_loss(agg)
                rec["w"] = [float(x) for x in w]
            else:
                agg.update(local_models=local)   # 원본 fed_avg / fed_mse_avg 그대로
            rec["global_loss"] = agg.val_loss
            if args.mode == "counterfactual":
                gd = max(float((agg.model.state_dict()[k].float() - mixed_mse[k]).abs().max()) for k in mixed_mse)
                rec["mseavg_reimpl_maxdiff"] = gd
            rec["global_auc"] = fast_auc_all(agg.model, clients)
            with open(fn, "a") as f:
                f.write(json.dumps({f"round_{rnd+1}": rec}) + "\n")
            print(f"[{tag}] run {run} round {rnd+1}/{args.rounds} meanAUC={np.mean(rec['global_auc']):.4f} "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)


def _load(agg, state):
    agg.scratch.load_state_dict(state)
    return agg.scratch


if __name__ == "__main__":
    main()
