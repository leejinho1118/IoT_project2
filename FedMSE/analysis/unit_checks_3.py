"""단위 검증 3 (PRE_RULES_3.md §8). 실행: cd src && python ../analysis/unit_checks_3.py"""
import sys, os, json, copy, subprocess
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import main_exp as M
import main_exp2 as M2
from DataLoader import load_data
from Trainer.global_aggregator_exp import make_weights, mix_states

ok = True
def check(name, cond, info=""):
    global ok
    ok &= bool(cond); print(("PASS" if cond else "FAIL"), name, info, flush=True)

EXP = "UNIT3"
IID_CFG = "Configuration/scen2-nba-iot-10clients.json"

# --- (a)(c) 서브프로세스: 기존 셀과 비교, 로깅 on/off 비교 -------------------------------------
def launch(cell, topo, eta, nolog, config=None):
    cmd = [sys.executable, "main_exp2.py", "--data_seed", "1234", "--runs", "0", "--rounds", "2", "--exp", EXP,
           "--cell", cell, "--topology", topo, "--server_eta", str(eta)]
    if nolog: cmd.append("--no_log")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
procs = [launch("P1_log", "parallel", 1.0, False), launch("P1_nolog", "parallel", 1.0, True),
         launch("S_log", "serial", 1.0, False)]

def rd(path): return [list(json.loads(l).values())[0] for l in open(path)]
dev = torch.device("cuda")

# --- (b) η 공식 (합성 state) ---------------------------------------------------------------
M.set_seeds(0)
clients = M.load_clients(1234, dev)
M.set_seeds(5)
agg = M.build_agg(clients, dev)
base = copy.deepcopy(agg.model.state_dict())
g = torch.Generator().manual_seed(1)
sts = [{k: (v + 0.1 * torch.randn(v.shape, generator=g).to(v.device)) for k, v in base.items()} for _ in range(5)]
n = [900, 1500, 1100, 700, 1300]
mean = mix_states(sts, make_weights(n, 0, None))
md = lambda a, b: max(float((a[k].double() - b[k].double()).abs().max()) for k in a)
for eta in (2.0, 5.0, 0.5):
    out = M2.aggregate(base, sts, n, True, eta)
    ref = {k: base[k].double() + eta * (sum(s[k].double() for s in sts) / 5 - base[k].double()) for k in base}
    check(f"(b) η={eta} 공식 start+η(mean−start)", md(out, ref) < 1e-5, f"maxdiff={md(out, ref):.2e}")
out1 = M2.aggregate(base, sts, n, True, 1.0)
check("(b) η=1은 mean 그대로(비트 동일)", all(torch.equal(out1[k], mean[k]) for k in mean))
outs = M2.aggregate(base, sts, n, False, 5.0)
check("(b) 직렬에는 η 미적용", all(torch.equal(outs[k], mean[k]) for k in mean))

# --- (e)(f) 직접 호출: 병렬 첫=직렬 첫, 로깅값 검산 ----------------------------------------------
def run_round(order, parallel, ep=3):
    agg.model.load_state_dict(base)
    local, ss, start, d = M2.train_round2(agg, clients, order, ep, "Checkpoint/EXP_scratch/unit3", parallel, True)
    return local, d, start
lp, dp, st0 = run_round([0, 1, 2], True)
ls, ds, _ = run_round([0, 1, 2], False)
d0 = md(lp[0][0], ls[0][0]); d1 = md(lp[1][0], ls[1][0])
check("(e) 병렬 첫 클라이언트 == 직렬 첫 클라이언트", d0 < 1e-6, f"maxdiff={d0:.2e}")
check("(e) 둘째 클라이언트는 다름(체인 효과)", d1 > 1e-4, f"diff={d1:.2e}")
check("(f) 첫 클라이언트 이동거리 병렬==직렬", abs(dp["local_disp"][0] - ds["local_disp"][0]) < 1e-6,
      f"{dp['local_disp'][0]:.5f} vs {ds['local_disp'][0]:.5f}")
flat = lambda s: torch.cat([s[k].reshape(-1) for k in base])   # 파라미터 순서 = state_dict 순서
recomputed = [float((flat(l[0]) - st0).norm()) for l in lp]
check("(f) 이동거리 로깅 == 로컬 state에서 재계산", max(abs(a - b) for a, b in zip(recomputed, dp["local_disp"])) < 1e-4,
      f"{max(abs(a-b) for a,b in zip(recomputed, dp['local_disp'])):.2e}")
vs = [flat(l[0]) for l in lp]
pm = np.mean([float((vs[i] - vs[j]).norm()) for i, j in ((0, 1), (0, 2), (1, 2))])
check("(f) 쌍별 거리 로깅 == 재계산", abs(pm - dp["pair_mean"]) < 1e-4, f"{pm:.5f} vs {dp['pair_mean']:.5f}")
check("(f) 쌍별 거리 합성 검산", abs(M2.pair_mean([torch.tensor([0., 0.]), torch.tensor([3., 4.]), torch.tensor([0., 4.])]) - (5 + 4 + 3) / 3) < 1e-9)
check("(f) 에폭=3, 스텝=에폭×배치수", dp["epochs"] == [3, 3, 3] and
      dp["steps"] == [3 * len(clients[i]["train_loader"]) for i in (0, 1, 2)], f"epochs={dp['epochs']} steps={dp['steps']}")
# 평균 이동: η=1 병렬의 global_disp = ‖mean−start‖
mean_p = M2.aggregate(base, [l[0] for l in lp], [l[1] for l in lp], True, 1.0)
gd2 = float((flat(M2.aggregate(base, [l[0] for l in lp], [l[2] for l in lp], True, 2.0)) - st0).norm())
gd1 = float((flat(mean_p) - st0).norm())
check("(f) η=2의 global 이동 == 2×(η=1 이동)", abs(gd2 - 2 * gd1) < 1e-4, f"{gd2:.5f} vs {2*gd1:.5f}")

# --- (d) IID 설정 ---------------------------------------------------------------------------
cfg = json.load(open(IID_CFG))
check("(d) IID 설정 data_path", cfg["data_path"].endswith("IID-10-Client_Data"), cfg["data_path"])
M.set_seeds(0)
ci = M2.load_clients_cfg(1234, dev, IID_CFG)
M.set_seeds(0)
cn = M2.load_clients_cfg(1234, dev, "Configuration/repro-noniid-10clients.json")
import random
random.seed(1234); np.random.seed(1234)
sel = random.sample(cfg["devices_list"], 10)
exp_sizes = [int(0.4 * len(load_data(os.path.join(cfg["data_path"], d["normal_data_path"])))) for d in sel]
got = [len(c["train_loader"].dataset) for c in ci]
check("(d) IID 클라이언트 학습 크기 == IID csv 행 수의 40%", got == exp_sizes, f"{got}")
check("(d) 비-IID와 크기 다름", got != [len(c["train_loader"].dataset) for c in cn],
      f"nonIID={[len(c['train_loader'].dataset) for c in cn]}")
check("(d) config_path 전역 복원", M.config_path == "Configuration/repro-noniid-10clients.json")

# --- (a)(c) 결과 비교 ------------------------------------------------------------------------
for p in procs: p.wait()
base_dir = f"Checkpoint/Results/EXP/{EXP}"
old = rd("Checkpoint/Results/EXP/H12/final_uniform_parallel/Run_0.jsonl")[:2]
olds = rd("Checkpoint/Results/EXP/H12/final_uniform/Run_0.jsonl")[:2]
newl = rd(f"{base_dir}/P1_log/Run_0.jsonl"); newn = rd(f"{base_dir}/P1_nolog/Run_0.jsonl"); news = rd(f"{base_dir}/S_log/Run_0.jsonl")
dauc = lambda A, B: max(abs(x - y) for a, b in zip(A, B) for x, y in zip(a["global_auc"], b["global_auc"]))
check("(a) η=1 병렬 == 기존 P-U run0 앞 2라운드 AUC", dauc(old, newl) <= 1e-9 and all(a["join_clients"] == b["join_clients"] for a, b in zip(old, newl)),
      f"AUCdiff={dauc(old, newl):.2e} loss={max(abs(a['global_loss']-b['global_loss']) for a,b in zip(old,newl)):.2e}")
check("(a) 직렬(코드 경로 재확인) == 기존 S-U run0", dauc(olds, news) <= 1e-9 and all(a["join_clients"] == b["join_clients"] for a, b in zip(olds, news)),
      f"AUCdiff={dauc(olds, news):.2e}")
check("(c) 로깅 on/off 동일 궤적", dauc(newl, newn) == 0.0 and all(a["global_loss"] == b["global_loss"] for a, b in zip(newl, newn)),
      f"AUCdiff={dauc(newl, newn):.2e}")
check("(c) no_log 기록에 diag2 없음 / log에는 있음", "diag2" not in newn[0] and "diag2" in newl[0])
print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
