"""단위 검증 2: 병렬 순서 불변, 병렬 첫 클라이언트=직렬 첫 클라이언트, 균등=평균, 직렬 재현(기존 결과 일치).
실행: cd src && python ../analysis/unit_checks_2.py"""
import sys, os, json, glob, copy, subprocess
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import main_exp as M
from Trainer.global_aggregator_exp import make_weights, mix_states

ok = True
def check(name, cond, info=""):
    global ok
    ok &= bool(cond); print(("PASS" if cond else "FAIL"), name, info, flush=True)

dev = torch.device("cuda")
M.set_seeds(0)
clients = M.load_clients(1234, dev)
M.set_seeds(5)
agg = M.build_agg(clients, dev)
base = copy.deepcopy(agg.model.state_dict())

def run(order, parallel):
    agg.model.load_state_dict(base)
    local, _ = M.train_round(agg, clients, order, 2, "Checkpoint/EXP_scratch/unit2", "mse_avg", parallel=parallel)
    return {i: l[0] for i, l in zip(order, local)}
md = lambda a, b: max(float((a[k] - b[k]).abs().max()) for k in a)

pa = run([0, 1, 2], True); pb = run([2, 1, 0], True)
d = max(md(pa[i], pb[i]) for i in (0, 1, 2))
check("병렬: 클라이언트 순서를 바꿔도 로컬 state 동일", d < 1e-6, f"maxdiff={d:.2e}")
sa = run([0, 1, 2], False)
d0 = md(pa[0], sa[0]); d1 = md(pa[1], sa[1])
check("병렬 첫 클라이언트 == 직렬 첫 클라이언트", d0 < 1e-6, f"maxdiff={d0:.2e}")
check("직렬 둘째 클라이언트 != 병렬 둘째(체인 효과 존재)", d1 > 1e-4, f"diff={d1:.2e}")
ps = copy.deepcopy(base)
agg.model.load_state_dict(base)
chk = md(base, agg.model.state_dict())
check("병렬: 라운드 후 base 불변 복사본 정상", chk == 0)

n = [900, 1500, 1100]
st = [pa[i] for i in (0, 1, 2)]
uni = mix_states(st, make_weights(n, 0, None))
ref = {k: sum(s[k].float() for s in st) / 3 for k in st[0]}
check("균등 집계 == 평균", md(uni, ref) < 1e-6)

# 직렬 재현: 기존 결과와 run 0 앞 2라운드 일치 (epoch 100)
R = glob.glob("Checkpoint/Results/Update/10/REPRO-nonIID*/")[0]
procs = []
for m in ("avg", "mse_avg"):
    procs.append(subprocess.Popen([sys.executable, "main_exp.py", "--mode", "final", "--method", m, "--data_seed", "1234",
        "--runs", "0", "--rounds", "2", "--exp", "H12_unit"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
for p in procs: p.wait()
names = [c["device"] for c in clients]
for m in ("avg", "mse_avg"):
    old = [list(json.loads(l).values())[0] for l in open(f"{R}/Run_0/AUC/FL-IoT_0.5_hybrid_{m}_results.json")][:2]
    new = [list(json.loads(l).values())[0] for l in open(f"Checkpoint/Results/EXP/H12_unit/final_{m}/Run_0.jsonl")][:2]
    dd = max(abs(o[nm] - n_["global_auc"][i]) for o, n_ in zip(old, new) for i, nm in enumerate(names))
    ds = all(o["join_clients"] == n_["join_clients"] for o, n_ in zip(old, new))
    dl = max(abs(o["global_loss"] - n_["global_loss"]) for o, n_ in zip(old, new))
    check(f"직렬 {m} 재현 (기존 결과 run0 2라운드)", dd < 1e-9 and ds and dl < 1e-6, f"AUCdiff={dd:.2e} loss={dl:.2e} sel_same={ds}")
print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
