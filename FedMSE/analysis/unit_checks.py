"""단위 검증: (1) 집계식 수치 일치 (2) 빠른 평가 = Evaluator (3) 조기종료 사후 시뮬레이션이 기존 라운드 수 복원. 실행: cd src && python ../analysis/unit_checks.py"""
import sys, os, json, glob, copy
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from torch.utils.data import DataLoader
from Model import Shrink_Autoencoder
from DataLoader import IoTDataset
from Evaluator import Evaluator
from Trainer import GlobalAggregator
from Trainer.global_aggregator_exp import ExpAggregator, make_weights, mix_states, b_score
import main_exp as M

ok = True
def check(name, cond, info=""):
    global ok
    ok &= bool(cond); print(("PASS" if cond else "FAIL"), name, info)

dev = torch.device("cuda")
torch.manual_seed(0); np.random.seed(0)
mk = lambda: Shrink_Autoencoder(input_dim=115, output_dim=115, shrink_lambda=10, latent_dim=11, hidden_neus=50)
agg = ExpAggregator(mk(), "mse_avg")
agg.create_dev_dataset({"dataset": np.random.randn(10*40, 115) * np.linspace(0.5, 2, 115)})
base = {k: v.clone() for k, v in agg.model.state_dict().items()}
states, n = [], [900, 1500, 1100, 2000, 700]
for _ in n:
    states.append({k: (v + 0.05 * torch.randn_like(v)) for k, v in base.items()})
local = [(s, sum(n), ni) for s, ni in zip(states, n)]

# 1a 균등평균
uni = mix_states(states, make_weights(n, 0, None))
ref = {k: sum(s[k] for s in states) / len(states) for k in states[0]}
d = max(float((uni[k] - ref[k]).abs().max()) for k in ref)
check("beta=0,T=inf == 균등평균", d < 1e-6, f"maxdiff={d:.2e}")
# 1b FedAvg (원본 fed_avg)
orig = GlobalAggregator(mk(), "avg"); orig.fed_avg(local)
fa = mix_states(states, make_weights(n, 1, None))
d = max(float((fa[k] - orig.model.state_dict()[k]).abs().max()) for k in fa)
check("beta=1,T=inf == 원본 fed_avg", d < 1e-6, f"maxdiff={d:.2e}")
# 1c MSEAvg 재구현 (원본 fed_mse_avg; 같은 dev 필요)
orig2 = GlobalAggregator(mk(), "mse_avg"); orig2.create_dev_dataset({"dataset": agg.dev_dataset}) if False else None
orig2.dev_dataset = agg.dev_dataset
orig2.fed_mse_avg(local)
stats = [agg.stats(s) for s in states]
ms = mix_states(states, agg.mse_avg_weights(stats))
d = max(float((ms[k] - orig2.model.state_dict()[k]).abs().max()) for k in ms)
check("MSEAvg 재구현 == 원본 fed_mse_avg", d < 1e-5, f"maxdiff={d:.2e}")
# 1d 통계 일관성: pooled mse = 블록 평균, shrink loss = forward 세 번째 값
m = agg.scratch; m.load_state_dict(states[0]); m.eval()
with torch.no_grad(): _, out, loss = m(agg.dev_t)
check("pooled mse == mean(block mse)", abs(stats[0]["mse"] - stats[0]["block_mse"].mean()) < 1e-6)
check("A1 == forward shrink loss", abs(stats[0]["A1"] - loss.item()) < 1e-5)
# 1e 점수 방향/불변성
bm = np.abs(np.random.randn(5, 10)) + 0.5
check("B 점수 규모 불변", np.allclose(b_score(bm, 0.5), b_score(bm * 7.3, 0.5)))
w = make_weights(n, 0.5, 1, np.array([1., 2, 3, 4, 5]))
check("낮은 점수가 큰 가중", w[0] > w[4] and abs(w.sum() - 1) < 1e-12)

# 2 빠른 평가 == Evaluator
clients = M.load_clients(1234, dev)
model = agg.scratch; model.load_state_dict(states[0]); model.eval()
fast = M.fast_auc_all(model, clients[:3])
slow = []
for c in clients[:3]:
    te = DataLoader(IoTDataset(c["test_x"].cpu().numpy(), c["test_y"]), batch_size=12)
    r = Evaluator(model, metric="AUC", model_type="hybrid").evaluate(te, c["train_loader"])
    slow.append(r[0])
d = max(abs(a - b) for a, b in zip(fast, slow))
check("fast_auc_all == Evaluator", d < 1e-6, f"maxdiff={d:.2e}")

# 3 조기종료 사후 시뮬레이션 (원본 코드 글자 그대로)
def sim_stop(losses, patience=1):
    mv, worse = float("inf"), 0
    for r, l in enumerate(losses, 1):
        if l < mv: mv, worse = l, 0
        if l >= mv:
            worse += 1
            if worse > patience: return r
    return len(losses)
R = glob.glob("Checkpoint/Results/Update/10/REPRO*/")[0]
exp = {}
for u in ("avg", "mse_avg"):
    for r in range(5):
        L = [json.loads(l) for l in open(f"{R}/Run_{r}/AUC/FL-IoT_0.5_hybrid_{u}_results.json")]
        losses = [list(x.values())[0]["global_loss"] for x in L]
        # 로그가 이미 조기종료로 끝난 것이므로 '전체 로그에 시뮬레이션을 적용하면 마지막 줄에서 멈춰야' 함
        s = sim_stop(losses)
        check(f"조기종료 복원 {u} run{r}", s == len(losses) and len(losses) >= 3 or (s == len(losses)), f"logged={len(losses)} sim={s}")
print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
