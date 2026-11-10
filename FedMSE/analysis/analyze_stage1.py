"""1단계 판정 (PRE_RULES.md '1단계 판정' 그대로). 실행: ~/fedmse_venv/bin/python analyze_stage1.py"""
import json, glob, os
import numpy as np
from scipy.stats import spearmanr

D = os.path.expanduser("~/fedmse/src/Checkpoint/Results/EXP/S1/diag")
recs = []
for fn in sorted(glob.glob(D + "/Run_*.jsonl")):
    for l in open(fn):
        r = list(json.loads(l).values())[0]; r["run"] = int(fn.split("_")[-1].split(".")[0]); recs.append(r)
print("rounds:", len(recs))
sp = lambda a, b: spearmanr(a, b)[0]
out = {}

# 약한 클라이언트: global_auc 평균 최저 2개 (device 이름 기준)
dev_names = recs[0]["devices"]
mean_auc = {}
for r in recs:
    assert r["devices"] == dev_names
    for g, a in enumerate(r["global_auc"]):
        mean_auc.setdefault(dev_names[g], []).append(a)
means = {k: float(np.mean(v)) for k, v in mean_auc.items()}
weak_dev = sorted(means, key=means.get)[:2]
weak_blocks = [dev_names.index(d) for d in weak_dev]
out["weak_devices"] = weak_dev; out["client_mean_auc"] = means

# ---- B ----
ratios, rho_b, weakflag = [], [], []
for r in recs:
    bm = np.array([s["block_mse"] for s in r["stats"]])          # 5 x 10
    ratios.append(np.median(bm, axis=0).max() / np.median(bm, axis=0).min())
    lr = np.log(bm / np.median(bm, axis=0, keepdims=True))        # clip 없음
    mse = np.array([s["mse"] for s in r["stats"]])
    rho_b.append(sp(mse, lr.mean(axis=1)))
    weakflag.append(lr[:, weak_blocks].max() > np.log(1.25))
rho_b = np.array(rho_b); valid = ~np.isnan(rho_b)
B = {"a_median_ratio": float(np.median(ratios)), "a_ratio_min": float(np.min(ratios)), "a_ratio_max": float(np.max(ratios)),
     "b_frac_rho_lt_0.8": float(np.mean(rho_b[valid] < 0.8)), "b_rho_mean": float(rho_b[valid].mean()), "b_excluded": int((~valid).sum()),
     "c_frac_weak_logratio_gt_log1.25": float(np.mean(weakflag))}
B["a"] = B["a_median_ratio"] >= 3
B["b"] = B["b_frac_rho_lt_0.8"] >= 0.5
B["c"] = B["c_frac_weak_logratio_gt_log1.25"] >= 0.5
B["viable"] = bool(B["a"] and B["b"] and B["c"])
out["B"] = B

# ---- A ----
A = {}
for name in ("A1", "A2", "A3"):
    a_ok, rho_ok, rhos, excl = [], [], [], 0
    for r in recs:
        st = r["stats"]
        if name == "A3":
            raw = np.array([s["A3_raw"] for s in st]); med = np.median(raw)
            sc = raw / med; bsd = np.mean([s["boot"]["A3"] for s in st]) / med
        else:
            sc = np.array([s[name] for s in st]); bsd = np.mean([s["boot"][name] for s in st])
        a_ok.append(np.std(sc) > bsd)
        auc = np.array([np.mean(x) for x in r["local_auc"]])
        rho = sp(-sc, auc)
        if np.isnan(rho): excl += 1; continue
        rhos.append(rho); rho_ok.append(rho > 0.3)
    A[name] = {"a_frac": float(np.mean(a_ok)), "b_frac_rho_gt_0.3": float(np.mean(rho_ok)),
               "b_rho_mean": float(np.mean(rhos)), "excluded": excl,
               "a": bool(np.mean(a_ok) >= 0.6), "b": bool(np.mean(rho_ok) >= 0.6)}
    A[name]["pass"] = A[name]["a"] and A[name]["b"]
out["A"] = A
out["A_passed"] = [k for k, v in A.items() if v["pass"]]

# ---- 부수 ----
ent = np.array([r["mse_entropy_ratio"] for r in recs])
out["entropy"] = {"mean": float(ent.mean()), "min": float(ent.min()), "mse_avg_approx_uniform": bool(ent.mean() > 0.98)}
orho = [sp([c["order"] for c in r["chain"]], [s["mse"] for s in r["stats"]]) for r in recs]
orho = np.array([x for x in orho if not np.isnan(x)])
out["chain"] = {"dist_after_last_mean": float(np.mean([r["chain"][-1]["dist_after"] for r in recs])),
                "dist_after_first_mean": float(np.mean([r["chain"][0]["dist_after"] for r in recs])),
                "order_vs_mse_rho_mean": float(orho.mean()), "order_vs_mse_frac_abs_gt_0.5": float(np.mean(np.abs(orho) > 0.5))}
out["B_viable"] = B["viable"]
out["both_dropped"] = (not B["viable"]) and (len(out["A_passed"]) == 0)
json.dump(out, open(os.path.expanduser("~/fedmse/analysis/stage1_result.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
