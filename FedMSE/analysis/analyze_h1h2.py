"""H1/H2 분석 (PRE_RULES_2.md 그대로). 실행: cd src && python ../analysis/analyze_h1h2.py --exp H12 --runs 0-9 --out ../analysis/h1h2_result.json"""
import argparse, json, os, sys, glob
import numpy as np
from scipy import stats as st

TOL = 1e-9
SEED = 20261109
NB = 10000
CELLS = {"S-U": "final_uniform", "S-A": "final_avg", "S-M": "final_mse_avg",
         "P-U": "final_uniform_parallel", "P-A": "final_avg_parallel", "P-M": "final_mse_avg_parallel"}
R_MAIN = slice(3, 10)    # 라운드 4~10 (포함)
R_LATE = slice(10, 20)   # 라운드 11~20
ge = lambda x, c: x >= c - TOL
gt = lambda x, c: x > c + TOL
le = lambda x, c: x <= c + TOL
lt = lambda x, c: x < c - TOL


def sim_stop(losses, patience=1):
    mv, worse = float("inf"), 0
    for r, l in enumerate(losses, 1):
        if l < mv: mv, worse = l, 0
        if l >= mv:
            worse += 1
            if worse > patience: return r
    return len(losses)


def load(base, exp, runs, rounds=20):
    D = {}
    for cell, tag in CELLS.items():
        auc, loss, cf = [], [], []
        for r in runs:
            fn = f"{base}/{exp}/{tag}/Run_{r}.jsonl"
            L = [list(json.loads(l).values())[0] for l in open(fn)]
            assert len(L) == rounds, (fn, len(L))
            auc.append([x["global_auc"] for x in L]); loss.append([x["global_loss"] for x in L])
            cf.append([x.get("cf") for x in L])
        D[cell] = {"auc": np.array(auc) * 100, "loss": np.array(loss), "cf": cf}   # auc: (run, round, client) %p
    return D


def metr(D, cell, sl):
    a = D[cell]["auc"][:, sl, :]
    return a.mean(axis=2).mean(axis=1), a.min(axis=2).mean(axis=1)      # M1, M2 (run별)


def paper_proto(D, cell):
    m1, m2 = [], []
    for ri in range(D[cell]["auc"].shape[0]):
        s = sim_stop(list(D[cell]["loss"][ri]))
        a = D[cell]["auc"][ri, s - 1]
        m1.append(a.mean()); m2.append(a.min())
    return np.array(m1), np.array(m2)


def boot_ci(d, level):
    rs = np.random.RandomState(SEED)
    idx = rs.randint(0, len(d), (NB, len(d)))
    bm = d[idx].mean(axis=1)
    a = (1 - level) / 2 * 100
    return float(np.percentile(bm, a)), float(np.percentile(bm, 100 - a))


def paired(d):
    d = np.asarray(d, float)
    lo95, hi95 = boot_ci(d, 0.95); lo90, hi90 = boot_ci(d, 0.90)
    if d.std() > 0:
        p = float(st.ttest_rel(d, np.zeros_like(d)).pvalue)
    else:   # 차이가 전부 같음: 0이면 p=1, 0이 아니면 t=무한대 → p=0
        p = 1.0 if abs(d.mean()) <= TOL else 0.0
    try:
        pw = float(st.wilcoxon(d).pvalue)
    except Exception:
        pw = float("nan")
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0, "n": len(d),
            "ci95": [lo95, hi95], "ci90": [lo90, hi90], "t_p": p, "wilcoxon_p": pw}


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); out[i] = min(1.0, run)
    return out


DELTA = {"M1": 0.3, "M2": 1.0}


def h1(D):
    res = {}
    for nm, pair in (("M1", 0), ("M2", 1)):
        d = metr(D, "S-M", R_MAIN)[pair] - metr(D, "S-U", R_MAIN)[pair]
        s = paired(d); dl = DELTA[nm]
        eq = ge(s["ci90"][0], -dl) and le(s["ci90"][1], dl)
        diff = (gt(s["ci95"][0], 0) or lt(s["ci95"][1], 0)) and ge(abs(s["mean"]), dl)
        s["verdict"] = "동등" if eq else ("차이 있음" if diff else "판정불가")
        res[nm] = s
    vs = [res["M1"]["verdict"], res["M2"]["verdict"]]
    res["overall"] = "차이 있음" if "차이 있음" in vs else ("동등" if vs == ["동등", "동등"] else "판정불가")
    return res


def contrast_verdict(sm1, sm2, p1h, p2h):
    """개선/악화 판정. sm: paired 결과, p*h: 해당 지표의 Holm p."""
    imp = wor = False
    for (m, o, ph, dl) in ((sm1, sm2, p1h, 0.3), (sm2, sm1, p2h, 1.0)):
        if gt(m["ci95"][0], 0) and lt(ph, 0.05) and ge(m["mean"], dl) and ge(o["mean"], -0.2): imp = True
        if lt(m["ci95"][1], 0) and lt(ph, 0.05) and le(m["mean"], -dl) and le(o["mean"], 0.2): wor = True
    if imp and wor: return "판정불가"
    return "개선" if imp else ("악화" if wor else "차이 없음/판정불가")


def h2(D):
    res = {"contrasts": {}}
    pairs = [("P-U", "S-U"), ("P-A", "S-A"), ("P-M", "S-M")]
    S, keys, ps = {}, [], []
    for a, b in pairs:
        for nm, pi in (("M1", 0), ("M2", 1)):
            s = paired(metr(D, a, R_MAIN)[pi] - metr(D, b, R_MAIN)[pi])
            S[(a, b, nm)] = s; keys.append((a, b, nm)); ps.append(s["t_p"])
    hp = holm(ps)
    for k, h in zip(keys, hp): S[k]["holm_p"] = float(h)
    for a, b in pairs:
        v = contrast_verdict(S[(a, b, "M1")], S[(a, b, "M2")], S[(a, b, "M1")]["holm_p"], S[(a, b, "M2")]["holm_p"])
        res["contrasts"][f"{a}-{b}"] = {"M1": S[(a, b, "M1")], "M2": S[(a, b, "M2")], "verdict": v}
    # 주 대비 P-A − S-M: Holm(2개) 기준 판정 + raw 기준 판정 병기
    ms = {nm: paired(metr(D, "P-A", R_MAIN)[pi] - metr(D, "S-M", R_MAIN)[pi]) for nm, pi in (("M1", 0), ("M2", 1))}
    h2p = holm([ms["M1"]["t_p"], ms["M2"]["t_p"]])
    for nm, h in zip(("M1", "M2"), h2p): ms[nm]["holm_p"] = float(h)
    res["main_P-A_vs_S-M"] = {"M1": ms["M1"], "M2": ms["M2"],
        "verdict_holm2": contrast_verdict(ms["M1"], ms["M2"], ms["M1"]["holm_p"], ms["M2"]["holm_p"]),
        "verdict_raw": contrast_verdict(ms["M1"], ms["M2"], ms["M1"]["t_p"], ms["M2"]["t_p"])}
    return res


def mech(D):
    ent, dauc = [], []
    for run in D["S-M"]["cf"]:
        for c in run:
            ent.append(c["entropy"])
            dauc.append(abs(np.mean(c["auc_uniform"]) - np.mean(c["auc_mse"])) * 100)
    ent = np.array(ent); dauc = np.array(dauc)
    frac = float(np.mean([gt(e, 0.98) for e in ent]))
    return {"n_rounds": len(ent), "entropy_gt_0.98_frac": frac, "pass_a": bool(ge(frac, 0.95)),
            "mean_abs_auc_diff_pp": float(dauc.mean()), "pass_b": bool(lt(dauc.mean(), 0.1)),
            "entropy_min": float(ent.min()), "entropy_mean": float(ent.mean())}


def extras(D):
    ex = {"sensitivity_11_20": {}, "paper_protocol": {}, "client_sd_4_10": {}, "weak_clients": {}, "S-A_minus_S-M": {}}
    for c in CELLS:
        a, b = metr(D, c, R_LATE); ex["sensitivity_11_20"][c] = {"M1": float(a.mean()), "M2": float(b.mean())}
        pa, pb = paper_proto(D, c); ex["paper_protocol"][c] = {"M1": float(pa.mean()), "M2": float(pb.mean())}
        ex["client_sd_4_10"][c] = float(D[c]["auc"][:, R_MAIN, :].std(axis=2).mean())
    for nm, pi in (("M1", 0), ("M2", 1)):
        ex["S-A_minus_S-M"][nm + "_fixed_4_10"] = paired(metr(D, "S-A", R_MAIN)[pi] - metr(D, "S-M", R_MAIN)[pi])
        pa = paper_proto(D, "S-A")[pi] - paper_proto(D, "S-M")[pi]
        ex["S-A_minus_S-M"][nm + "_paper_protocol"] = paired(pa)
    cm = D["S-M"]["auc"][:, R_MAIN, :].mean(axis=(0, 1))
    weak = list(np.argsort(cm)[:2])
    ex["weak_clients"]["idx"] = [int(i) for i in weak]
    ex["weak_clients"]["S-M_mean_auc"] = [float(cm[i]) for i in weak]
    for a, b in (("P-U", "S-U"), ("P-A", "S-A"), ("P-M", "S-M")):
        d = (D[a]["auc"][:, R_MAIN, :].mean(axis=1) - D[b]["auc"][:, R_MAIN, :].mean(axis=1)).mean(axis=0)
        ex["weak_clients"][f"{a}-{b}_delta_per_client"] = [float(x) for x in d]
    return ex


def diverge(D):
    bad = []
    for c in ("P-U", "P-A", "P-M"):
        m = D[c]["auc"].mean(axis=2)[:, 4:]            # 라운드 5~
        if (m < 90.0 - TOL).any(): bad.append(c)
    return bad


def h2a(base):
    L = glob.glob(f"{base}/S1/diag/Run_*.jsonl")
    rhos = []
    for fn in L:
        for l in open(fn):
            d = list(json.loads(l).values())[0]
            la = np.array([np.mean(x) for x in d["local_auc"]])
            if len(la) > 2 and la.std() > 0:
                rhos.append(st.spearmanr(np.arange(len(la)), la).correlation)
    rhos = np.array(rhos)
    return {"n_rounds": len(rhos), "mean_rho": float(np.nanmean(rhos)), "frac_rho_gt_0.3": float(np.mean([gt(r, 0.3) for r in rhos]))} if len(rhos) else {}


def parse_runs(s):
    a, b = s.split("-"); return list(range(int(a), int(b) + 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="H12"); ap.add_argument("--runs", default="0-9"); ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Checkpoint/Results/EXP")
    a = ap.parse_args()
    runs = parse_runs(a.runs)
    D = load(a.base, a.exp, runs)
    out = {"exp": a.exp, "runs": runs, "means": {c: {"M1": float(metr(D, c, R_MAIN)[0].mean()), "M2": float(metr(D, c, R_MAIN)[1].mean())} for c in CELLS},
           "diverged_parallel_cells": diverge(D), "min_parallel_mean_auc_round5plus": float(min(D[c]["auc"].mean(axis=2)[:, 4:].min() for c in ("P-U","P-A","P-M"))), "H1": h1(D), "H1_mechanism": mech(D), "H2": h2(D), "extras": extras(D)}
    if a.exp == "H12": out["H2a"] = h2a(a.base)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"means": out["means"], "H1": {k: (v["verdict"], round(v["mean"], 3), [round(x, 3) for x in v["ci90"]]) if k != "overall" else v for k, v in out["H1"].items()},
                      "mech": out["H1_mechanism"], "H2": {k: v["verdict"] for k, v in out["H2"]["contrasts"].items()},
                      "main": (out["H2"]["main_P-A_vs_S-M"]["verdict_holm2"], out["H2"]["main_P-A_vs_S-M"]["verdict_raw"]),
                      "diverged": out["diverged_parallel_cells"]}, ensure_ascii=False, indent=1))
