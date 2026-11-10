"""1단계 분석 (PRE_RULES_3.md §2~§6 그대로).
실행: cd src && python ../analysis/analyze_cause.py --exp CAUSE --out ../analysis/cause_result.json
스모크: python ../analysis/analyze_cause.py --exp SMOKE3 --smoke   (필드·로깅 함수만 점검, 판정 없음)"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_h1h2 import load, metr, paired, ge, gt, le, lt, TOL, R_MAIN, SEED, NB

DELTA = {"M1": 0.3, "M2": 1.0}
BASE = "Checkpoint/Results/EXP"
N_A, N_B = 3, 3      # 탐색 모드(PRE_RULES_3_ADD1): run 0~2
CELLS = {"n_S_log": range(N_B), "n_P1_log": range(N_B), "n_P2": range(N_A), "n_P5": range(N_A),
         "i_S": range(N_B), "i_P1": range(N_B)}


def read_cell(exp, cell, runs, rounds=20):
    recs, diverged = {}, os.path.exists(f"{BASE}/{exp}/{cell}/DIVERGED")
    for r in runs:
        fn = f"{BASE}/{exp}/{cell}/Run_{r}.jsonl"
        if not os.path.exists(fn): continue
        L = [list(json.loads(l).values())[0] for l in open(fn)]
        if len(L) == rounds or (diverged and L and L[-1].get("diverged")): recs[r] = L
    return recs, diverged


def arr_auc(recs, runs):
    return np.array([[x["global_auc"] for x in recs[r]] for r in runs]) * 100


def diag(recs, runs, key, sl):
    """run별 라운드 구간 평균. key: global_disp / pair_mean / local_disp(평균) / epochs / steps."""
    out = []
    for r in runs:
        v = []
        for x in recs[r][sl]:
            d = x["diag2"]
            v.append(np.mean(d[key]) if isinstance(d[key], list) else d[key])
        out.append(np.mean(v))
    return np.array(out)


def rel_drift(recs, runs, sl):
    out = []
    for r in runs:
        out.append(np.mean([x["diag2"]["pair_mean"] / np.mean(x["diag2"]["local_disp"]) for x in recs[r][sl]]))
    return np.array(out)


def boot_diff(a, b, level=0.95):
    rs = np.random.RandomState(SEED)
    ia = rs.randint(0, len(a), (NB, len(a))); ib = rs.randint(0, len(b), (NB, len(b)))
    bd = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    q = (1 - level) / 2 * 100
    return float(np.percentile(bd, q)), float(np.percentile(bd, 100 - q))


def metric_pair(a):
    return a[:, R_MAIN, :].mean(axis=2).mean(axis=1), a[:, R_MAIN, :].min(axis=2).mean(axis=1)


def eta_status(d, G1, Geta, dl):
    """지표 상태. d: P_eta − S (run별 %p)."""
    s = paired(d)
    closed = (ge(s["ci90"][0], -dl) and le(s["ci90"][1], dl)) or (gt(s["ci95"][0], 0) and ge(s["mean"], dl))
    if closed: st = "닫힘"
    elif G1 > 0 and le(Geta, 0.5 * G1): st = "부분 닫힘"
    else: st = "안 닫힘"
    s["status"] = st
    return s


def smoke(exp):
    print("스모크 점검:", exp)
    for cell in CELLS:
        recs, dv = read_cell(exp, cell, [0], rounds=3)
        assert 0 in recs and len(recs[0]) in (3,), (cell, "records missing or wrong count")
        x = recs[0][0]
        assert {"global_auc", "global_loss", "join_clients", "diag2", "eta"} <= set(x), x.keys()
        d = x["diag2"]
        assert len(d["epochs"]) == len(d["steps"]) == len(d["local_disp"]) == 5 and "global_disp" in d and "pair_mean" in d
        sl = slice(0, 3)
        print(cell, "eta", x["eta"], "epochs(mean)", round(float(diag(recs, [0], "epochs", sl)[0]), 2),
              "steps", round(float(diag(recs, [0], "steps", sl)[0]), 1), "local_disp", round(float(diag(recs, [0], "local_disp", sl)[0]), 3),
              "global_disp", round(float(diag(recs, [0], "global_disp", sl)[0]), 3), "rd", round(float(rel_drift(recs, [0], sl)[0]), 3),
              "meanAUC_r3", round(float(np.mean(recs[0][-1]["global_auc"])), 4), "DIVERGED" if dv else "")
    print("스모크 구조 점검 OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="CAUSE"); ap.add_argument("--out"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plots", default="")
    a = ap.parse_args()
    if a.smoke:
        return smoke(a.exp)

    H = load(BASE, "H12", list(range(10)))
    SU, PU = H["S-U"], H["P-U"]            # auc (run, round, client) %p
    out = {}
    data, cells_div = {}, {}
    for c, runs in CELLS.items():
        recs, dv = read_cell(a.exp, c, runs); data[c] = recs; cells_div[c] = dv
    out["diverged_cells"] = [c for c, v in cells_div.items() if v]

    # --- 일치 검사 (멈춤 조건) ---
    chk = {}
    for cell, ref in (("n_S_log", SU), ("n_P1_log", PU)):
        runs = sorted(data[cell]); assert runs == list(range(N_B)), (cell, runs)
        new = arr_auc(data[cell], runs)
        chk[cell] = float(np.abs(new - ref["auc"][runs]).max())
    chk["join_same"] = all(
        data[c][r][k]["join_clients"] == [list(json.loads(l).values())[0] for l in open(
            f"{BASE}/H12/{'final_uniform' if c=='n_S_log' else 'final_uniform_parallel'}/Run_{r}.jsonl")][k]["join_clients"]
        for c in ("n_S_log", "n_P1_log") for r in range(N_B) for k in range(20))
    chk["pass"] = bool(chk["n_S_log"] <= 1e-9 and chk["n_P1_log"] <= 1e-9 and chk["join_same"])
    out["match_check"] = chk
    if not chk["pass"]:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1, default=float); sys.exit("일치 검사 실패: 멈춤 조건")

    # --- G(1) ---
    s1, s2 = metric_pair(SU["auc"][:N_A]); p1, p2 = metric_pair(PU["auc"][:N_A])
    G1 = {"M1": float((s1 - p1).mean()), "M2": float((s2 - p2).mean())}
    out["G1"] = G1

    # --- 이동거리 (라운드 4~10) ---
    ref_recs = data["n_S_log"]; rr = list(range(N_B))
    disp = {"S": float(diag(ref_recs, rr, "global_disp", R_MAIN).mean()),
            "P1": float(diag(data["n_P1_log"], rr, "global_disp", R_MAIN).mean())}
    out["global_disp_r4_10"] = disp

    # --- H-A ---
    ha = {"cells": {}}
    for eta, cell in ((2, "n_P2"), ("5", "n_P5")):
        recs = data[cell]; runs10 = list(range(N_A))
        info = {"diverged": cells_div[cell], "n_runs_done": len(recs)}
        if recs:
            gd = float(diag(recs, sorted(recs), "global_disp", slice(3, 10)).mean()) if all(len(recs[r]) >= 10 for r in recs) else None
            info["global_disp_r4_10"] = gd
            info["disp_ratio_vs_serial"] = gd / disp["S"] if gd is not None else None
            info["disp_match"] = bool(info["disp_ratio_vs_serial"] is not None and ge(info["disp_ratio_vs_serial"], 0.67) and le(info["disp_ratio_vs_serial"], 1.5)) and not cells_div[cell]
        if cells_div[cell] or len(recs) < N_A or any(len(recs[r]) < 20 for r in recs):
            info["cell_status"] = "안 닫힘(발산)" if cells_div[cell] else "미완료"
            info["disp_match"] = False
        else:
            P = arr_auc(recs, runs10)
            q1, q2 = metric_pair(P)
            d1, d2 = q1 - s1, q2 - s2
            G = {"M1": float((s1 - q1).mean()), "M2": float((s2 - q2).mean())}
            m1 = eta_status(d1, G1["M1"], G["M1"], DELTA["M1"]); m2 = eta_status(d2, G1["M2"], G["M2"], DELTA["M2"])
            sts = (m1["status"], m2["status"])
            cs = "닫힘" if sts == ("닫힘", "닫힘") else ("부분 닫힘" if all(s in ("닫힘", "부분 닫힘") for s in sts) else "안 닫힘")
            info.update({"G": G, "M1": m1, "M2": m2, "cell_status": cs,
                         "gap_reduction": {"M1": 1 - G["M1"] / G1["M1"], "M2": 1 - G["M2"] / G1["M2"]},
                         "means": {"M1": float(q1.mean()), "M2": float(q2.mean())},
                         "late_11_20": {"M1": float(P[:, 10:, :].mean(axis=2).mean()), "M2": float(P[:, 10:, :].min(axis=2).mean())}})
        ha["cells"][str(eta)] = info
    cs = ha["cells"]
    support = any(i["cell_status"] in ("닫힘", "부분 닫힘") and i.get("disp_match") for i in cs.values())
    nondiv = [i for i in cs.values() if not i["diverged"] and i["cell_status"] != "미완료"]
    reject = (len(nondiv) > 0 and all(i["cell_status"] == "안 닫힘" for i in nondiv) and any(i.get("disp_match") for i in nondiv))
    ha["verdict"] = "H-A 지지" if support else ("H-A 기각" if reject else "H-A 판정불가")
    out["H-A"] = ha

    # --- H-B ---
    rr5 = list(range(N_B)); hb = {}
    nS, nP = SU["auc"][rr5], PU["auc"][rr5]
    n1, n2 = metric_pair(nS); np1, np2 = metric_pair(nP)
    Gn = {"M1": float((n1 - np1).mean()), "M2": float((n2 - np2).mean())}
    iS, iP = arr_auc(data["i_S"], rr5), arr_auc(data["i_P1"], rr5)
    i1, i2 = metric_pair(iS); ip1, ip2 = metric_pair(iP)
    Gi = {"M1": float((i1 - ip1).mean()), "M2": float((i2 - ip2).mean())}
    ceiling = bool(ge(i1.mean(), 99.5) and ge(ip1.mean(), 99.5))
    hb.update({"G_noniid_same_runs": Gn, "G_iid": Gi, "iid_S_M1": float(i1.mean()), "iid_P1_M1": float(ip1.mean()), "ceiling": ceiling})
    small, big, undefined = {}, {}, False
    for nm, ii, dl in (("M1", i1 - ip1, 0.3), ("M2", i2 - ip2, 1.0)):
        if Gn[nm] < dl - TOL: undefined = True; small[nm] = None; big[nm] = None; continue
        s = paired(ii)
        small[nm] = bool(lt(Gi[nm], 0.5 * Gn[nm]) or (ge(s["ci90"][0], -dl) and le(s["ci90"][1], dl)))
        big[nm] = bool(ge(Gi[nm], 0.5 * Gn[nm]))
        hb[f"iid_gap_{nm}"] = s
    rd_n = rel_drift(data["n_P1_log"], rr5, R_MAIN); rd_i = rel_drift(data["i_P1"], rr5, R_MAIN)
    lo, hi = boot_diff(rd_n, rd_i)
    hb["rel_drift"] = {"noniid": rd_n.tolist(), "iid": rd_i.tolist(), "diff_mean": float(rd_n.mean() - rd_i.mean()), "ci95": [lo, hi], "significant": bool(gt(lo, 0))}
    hb["pair_mean_raw"] = {"noniid": float(diag(data["n_P1_log"], rr5, "pair_mean", R_MAIN).mean()), "iid": float(diag(data["i_P1"], rr5, "pair_mean", R_MAIN).mean())}
    hb["local_disp_raw"] = {"noniid": float(diag(data["n_P1_log"], rr5, "local_disp", R_MAIN).mean()), "iid": float(diag(data["i_P1"], rr5, "local_disp", R_MAIN).mean())}
    hb["small"], hb["big"] = small, big
    if ceiling or undefined: hv = "H-B 판정불가"
    elif small["M1"] and small["M2"] and hb["rel_drift"]["significant"]: hv = "H-B 지지"
    elif big["M1"] or big["M2"]: hv = "H-B 기각(주요 원인 아님)"
    else: hv = "H-B 판정불가"
    hb["verdict"] = hv + (" [천장]" if ceiling else "") + (" [비-IID 격차<δ]" if undefined else "")
    if gt(Gn["M1"], 0) and Gn["M1"] != 0: hb["ratio_G_iid_over_noniid"] = {k: Gi[k] / Gn[k] for k in Gn}
    out["H-B"] = hb

    # --- H-C ---
    hc = {}
    for key in ("epochs", "steps"):
        s_ = float(diag(data["n_S_log"], rr, key, R_MAIN).mean()); p_ = float(diag(data["n_P1_log"], rr, key, R_MAIN).mean())
        s20 = float(diag(data["n_S_log"], rr, key, slice(0, 20)).mean()); p20 = float(diag(data["n_P1_log"], rr, key, slice(0, 20)).mean())
        hc[key] = {"serial_r4_10": s_, "parallel_r4_10": p_, "ratio_max_over_min": max(s_, p_) / min(s_, p_),
                   "serial_r1_20": s20, "parallel_r1_20": p20}
    hc["verdict"] = "H-C 지지" if any(ge(hc[k]["ratio_max_over_min"], 1.5) for k in ("epochs", "steps")) else "H-C 약함"
    # 클라이언트별 에폭(기술통계): 디바이스 인덱스별 평균
    per = {}
    for c in ("n_S_log", "n_P1_log"):
        acc = {i: [] for i in range(10)}
        for r in rr:
            for x in data[c][r][3:10]:
                for ci, e in zip(x["join_clients"], x["diag2"]["epochs"]): acc[ci].append(e)
        per[c] = {i: (float(np.mean(v)) if v else None) for i, v in acc.items()}
    hc["per_client_epochs"] = per
    out["H-C"] = hc

    # 라운드별 평균 곡선/이동거리 (그림·민감도)
    curves = {}
    for nm, A in (("S-U", SU["auc"][:N_A]), ("P-U", PU["auc"][:N_A])):
        curves[nm] = {"M1": A.mean(axis=2).mean(axis=0).tolist(), "M2": A.min(axis=2).mean(axis=0).tolist()}
    for c in ("n_P2", "n_P5"):
        if data[c] and all(len(data[c][r]) == 20 for r in data[c]):
            A = arr_auc(data[c], sorted(data[c])); curves[c] = {"M1": A.mean(axis=2).mean(axis=0).tolist(), "M2": A.min(axis=2).mean(axis=0).tolist()}
    out["curves"] = curves
    out["disp_curves"] = {c: np.mean([[x["diag2"]["global_disp"] for x in data[c][r]] for r in data[c]], axis=0).tolist()
                          for c in ("n_S_log", "n_P1_log", "n_P2", "n_P5", "i_S", "i_P1") if data[c] and all(len(data[c][r]) == 20 for r in data[c])}
    out["mode"] = "탐색(PRE_RULES_3_ADD1): n=3, 확정 판정 아님"
    out["verdicts"] = {"H-A": "탐색: " + ha["verdict"], "H-B": "탐색: " + hb["verdict"], "H-C": "탐색: " + hc["verdict"]}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({"match_check": chk, "G1": G1, "disp": disp, "H-A": {k: {kk: vv for kk, vv in v.items() if kk in ("cell_status", "G", "disp_ratio_vs_serial", "disp_match", "gap_reduction", "means", "diverged")} for k, v in cs.items()},
                      "H-A_verdict": ha["verdict"], "H-B": {k: v for k, v in hb.items() if k in ("G_noniid_same_runs", "G_iid", "rel_drift", "ceiling", "small", "big", "verdict", "pair_mean_raw", "local_disp_raw", "iid_S_M1", "iid_P1_M1")},
                      "H-C": {k: v for k, v in hc.items() if k != "per_client_epochs"}}, ensure_ascii=False, indent=1, default=float))
    if a.plots:
        plot(out, a.plots)


def plot(out, d):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(d, exist_ok=True)
    x = np.arange(1, 21)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for nm, c in out["curves"].items():
        ax[0].plot(x, c["M1"], label=nm); ax[1].plot(x, c["M2"], label=nm)
    ax[0].set_title("M1 (mean AUC of 10 clients)"); ax[1].set_title("M2 (min AUC)")
    for a_ in ax: a_.set_xlabel("round"); a_.legend(); a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{d}/rounds_curves.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    etas = ["1", "2", "5"]; g1 = out["G1"]
    for j, m in enumerate(("M1", "M2")):
        vals = [g1[m]] + [out["H-A"]["cells"][e].get("G", {}).get(m, np.nan) for e in ("2", "5")]
        ax.bar(np.arange(3) + j * 0.38, vals, 0.38, label=m)
    ax.set_xticks(np.arange(3) + 0.19); ax.set_xticklabels([f"eta={e}" for e in etas]); ax.set_ylabel("gap S-U minus P-U (pp)"); ax.legend(); ax.axhline(0, c="k", lw=.5)
    fig.tight_layout(); fig.savefig(f"{d}/gap_by_eta.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, v in out["disp_curves"].items(): ax.plot(x, v, label=c)
    ax.set_xlabel("round"); ax.set_ylabel("||W_global_after - W_start||"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{d}/global_disp.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    hb = out["H-B"]["rel_drift"]
    ax.boxplot([hb["noniid"], hb["iid"]], tick_labels=["non-IID P1", "IID P1"]); ax.set_ylabel("pairwise / local displacement")
    fig.tight_layout(); fig.savefig(f"{d}/rel_drift.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
