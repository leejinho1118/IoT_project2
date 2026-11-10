"""0단계 무비용 분석 (PRE_RULES_3.md §1 그대로). 실행: cd src && python ../analysis/analyze_cause0.py --out ../analysis/cause0_result.json"""
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_h1h2 import load, metr, paired, boot_ci, ge, gt, le, lt, TOL, R_MAIN

RUNS = list(range(10))


def curves(a, loss):
    """a: (run, round, client) %p. 라운드별 M1(10클라 평균), M2(최저)의 run 평균·표준편차."""
    m1 = a.mean(axis=2); m2 = a.min(axis=2)
    f = lambda x: {"mean": x.mean(axis=0).tolist(), "sd": x.std(axis=0, ddof=1).tolist()}
    return {"M1": f(m1), "M2": f(m2), "loss": f(loss)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--base", default="Checkpoint/Results/EXP")
    ap.add_argument("--exp", default="H12")
    a = ap.parse_args()
    D = load(a.base, a.exp, RUNS)
    S, P = D["S-U"], D["P-U"]
    out = {"curves": {"S-U": curves(S["auc"], S["loss"]), "P-U": curves(P["auc"], P["loss"])}}

    # 기준선/격차
    sm1, sm2 = metr(D, "S-U", R_MAIN); pm1, pm2 = metr(D, "P-U", R_MAIN)
    R1, R2 = float(sm1.mean()), float(sm2.mean())
    G1 = {"M1": float((sm1 - pm1).mean()), "M2": float((sm2 - pm2).mean())}
    out["ref"] = {"R_M1": R1, "R_M2": R2, "G1": G1, "P_M1": float(pm1.mean()), "P_M2": float(pm2.mean())}

    # 상승 중 판정
    rise = {}
    for nm, fn in (("M1", lambda x: x.mean(axis=2)), ("M2", lambda x: x.min(axis=2))):
        m = fn(P["auc"])                                   # (run, round)
        d = m[:, 17:20].mean(axis=1) - m[:, 10:13].mean(axis=1)
        s = paired(d); s["rising"] = bool(gt(s["ci95"][0], 0) and ge(s["mean"], 0.1))
        rise[nm] = s
    out["rise_P_18_20_minus_11_13"] = rise

    # 도달 라운드
    pr1 = P["auc"].mean(axis=2).mean(axis=0); pr2 = P["auc"].min(axis=2).mean(axis=0)
    first = lambda arr, thr: next((r + 1 for r, v in enumerate(arr) if ge(v, thr)), None)
    out["reach_round"] = {"M1": first(pr1, R1), "M2": first(pr2, R2)}

    # 후반 격차 (라운드 18~20)
    late = slice(17, 20)
    sl1, sl2 = metr(D, "S-U", late); pl1, pl2 = metr(D, "P-U", late)
    gl = {"M1": paired(sl1 - pl1), "M2": paired(sl2 - pl2)}
    out["late_gap_S_minus_P_18_20"] = gl
    out["L"] = {"L_M1": float(pl1.mean()), "L_M2": float(pl2.mean()), "S_late_M1": float(sl1.mean()), "S_late_M2": float(sl2.mean())}

    # 멈춤 조건
    stop_i = bool(ge(pl1.mean(), R1 - 0.3) and ge(pl2.mean(), R2 - 1.0))
    stop_ii = bool(ge(gl["M1"]["ci90"][0], -0.3) and le(gl["M1"]["ci90"][1], 0.3) and
                   ge(gl["M2"]["ci90"][0], -1.0) and le(gl["M2"]["ci90"][1], 1.0))
    out["stop"] = {"i_parallel_reaches_serial_level": stop_i, "ii_late_gap_vanishes": stop_ii, "any": stop_i or stop_ii}

    # 가설 선별
    rising_any = rise["M1"]["rising"] or rise["M2"]["rising"]
    flat_both = (not rise["M1"]["rising"]) and (not rise["M2"]["rising"])
    half = gl["M1"]["mean"] <= 0.5 * G1["M1"] + TOL
    if rising_any and half: sel = "H-A 후보 강화"
    elif flat_both and not half: sel = "H-A 후보 약화"
    else: sel = "어느 쪽도 아님(판정불가)"
    out["screen"] = {"verdict": sel, "rising_any": rising_any, "flat_both": flat_both, "late_gap_M1_le_half_G1": bool(half)}

    # 클라이언트별 Δ
    d = (P["auc"][:, R_MAIN, :].mean(axis=1) - S["auc"][:, R_MAIN, :].mean(axis=1)).mean(axis=0)
    cm = S["auc"][:, R_MAIN, :].mean(axis=(0, 1)); weak = np.argsort(cm)[:2]
    neg = np.minimum(d, 0)
    out["per_client"] = {"delta_P_minus_S": d.tolist(), "S_mean_auc": cm.tolist(), "weak_idx": [int(i) for i in weak],
                         "weak_share_of_total_drop": float(neg[weak].sum() / neg.sum()) if neg.sum() < 0 else None}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    r = lambda x: round(float(x), 3)
    print(json.dumps({"ref": out["ref"], "rise": {k: (r(v["mean"]), [r(x) for x in v["ci95"]], v["rising"]) for k, v in rise.items()},
                      "reach": out["reach_round"], "L": out["L"],
                      "late_gap": {k: (r(v["mean"]), [r(x) for x in v["ci90"]]) for k, v in gl.items()},
                      "stop": out["stop"], "screen": out["screen"], "per_client": out["per_client"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
