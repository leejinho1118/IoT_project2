"""mse_fx 단위시험. 실행: cd src && python -m pytest ../agg_variants_2026-11-21/confirm/tests_fedexp.py -q"""
import os, sys, copy
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import run_confirm as R


class Net(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.l = torch.nn.Linear(6, 6)
    def forward(s, x):
        g = s.l(x); return None, g, None


class FakeAgg:
    def __init__(s):
        s.model = Net(); s.dev_dataset = np.random.RandomState(0).randn(20, 6).astype("float32"); s.device = "cpu"


def mk(dim_scale=1.0, M=4, seed=0, deltas=None):
    torch.manual_seed(seed)
    agg = FakeAgg(); start = copy.deepcopy(agg.model.state_dict())
    keys = list(start.keys()); flat = R.state_flat(start, keys)
    rs = np.random.RandomState(seed)
    if deltas is None:
        deltas = [torch.tensor(rs.randn(flat.numel()), dtype=torch.float32) * 0.05 * dim_scale for _ in range(M)]
    states = []
    for d in deltas:
        v = flat + d; st = {}; o = 0
        for k in keys:
            nn_ = start[k].numel(); st[k] = v[o:o + nn_].reshape(start[k].shape).clone(); o += nn_
        states.append(st)
    return agg, start, states, keys


def ref_eta(deltas, q, eps):
    D = np.array([d.double().numpy() for d in deltas]); q = np.asarray(q, dtype=np.float64)
    dbar = (q[:, None] * D).sum(0)  # weighted mean
    return max(1.0, (q * (D ** 2).sum(1)).sum() / (2 * ((dbar ** 2).sum() + eps)))


def run(agg, start, states, eps=1e-3):
    return R.combine_variant(agg, start, states, [1] * len(states), "mse_fx", None, fx_eps=eps)[0]


def test_eta_matches_numpy_ref_uniform_and_weighted():
    rs = np.random.RandomState(1)
    for M in (4, 5):
        D = [torch.tensor(rs.randn(50)) for _ in range(M)]
        q = np.ones(M) / M
        dn2 = [float((d ** 2).sum()) for d in D]; db2 = float(((sum(D) / M) ** 2).sum())
        assert abs(R.fedexp_eta(dn2, q, db2, 1e-3)[0] - ref_eta(D, q, 1e-3)) < 1e-9
        paper = max(1.0, sum(dn2) / (2 * M * (db2 + 1e-3)))
        assert abs(R.fedexp_eta(dn2, q, db2, 1e-3)[0] - paper) < 1e-9
        qw = rs.rand(M); qw /= qw.sum()
        db2w = float(((sum(float(a) * d for a, d in zip(qw, D))) ** 2).sum())
        assert abs(R.fedexp_eta(dn2, qw, db2w, 1e-3)[0] - ref_eta(D, qw, 1e-3)) < 1e-9


def test_combine_logged_eta_matches_ref_with_weights():
    # 큰 Δ를 서로 직교에 가깝게 -> eta>1
    agg, start, states, keys = mk(M=4, seed=3)
    new = run(agg, start, states)
    deltas = [R.state_flat(st, keys) - R.state_flat(start, keys) for st in states]
    assert abs(sum(R.MSE_Q) - 1) < 1e-6
    assert abs(R.FX["eta_g"] - ref_eta(deltas, R.MSE_Q, 1e-3)) < 1e-4 * max(1, R.FX["eta_g"])
    assert R.FX["eta_g"] > 1.0
    for key in ("eta_g", "sum_q_dnorm2", "dbar_norm2"): assert key in R.FX


def test_eta_below_one_equals_mse_bitwise():
    agg, start, states, keys = mk(M=4, seed=4)
    d0 = R.combine_variant(agg, start, states, [1] * 4, "mse", None)[0]
    new = run(agg, start, states, eps=1e9)  # eps huge -> eta<1 -> clipped
    assert R.FX["eta_g"] == 1.0
    assert all(torch.equal(new[k], d0[k]) for k in d0)


def test_identical_deltas_eta_one():
    rs = np.random.RandomState(5)
    agg, start, _, keys = mk(M=1)
    d = torch.tensor(rs.randn(R.state_flat(start, keys).numel()), dtype=torch.float32) * 0.05
    agg, start, states, keys = mk(deltas=[d] * 4)
    run(agg, start, states, eps=1e-12)
    assert R.FX["eta_g"] == 1.0
    assert abs(R.FX["sum_q_dnorm2"] / (2 * R.FX["dbar_norm2"]) - 0.5) < 1e-3


def test_orthogonal_equal_norm_eta_half_M():
    n = R.state_flat(FakeAgg().model.state_dict(), list(FakeAgg().model.state_dict().keys())).numel()
    deltas = []
    for i in range(4):
        v = torch.zeros(n); v[i] = 0.5; deltas.append(v)   # orthogonal, equal norm
    agg, start, states, keys = mk(deltas=deltas)
    # 균등 q를 강제해 식 검증(정확한 MSE 가중은 위 시험에서 확인)
    dn2 = [float(((R.state_flat(st, keys) - R.state_flat(start, keys)) ** 2).sum()) for st in states]
    db2 = float(((sum(R.state_flat(st, keys) - R.state_flat(start, keys) for st in states) / 4) ** 2).sum())
    eta = R.fedexp_eta(dn2, [0.25] * 4, db2, 1e-3)[0]
    assert abs(eta - 2.0 * 0.0625 / (0.0625 + 1e-3) * 1.0) < 1e-6 and abs(eta - 2.0) < 0.05, eta
    run(agg, start, states)  # MSE 가중이어도 상한은 M/2 이하, 1 이상
    assert 1.0 <= R.FX["eta_g"] <= 2.0 + 1e-6


def test_new_minus_start_is_eta_times_dbar():
    agg, start, states, keys = mk(M=4, seed=7)
    d0 = R.combine_variant(agg, start, states, [1] * 4, "mse", None)[0]
    new = run(agg, start, states)
    eta = R.FX["eta_g"]; assert eta > 1
    lhs = R.state_flat(new, keys) - R.state_flat(start, keys)
    rhs = eta * (R.state_flat(d0, keys) - R.state_flat(start, keys))
    assert torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-4)
