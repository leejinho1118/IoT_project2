"""mse_nag 단위시험. 실행: cd src && python -m pytest ../agg_variants_2026-11-21/confirm/tests_nag.py -v"""
import os, sys, copy
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import run_confirm as R
from tests_fedexp import mk


def unflat(start, keys, v):
    out, o = {}, 0
    for k in keys:
        n = start[k].numel(); out[k] = v[o:o + n].reshape(start[k].shape).clone(); o += n
    return out


def step(agg, W, keys, rs, vel, variant="mse_nag", M=4, scale=0.05):
    """W: state dict. 임의 클라이언트 상태를 만들고 (new, vel', Delta_flat) 반환."""
    flat = R.state_flat(W, keys)
    states = [unflat(W, keys, flat + torch.tensor(rs.randn(flat.numel()), dtype=torch.float32) * scale) for _ in range(M)]
    avg = R.combine_variant(agg, W, states, [1] * M, "mse", None)[0]
    delta = R.state_flat(avg, keys) - flat
    new, v2 = R.combine_variant(agg, W, states, [1] * M, variant, vel)
    return new, v2, delta, flat


def test_matches_numpy_reference():
    agg, W, _, keys = mk(M=1, seed=1); rs = np.random.RandomState(7)
    vel = None; d = 0.0
    for t in range(6):
        new, vel, delta, flat = step(agg, W, keys, rs, vel)
        D = delta.double().numpy()
        d = 0.9 * d + 0.1 * D; h = 0.1 * D + 0.9 * d
        ref = flat.double().numpy() + h
        assert np.allclose(R.state_flat(new, keys).double().numpy(), ref, rtol=1e-5, atol=1e-6)
        W = new


def test_first_step_is_019_delta():
    agg, W, _, keys = mk(M=1, seed=2); rs = np.random.RandomState(3)
    new, vel, delta, flat = step(agg, W, keys, rs, None)
    assert torch.allclose(R.state_flat(new, keys) - flat, 0.19 * delta, rtol=1e-4, atol=1e-7)
    assert torch.allclose(R.state_flat(vel, keys), 0.1 * delta, rtol=1e-4, atol=1e-7)


def test_constant_delta_steady_state():
    # 순수 재귀: Delta 상수 -> h -> Delta, 단계 크기가 Delta를 넘지 않음
    D = np.ones(5) * 2.0; d = 0.0; hs = []
    for _ in range(200):
        d = 0.9 * d + 0.1 * D; h = 0.1 * D + 0.9 * d; hs.append(h)
    assert all(np.all(h <= D * 1.0 + 1e-12) for h in hs)
    assert np.allclose(hs[-1], D, rtol=1e-6)
    # 실제 코드 경로: 같은 Delta를 매번 주입(W+Delta 상태 4개 동일)
    agg, W, _, keys = mk(M=1, seed=4)
    rs = np.random.RandomState(0); flat0 = R.state_flat(W, keys)
    dl = torch.tensor(rs.randn(flat0.numel()), dtype=torch.float32) * 0.01
    vel = None
    for t in range(150):
        flat = R.state_flat(W, keys)
        st = [unflat(W, keys, flat + dl) for _ in range(3)]
        new, vel = R.combine_variant(agg, W, st, [1] * 3, "mse_nag", vel)
        h = R.state_flat(new, keys) - flat
        assert float(h.norm()) <= float(dl.norm()) * (1 + 1e-4)
        W = new
    assert torch.allclose(h, dl, rtol=1e-2, atol=1e-5)


def test_other_variants_unchanged():
    agg, W, _, keys = mk(M=1, seed=5); rs = np.random.RandomState(9)
    for variant, f in (("mse_dm", lambda p, D: (0.9 * p + 0.1 * D)), ("mse_m", lambda p, D: (0.9 * p + D))):
        vel = None; p = 0.0; Wc = W
        for t in range(4):
            new, vel, delta, flat = step(agg, Wc, keys, rs, vel, variant)
            D = delta.double().numpy(); p = f(p, D)
            assert np.allclose(R.state_flat(new, keys).double().numpy(), flat.double().numpy() + p, rtol=1e-5, atol=1e-6)
            Wc = new
