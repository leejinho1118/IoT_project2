"""
실험용 집계기 (원본 GlobalAggregator 상속, 원본 불변).
- 궤적용 집계는 원본(fed_avg / fed_mse_avg)을 그대로 호출.
- 진단/반사실용 통계와 일반 가중식(w ∝ n^β · exp(-z/T))만 새로 추가.
- 모든 통계/후보 계산은 스크래치 모델에서 수행해 global model을 건드리지 않음.
"""
import copy
import numpy as np
import torch

from Trainer.global_aggregator import GlobalAggregator

EPS = 1e-8


def standardize(s):
    s = np.asarray(s, dtype=np.float64)
    return (s - np.median(s)) / (np.std(s) + EPS)


def make_weights(n, beta, T, score=None):
    """w ∝ n^beta * exp(-z/T). T=None/inf -> 점수 무시."""
    n = np.asarray(n, dtype=np.float64)
    logw = beta * np.log(n)
    if score is not None and T is not None and np.isfinite(T):
        logw = logw - standardize(score) / T
    logw -= logw.max()
    w = np.exp(logw)
    return w / w.sum()


def b_score(block_mse, lam, clip=2.0, raw=False):
    """block_mse: (k models, G blocks). 낮을수록 좋음."""
    m = np.asarray(block_mse, dtype=np.float64)
    r = np.median(m, axis=0, keepdims=True)
    l = np.log(m / r)
    if not raw:
        l = np.clip(l, -clip, clip)
    return l.mean(axis=1) + lam * l.max(axis=1)


def mix_states(states, w):
    """state_dict 가중합 (float32 유지)."""
    out = {}
    for k in states[0].keys():
        acc = None
        for st, wi in zip(states, w):
            term = st[k].float() * float(wi)
            acc = term if acc is None else acc + term
        out[k] = acc
    return out


class ExpAggregator(GlobalAggregator):
    def __init__(self, model, update_type="mse_avg", n_blocks=10):
        super().__init__(model, update_type=update_type)
        self.n_blocks = n_blocks
        self.scratch = copy.deepcopy(model).to(self.device)

    def create_dev_dataset(self, dataset):
        super().create_dev_dataset(dataset)
        self.dev_t = torch.Tensor(self.dev_dataset).to(self.device)
        self.min_len = len(self.dev_dataset) // self.n_blocks
        assert self.min_len * self.n_blocks == len(self.dev_dataset)

    @torch.no_grad()
    def stats(self, state, boot_rng=None, n_boot=0):
        """한 로컬 모델의 dev 통계 (스크래치 모델)."""
        m = self.scratch
        m.load_state_dict(state)
        m.eval()
        x = self.dev_t
        z, out, loss = m(x)
        per = ((x - out) ** 2).mean(dim=1)               # 샘플별 MSE
        pooled = per.mean().item()
        blocks = per.view(self.n_blocks, self.min_len).mean(dim=1).cpu().numpy()
        res = {"mse": pooled, "block_mse": blocks.astype(np.float64),
               "A1": loss.item()}
        z_np = z.cpu().numpy().astype(np.float64)
        res.update(self._a23(z_np))
        if n_boot:
            bs = {"A1": [], "A2": [], "A3": []}
            per_np = per.cpu().numpy()
            N = len(per_np)
            for _ in range(n_boot):
                idx = boot_rng.randint(0, N, N)
                zb = z_np[idx]
                # A1 재계산: MSE + lambda*mean norm (idx 행 기준)
                xb, ob = x[idx], out[idx]
                lb = ((xb - ob) ** 2).mean().item() + m.shrink_lambda * float(
                    np.linalg.norm(zb, axis=1).mean())
                r = self._a23(zb)
                bs["A1"].append(lb); bs["A2"].append(r["A2"]); bs["A3"].append(r["A3_raw"])
            res["boot"] = {k: float(np.std(v)) for k, v in bs.items()}
        return res

    @staticmethod
    def _a23(z):
        mu, sd = z.mean(0), z.std(0) + EPS
        d = np.linalg.norm((z - mu) / sd, axis=1)
        a2 = np.percentile(d, 95) / (np.percentile(d, 50) + EPS)
        a3 = float(np.linalg.norm(z, axis=1).mean())
        return {"A2": float(a2), "A3_raw": a3}

    @staticmethod
    def a3_normalized(stats_list):
        raw = np.array([s["A3_raw"] for s in stats_list])
        return raw / (np.median(raw) + EPS)

    def scores_all(self, stats_list, lam=0.0):
        """점수 dict (모두 낮을수록 좋음)."""
        bm = np.stack([s["block_mse"] for s in stats_list])
        return {
            "A1": np.array([s["A1"] for s in stats_list]),
            "A2": np.array([s["A2"] for s in stats_list]),
            "A3": self.a3_normalized(stats_list),
            "mse": np.array([s["mse"] for s in stats_list]),
            "B_lam0": b_score(bm, 0.0), "B_lam0.5": b_score(bm, 0.5),
        }

    # ---- 후보 생성 (스크래치 모델의 state_dict 반환) ----
    def candidate_state(self, states, w):
        return mix_states(states, w)

    def mse_avg_weights(self, stats_list):
        a = 1.0 / np.array([s["mse"] for s in stats_list], dtype=np.float64)
        return a / a.sum()
