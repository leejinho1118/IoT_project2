"""ToN-IoT(호스트별) / CIC-IoT-2023(KMeans 인위 비IID 또는 IID) 클라이언트 로더. main_exp.load_clients와 같은 딕셔너리 구조.
전처리는 이전 탐색(ton_iot/run_ton.py의 'flow' 특징, ciciot23/run_cic.py의 분할)을 따른다."""
import os, glob, numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
HOME = os.path.expanduser('~/Desktop/fedmse_drift')
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache'); os.makedirs(CACHE, exist_ok=True)
DEV_CAP = 300          # 클라이언트 dev 행 수(= n_blocks 10으로 나눠떨어지게 300)
BATCH = 12
CIC_PER_CLIENT = 3000  # CIC 클라이언트당 정상 행 상한(연산량 제한)
CIC_ATT_PER_TYPE = 200

def _ton_features():
    d = pd.read_csv(f'{HOME}/ton_iot/train_test_network.csv'); d = d[d.type != 'mitm'].reset_index(drop=True)
    num = lambda c: pd.to_numeric(d[c], errors='coerce').fillna(-1).to_numpy(dtype=float)
    logn = lambda c: np.log1p(np.clip(num(c), 0, None))[:, None]
    def onehot(c, top=12):
        v = d[c].astype(str); keep = v.value_counts().index[:top]
        return pd.get_dummies(v.where(v.isin(keep), '_other'), prefix=c).to_numpy(dtype=float)
    X = np.hstack([logn(c) for c in ['duration','src_bytes','dst_bytes','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes']]
                  + [onehot('proto'), onehot('service'), onehot('conn_state')])
    return X, d.label.to_numpy(), d.type.to_numpy(), d.src_ip.to_numpy()

def _cic_cache():
    f = f'{CACHE}/cic.npz'
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True); return z['ben'], z['att'], z['atype']
    clean = lambda df: df.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype='float32')
    root = f'{HOME}/ciciot23/raw'
    ben = clean(pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f'{root}/Benign_Final/*.csv'))]))
    ben = ben[np.random.default_rng(1234).permutation(len(ben))[:400000]]
    rng = np.random.default_rng(1234); att, at = [], []
    for dd in sorted(os.listdir(root)):
        if dd == 'Benign_Final': continue
        a = clean(pd.concat([pd.read_csv(p) for p in glob.glob(f'{root}/{dd}/*.csv')]))
        a = a[rng.permutation(len(a))[:CIC_ATT_PER_TYPE]]; att.append(a); at += [dd] * len(a)
    att = np.concatenate(att); np.savez(f, ben=ben, att=att, atype=np.array(at)); return ben, att, np.array(at)

TON_PER_CLIENT = 3750

def _mk(name, x_normal, att, rng, dev, cap):
    """x_normal: 이 클라이언트의 정상 원본 행. 40/10/30/20 분할(train/valid/dev/test), 공격 풀은 공통."""
    x = x_normal[rng.permutation(len(x_normal))][:cap]; n = len(x)
    a, b, c, e = int(.4*n), int(.5*n), int(.8*n), n
    tr, va, dv, te = x[:a], x[a:b], x[b:c], x[c:e]
    sc = StandardScaler().fit(tr)
    ptr, pva = sc.transform(tr).astype(np.float32), sc.transform(va).astype(np.float32)
    tx = np.concatenate([sc.transform(te), sc.transform(att)]).astype(np.float32)
    ty = np.r_[np.zeros(len(te)), np.ones(len(att))]
    from DataLoader import IoTDataset
    mk = lambda p: DataLoader(IoTDataset(p, np.zeros(len(p))), batch_size=BATCH, pin_memory=True)
    return {"device": name, "train_loader": mk(ptr), "valid_loader": mk(pva),
            "dev_normal_dataset": pd.DataFrame(dv[:DEV_CAP]), "train_x": torch.tensor(ptr).to(dev), "test_x": torch.tensor(tx).to(dev), "test_y": ty}

def load(dataset, data_seed, dev):
    rng = np.random.default_rng(data_seed)
    if dataset == 'ton':
        X, y, typ, ip = _ton_features()
        hosts = [h for h, c in pd.Series(ip[y == 0]).value_counts().items() if c >= 1500]
        hosts = [hosts[i] for i in rng.permutation(len(hosts))]
        aid = np.flatnonzero(y == 1); aid = aid[rng.permutation(len(aid))]
        att = X[np.concatenate([aid[typ[aid] == t][:1000] for t in sorted(set(typ[aid]))])]
        cl = [_mk(str(h), X[(y == 0) & (ip == h)], att, rng, dev, TON_PER_CLIENT) for h in hosts]
    elif dataset in ('cic_km', 'cic_iid'):
        ben, att, _ = _cic_cache(); NCL = 10
        if dataset == 'cic_iid': lab = rng.integers(0, NCL, len(ben))
        else:
            sc = StandardScaler().fit(ben[:200000]); z = sc.transform(ben)
            km = KMeans(NCL, n_init=3, random_state=int(data_seed)).fit(z[:200000]); lab = km.predict(z)
        order = rng.permutation(NCL)
        cl = [_mk(f'cic{k}', ben[lab == k][:100000], att, rng, dev, CIC_PER_CLIENT) for k in order]
    else: raise ValueError(dataset)
    return cl, int(cl[0]['train_x'].shape[1])
