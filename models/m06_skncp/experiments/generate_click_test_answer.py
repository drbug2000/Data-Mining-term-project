"""
generate_click_test_answer.py — produce click_test_answer.csv for submission.

Pipeline mirrors task1_skncp_boosted.py exactly, with two changes:
  1. Scores the click_test_query rows instead of click_validation rows.
  2. Writes click_test_answer.csv instead of validation metrics.

Weights, K, alpha, threshold — all identical to the submitted model:
  weights: logHist=1.0, ad_ctr=1.0, ip_ctr=1.0, dev_ctr=2.5, cat_ctr=1.0,
           IPS_pos=1.0, rank=1.0, SKNCP=3.0, content_score=1.0
  K=200, alpha=0.2, threshold=train prevalence (1.11%)

Content head: N_SEEDS=30 seeds retrained here (same arch/HPs as exp_content_strong.py).
The z-score normalization uses internal-val content / log-linear stats,
so the frozen cache MD5 does NOT need to match.

Run:
    python -X utf8 models/m06_skncp/experiments/generate_click_test_answer.py
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.m04_gated.gated_ctr import _best_f1_topk, _binary_auc, _l2_normalize
from models.m06_skncp.experiments.task1_skncp_model6 import (
    DATASET_DIR,
    _build_skncp_index,
    _ctr_series,
    _lookup,
    _rank_feature,
    _safe_log,
    _skncp_scores,
)
from shared.data.dataset import RecoDataset

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS = 30
K_SKNN = 200
BLEND_ALPHA = 0.2
KS = 20
LOGIN_BOOST = np.log(1.7)

WEIGHTS = {
    "logHist": 1.0,
    "ad_ctr": 1.0,
    "ip_ctr": 1.0,
    "dev_ctr": 2.5,
    "cat_ctr": 1.0,
    "IPS_pos": 1.0,
    "rank": 1.0,
    "SKNCP": 3.0,
    "content_score": 1.0,
}
W = np.array(list(WEIGHTS.values()), dtype=np.float64)

OUT_CSV = ROOT / "click_test_answer.csv"


# ---------------------------------------------------------------------------
# Content head (identical arch to exp_content_strong.py)
# ---------------------------------------------------------------------------

class Head(nn.Module):
    def __init__(self, d=384, h=256, p=128):
        super().__init__()
        self.sq = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, p))
        self.aq = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, p))
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, q, a):
        qp = nn.functional.normalize(self.sq(q), dim=1)
        ap = nn.functional.normalize(self.aq(a), dim=1)
        return self.scale * (qp * ap).sum(1) + self.b


def _train_head(qt, at, yt, qv, av, Yva, seed, epochs=55):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = Head().to(DEV)
    pw = torch.tensor(
        (yt.cpu().numpy() == 0).sum() / max(1, (yt.cpu().numpy() == 1).sum()),
        device=DEV,
    )
    lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
    n = len(yt)
    bs = 8192
    best = -1.0
    state = None
    bad = 0
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i: i + bs]
            opt.zero_grad()
            lf(m(qt[idx], at[idx]), yt[idx]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            lv = m(qv, av).cpu().numpy()
        au = _binary_auc(lv, Yva)
        if au > best:
            best = au
            state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 8:
                break
    m.load_state_dict(state)
    return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _z_params(x):
    return float(x.mean()), float(x.std() + 1e-9)


def _z(x, mean, sd):
    return (x - mean) / sd


def _build_feature_matrix(rows, ctr_maps, pos_ctr, global_ctr, skncp, content_score):
    ad_ctr = _lookup(rows["ad"], ctr_maps["ad"], global_ctr)
    ip_ctr = _lookup(rows["ip"], ctr_maps["ip"], global_ctr)
    dev_ctr = _lookup(rows["dev"], ctr_maps["dev"], global_ctr)
    cat_ctr = _lookup(rows["cat"], ctr_maps["cat"], global_ctr)
    pos_rate = _lookup(rows["pos"], pos_ctr, global_ctr)
    rank = _rank_feature(rows["sid"], rows["hist"])
    return np.column_stack([
        _safe_log(rows["hist"]),
        _safe_log(ad_ctr),
        _safe_log(ip_ctr),
        _safe_log(dev_ctr),
        _safe_log(cat_ctr),
        _safe_log(pos_rate) - np.log(global_ctr),
        rank,
        skncp,
        content_score,
    ]).astype(np.float64)


def _collect_train(ds, sid2ip, uid2dev):
    q, a, y = [], [], []
    ad, cat, uid, sid, hist, logged, pos = [], [], [], [], [], [], []
    for ev in ds.training_stream():
        for rec in ev.ads:
            q.append(ev.search_emb)
            a.append(rec.ad_emb)
            y.append(int(rec.is_click))
            ad.append(int(rec.ad_id))
            cat.append(int(rec.category_id))
            uid.append(int(ev.user_id))
            sid.append(int(ev.search_id))
            hist.append(float(rec.hist_ctr) if rec.hist_ctr is not None else 0.0)
            logged.append(float(ev.is_logged_on))
            pos.append(int(rec.position))
    out = {
        "Q": _l2_normalize(np.asarray(q, np.float32)),
        "A": _l2_normalize(np.asarray(a, np.float32)),
        "Y": np.asarray(y, np.int8),
        "ad": np.asarray(ad),
        "cat": np.asarray(cat),
        "uid": np.asarray(uid),
        "sid": np.asarray(sid),
        "hist": np.asarray(hist, np.float64),
        "logged": np.asarray(logged, np.float64),
        "pos": np.asarray(pos, np.float64),
    }
    out["ip"] = np.asarray([sid2ip.get(int(s), -1) for s in out["sid"]])
    out["dev"] = np.asarray([uid2dev.get(int(u), -1) for u in out["uid"]])
    return out


def _collect_test(ds, sid2ip, uid2dev):
    pairs = ds.test_click_queries()
    out = {
        "Q": _l2_normalize(np.asarray([ev.search_emb for ev, _ in pairs], np.float32)),
        "A": _l2_normalize(np.asarray([rec.ad_emb for _, rec in pairs], np.float32)),
        "ad": np.asarray([int(rec.ad_id) for _, rec in pairs]),
        "cat": np.asarray([int(rec.category_id) for _, rec in pairs]),
        "uid": np.asarray([int(ev.user_id) for ev, _ in pairs]),
        "sid": np.asarray([int(ev.search_id) for ev, _ in pairs]),
        "hist": np.asarray([float(rec.hist_ctr) for _, rec in pairs], np.float64),
        "logged": np.asarray([float(ev.is_logged_on) for ev, _ in pairs], np.float64),
        "pos": np.asarray([int(rec.position) for _, rec in pairs], np.float64),
        "SearchID": np.asarray([int(ev.search_id) for ev, _ in pairs]),
        "AdID": np.asarray([int(rec.ad_id) for _, rec in pairs]),
    }
    out["ip"] = np.asarray([sid2ip.get(int(s), -1) for s in out["sid"]])
    out["dev"] = np.asarray([uid2dev.get(int(u), -1) for u in out["uid"]])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"Device: {DEV}")

    ds = RecoDataset(DATASET_DIR).load()
    si = pd.read_csv(ds.dir / "searchinfo.csv")
    ui = pd.read_csv(ds.dir / "userinfo.csv")
    sid2ip = dict(zip(si["SearchID"], si["IPID"]))
    uid2dev = dict(zip(ui["UserID"], ui["UserDeviceID"]))

    print("Loading training data...")
    train = _collect_train(ds, sid2ip, uid2dev)
    print("Loading test data...")
    test = _collect_test(ds, sid2ip, uid2dev)

    y = train["Y"]
    global_ctr = float(y.mean())
    print(f"train_rows={len(y):,}  global_ctr={global_ctr:.6f}  test_rows={len(test['sid']):,}")

    # Internal 80/20 split (for z-score normalization params)
    unique_sids = np.sort(np.unique(train["sid"]))
    val_sids = set(unique_sids[int(len(unique_sids) * 0.8):].tolist())
    mask_val = np.asarray([sid in val_sids for sid in train["sid"]], dtype=bool)
    mask_train = ~mask_val
    y_val = y[mask_val]

    # CTR maps
    keys = ("ad", "ip", "dev", "cat")
    ctr_train = {k: _ctr_series(train[k][mask_train], y[mask_train], global_ctr) for k in keys}
    ctr_full = {k: _ctr_series(train[k], y, global_ctr) for k in keys}
    pos_train = _ctr_series(train["pos"][mask_train], y[mask_train], global_ctr)
    pos_full = _ctr_series(train["pos"], y, global_ctr)

    # SKNCP index (full training data → for test scoring)
    print("Building SKNCP index (full train)...")
    index_full = _build_skncp_index(train, np.ones(len(y), dtype=bool), ctr_full["ad"], global_ctr)
    # Also need internal-train index for z-score normalization (val scores)
    index_train = _build_skncp_index(train, mask_train, ctr_train["ad"], global_ctr)

    print(f"full_clicked_index={index_full['n_clicks']:,}")
    print(f"Computing SKNCP for internal-val (z-score norm)...")
    val_rows = {k: v[mask_val] if isinstance(v, np.ndarray) and len(v) == len(y) else v
                for k, v in train.items()}
    skncp_val, _ = _skncp_scores(train["Q"][mask_val], train["A"][mask_val], index_train, K_SKNN)
    print(f"Computing SKNCP for test (K={K_SKNN})...")
    skncp_test, _ = _skncp_scores(test["Q"], test["A"], index_full, K_SKNN)

    # Content head: train N_SEEDS, accumulate logits for val + test
    print(f"Training content head ({N_SEEDS} seeds)...")
    qt = torch.tensor(train["Q"][mask_train], device=DEV)
    at = torch.tensor(train["A"][mask_train], device=DEV)
    yt = torch.tensor(y[mask_train].astype(np.float32), device=DEV)
    qv = torch.tensor(train["Q"][mask_val], device=DEV)
    av = torch.tensor(train["A"][mask_val], device=DEV)
    qtest = torch.tensor(test["Q"], device=DEV)
    atest = torch.tensor(test["A"], device=DEV)

    con_val = np.zeros(int(mask_val.sum()))
    con_test = np.zeros(len(test["sid"]))
    for seed in range(1, N_SEEDS + 1):
        m = _train_head(qt, at, yt, qv, av, y[mask_val], seed)
        with torch.no_grad():
            con_val += m(qv, av).cpu().numpy()
            con_test += m(qtest, atest).cpu().numpy()
        if seed % 5 == 0:
            print(f"  seed {seed}/{N_SEEDS} done  ({time.time()-t0:.1f}s)")
    con_val /= N_SEEDS
    con_test /= N_SEEDS

    # Build feature matrices
    x_val = _build_feature_matrix(val_rows, ctr_train, pos_train, global_ctr, skncp_val, con_val)
    x_test = _build_feature_matrix(test, ctr_full, pos_full, global_ctr, skncp_test, con_test)

    base_val = LOGIN_BOOST * (1.0 - train["logged"][mask_val])
    base_test = LOGIN_BOOST * (1.0 - test["logged"])

    ll_val = x_val @ W + base_val
    ll_test = x_test @ W + base_test

    # Z-score normalization params from internal-val
    content_mean, content_sd = _z_params(con_val)
    ll_mean, ll_sd = _z_params(ll_val)

    boosted_test = (
        _z(con_test, content_mean, content_sd)
        + BLEND_ALPHA * _z(ll_test, ll_mean, ll_sd)
    )

    # Threshold: top train_prevalence fraction
    n_test = len(boosted_test)
    k_predict = max(1, int(round(global_ctr * n_test)))
    order = np.argsort(-boosted_test)
    pred = np.zeros(n_test, dtype=np.int8)
    pred[order[:k_predict]] = 1

    print(f"Predicting {int(pred.sum())} clicks out of {n_test} rows "
          f"(rate={int(pred.sum())/n_test:.4%})")

    df_out = pd.DataFrame({
        "SearchID": test["SearchID"],
        "AdID": test["AdID"],
        "IsClick": pred.astype(int),
    })
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
