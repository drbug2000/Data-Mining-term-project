"""Task 1 utility functions: CTR features, SKNCP, AUC/F1 metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

KS = 20       # Laplace smoothing count for entity CTRs
LOGIN_BOOST = np.log(1.7)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    return float(
        (np.sum(pos[:, None] > neg[None, :]) +
         0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))
    )


def best_f1_topk(scores: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Return (best_F1, k) by sweeping the top-k threshold."""
    order = np.argsort(-scores)
    ys = y[order].astype(np.float64)
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    P = ys.sum()
    if P == 0:
        return 0.0, 1
    f1 = 2 * tp / (k + P)
    j = int(np.argmax(f1))
    return float(f1[j]), int(k[j])


def f1_at_rate(scores: np.ndarray, y: np.ndarray, rate: float) -> dict:
    k = max(1, int(round(float(rate) * len(scores))))
    order = np.argsort(-scores)
    pred = np.zeros(len(scores), dtype=np.int8)
    pred[order[:k]] = 1
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return {
        "f1": 2 * tp / denom if denom else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "k": k, "tp": tp, "fp": fp, "fn": fn,
    }


def bootstrap_f1(scores: np.ndarray, y: np.ndarray, rate: float,
                 seed: int = 1, n_boot: int = 2000) -> dict:
    rng = np.random.RandomState(seed)
    n = len(y)
    vals = [f1_at_rate(scores[rng.randint(0, n, n)], y[rng.randint(0, n, n)], rate)["f1"]
            for _ in range(n_boot)]
    arr = np.asarray(vals)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
    }


def cohens_d(scores: np.ndarray, y: np.ndarray) -> float:
    scores = np.asarray(scores, np.float64)
    sd = float(scores.std() + 1e-12)
    return float((scores[y == 1].mean() - scores[y == 0].mean()) / sd)


# ---------------------------------------------------------------------------
# Coordinate-ascent F1 weight fitting
# ---------------------------------------------------------------------------

def fit_f1_weights(X: np.ndarray, y: np.ndarray, base=0.0,
                   grid=None, n_pass: int = 5) -> np.ndarray:
    """Coordinate ascent: find non-negative exponents w that maximize top-k F1.

    score = X @ w + base
    """
    if grid is None:
        grid = np.arange(0.0, 3.05, 0.5)
    w = np.ones(X.shape[1], dtype=np.float64)
    best = best_f1_topk(X @ w + base, y)[0]
    for _ in range(n_pass):
        improved = False
        for j in range(X.shape[1]):
            bj, bf = w[j], best
            for gv in grid:
                w[j] = gv
                f = best_f1_topk(X @ w + base, y)[0]
                if f > bf:
                    bf, bj = f, gv
            w[j] = bj
            if bf > best + 1e-12:
                best, improved = bf, True
        if not improved:
            break
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# CTR features
# ---------------------------------------------------------------------------

def safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.asarray(x, np.float64), 1e-6))


def ctr_map(keys: np.ndarray, y: np.ndarray, global_ctr: float) -> pd.Series:
    """Laplace-smoothed per-key CTR from a subset of rows."""
    df = pd.DataFrame({"k": keys, "y": y})
    gp = df.groupby("k")["y"].agg(["sum", "count"])
    return (gp["sum"] + KS * global_ctr) / (gp["count"] + KS)


def lookup_ctr(keys: np.ndarray, series: pd.Series, default: float) -> np.ndarray:
    return pd.Series(keys).map(series).fillna(default).to_numpy(dtype=np.float64)


def rank_feature(search_ids: np.ndarray, hist_ctrs: np.ndarray) -> np.ndarray:
    """Within-search HistCTR rank, centered at 0 (0.5 - percentile)."""
    rank = pd.DataFrame({"sid": search_ids, "hist": hist_ctrs}).groupby("sid")["hist"].rank(pct=True)
    return 0.5 - rank.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_train(ds, sid2ip: dict, uid2dev: dict) -> dict:
    q, a, y = [], [], []
    ad, cat, uid, sid, hist, logged, pos = [], [], [], [], [], [], []
    for ev in ds.training_stream():
        for rec in ev.ads:
            q.append(ev.search_emb); a.append(rec.ad_emb)
            y.append(int(rec.is_click))
            ad.append(int(rec.ad_id)); cat.append(int(rec.category_id))
            uid.append(int(ev.user_id)); sid.append(int(ev.search_id))
            hist.append(float(rec.hist_ctr) if rec.hist_ctr is not None else 0.0)
            logged.append(float(ev.is_logged_on)); pos.append(int(rec.position))
    out = {
        "Q": l2_normalize(np.asarray(q, np.float32)),
        "A": l2_normalize(np.asarray(a, np.float32)),
        "Y": np.asarray(y, np.int8),
        "ad": np.asarray(ad), "cat": np.asarray(cat),
        "uid": np.asarray(uid), "sid": np.asarray(sid),
        "hist": np.asarray(hist, np.float64),
        "logged": np.asarray(logged, np.float64),
        "pos": np.asarray(pos, np.float64),
    }
    out["ip"] = np.asarray([sid2ip.get(int(s), -1) for s in out["sid"]])
    out["dev"] = np.asarray([uid2dev.get(int(u), -1) for u in out["uid"]])
    return out


def collect_click_pairs(pairs, sid2ip: dict, uid2dev: dict, y_arr=None) -> dict:
    """Convert (SearchEvent, AdRecord) pairs to feature arrays."""
    out = {
        "Q": l2_normalize(np.asarray([ev.search_emb for ev, _ in pairs], np.float32)),
        "A": l2_normalize(np.asarray([rec.ad_emb for _, rec in pairs], np.float32)),
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
    if y_arr is not None:
        out["Y"] = y_arr
    return out


# ---------------------------------------------------------------------------
# SKNCP (Semantic K-Nearest-Neighbor Click Prediction)
# ---------------------------------------------------------------------------

def build_skncp_index(train: dict, mask: np.ndarray,
                      ad_ctr_series: pd.Series, global_ctr: float) -> dict:
    """Index of clicked ads from selected training rows.

    mask: boolean array selecting which train rows to include.
    """
    click_idx = np.flatnonzero(mask & (train["Y"] == 1))
    clicked_ad_ctr = lookup_ctr(train["ad"][click_idx], ad_ctr_series, global_ctr)
    return {
        "Q": train["Q"][click_idx].astype(np.float32),
        "A": train["A"][click_idx].astype(np.float32),
        "ad_ctr": clicked_ad_ctr.astype(np.float32),
        "n_clicks": int(len(click_idx)),
    }


def compute_skncp(q: np.ndarray, a: np.ndarray, index: dict,
                  k: int, block_size: int = 1024) -> np.ndarray:
    """SKNCP(s, ad) = max over K nearest clicked ads of cos(ad_emb, clicked_ad_emb).

    Equation (1) in the report:
        SKNCP(s, a) = max_{a' in C_K(s)} cos(z_a, z_{a'})
    where C_K(s) = clicked ads of K training searches most similar to s.
    """
    n = len(q)
    if index["n_clicks"] == 0:
        return np.zeros(n, np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_t = torch.tensor(q, dtype=torch.float32, device=device)
    a_t = torch.tensor(a, dtype=torch.float32, device=device)
    tq = torch.tensor(index["Q"], dtype=torch.float32, device=device)
    ta = torch.tensor(index["A"], dtype=torch.float32, device=device)
    k_eff = min(int(k), tq.shape[0])
    out = np.zeros(n, np.float32)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        sims = q_t[start:end] @ tq.T
        top = sims.topk(k_eff, dim=1).indices
        pools = ta[top]
        target = a_t[start:end].unsqueeze(1)
        out[start:end] = (pools * target).sum(dim=2).max(dim=1).values.detach().cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(rows: dict, ctr_maps: dict, pos_ctr: pd.Series,
                         global_ctr: float, skncp: np.ndarray,
                         content_score: np.ndarray) -> np.ndarray:
    """9-column feature matrix: [logHist, ad_ctr, ip_ctr, dev_ctr, cat_ctr,
                                  IPS_pos, rank, SKNCP, content_score]"""
    ad_ctr = lookup_ctr(rows["ad"], ctr_maps["ad"], global_ctr)
    ip_ctr = lookup_ctr(rows["ip"], ctr_maps["ip"], global_ctr)
    dev_ctr = lookup_ctr(rows["dev"], ctr_maps["dev"], global_ctr)
    cat_ctr = lookup_ctr(rows["cat"], ctr_maps["cat"], global_ctr)
    pos_rate = lookup_ctr(rows["pos"], pos_ctr, global_ctr)
    rank = rank_feature(rows["sid"], rows["hist"])
    return np.column_stack([
        safe_log(rows["hist"]),
        safe_log(ad_ctr),
        safe_log(ip_ctr),
        safe_log(dev_ctr),
        safe_log(cat_ctr),
        safe_log(pos_rate) - np.log(global_ctr),
        rank,
        skncp,
        content_score,
    ]).astype(np.float64)


# ---------------------------------------------------------------------------
# Boosted score (Equation 2 in report)
# ---------------------------------------------------------------------------

FEATURE_NAMES = ("logHist", "ad_ctr", "ip_ctr", "dev_ctr", "cat_ctr",
                 "IPS_pos", "rank", "SKNCP", "content_score")

FITTED_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.5, 1.0, 1.0, 1.0, 3.0, 1.0], dtype=np.float64)

BLEND_ALPHA = 0.2   # alpha in Eq. (2): score = z(content) + alpha * z(s_LL)


def boosted_score(x: np.ndarray, content: np.ndarray, base: np.ndarray,
                  z_content_mean: float, z_content_sd: float,
                  z_ll_mean: float, z_ll_sd: float,
                  weights: np.ndarray = FITTED_WEIGHTS,
                  alpha: float = BLEND_ALPHA) -> np.ndarray:
    """Eq. (2): score = z(content) + alpha * z(s_LL).

    x       : feature matrix (N, 9)
    content : content head scores (N,)  — column 8 of x, extracted for z-scoring
    base    : login offset term (N,)
    z_*     : z-score normalization params computed on internal-val
    """
    s_ll = x @ weights + base
    z_c = (content - z_content_mean) / z_content_sd
    z_ll = (s_ll - z_ll_mean) / z_ll_sd
    return z_c + alpha * z_ll
