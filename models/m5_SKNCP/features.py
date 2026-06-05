"""
features.py - Feature engineering for the log-linear scoring model

Feature vector (9 dimensions):
    0  log_hctr      log(HistCTR)                   — main CTR signal
    1  log_adctr     log(ad Laplace CTR)             — ad historical quality
    2  log_catctr    log(category Laplace CTR)       — category popularity
    3  not_logged    1 - IsUserLoggedOn              — login status (fixed offset)
    4  log_pos       log(pos_ctr / global_ctr)       — position debiasing (IPS)
    5  sem_sim       (cosine_sim - 0.50) / 0.22      — semantic similarity
    6  rank          0.5 - normalised_rank           — within-search rank
    7  skncp         SKNCP_score - 0.30              — KNN click support
    8  nbhd_ctr      logit(nbhd_ctr) - logit(GCT)   — ad neighbourhood CTR

Note: feature 3 (not_logged) uses a *fixed* LOGIN_OFFSET instead of a
      learned weight, because it is already in log-odds units.
"""
from __future__ import annotations

import numpy as np

from config import LOGIN_OFFSET
from skncp import l2norm_matrix


def _logit(p: float, eps: float = 1e-7) -> float:
    p = max(eps, min(1 - eps, p))
    return float(np.log(p / (1 - p)))


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float((a @ b) / (na * nb + eps))


def build_neighbourhood_ctr(
    query_df,
    ad_embs:  np.ndarray,
    ad_sc:    np.ndarray,
    aid2row:  dict,
    top_n:    int = 30,
) -> dict[int, float]:
    """
    For each unique AdID in query_df, compute the average Laplace-smoothed CTR
    of its top_n most semantically similar ads (excluding itself).

    This provides a "neighbourhood prior" for ads with sparse click history.
    """
    ae_norm = l2norm_matrix(ad_embs)
    unique_aids = [a for a in query_df["AdID"].unique() if a in aid2row]
    nbhd: dict[int, float] = {}

    batch = 1000
    for i in range(0, len(unique_aids), batch):
        batch_aids = unique_aids[i : i + batch]
        rows = [aid2row[a] for a in batch_aids]
        sims = ae_norm[rows] @ ae_norm.T

        for j, a in enumerate(batch_aids):
            sims[j, aid2row[a]] = -1.0         # exclude self
            top = np.argpartition(-sims[j], top_n)[:top_n]
            nbhd[a] = float(ad_sc[top].mean())

    return nbhd


def build_within_search_rank(query_df) -> dict[tuple[int, int], float]:
    """
    Rank each ad within its search event by HistCTR (descending).
    Returns normalised rank in [0, 1]:
        best ad  → rank = 1 / n_ads
        worst ad → rank = n_ads / n_ads = 1
    """
    rank_map: dict[tuple[int, int], float] = {}
    for sid, grp in query_df.groupby("SearchID"):
        aids = grp.sort_values("HistCTR", ascending=False)["AdID"].tolist()
        n    = len(aids)
        for i, a in enumerate(aids, start=1):
            rank_map[(int(sid), int(a))] = i / n
    return rank_map


def build_feature_matrix(
    query_df,
    skncp_scores: np.ndarray,
    ctr_stats:    dict,
    sid2row:      dict,
    aid2row:      dict,
    sid2cat:      dict,
    sid2log:      dict,
    search_embs:  np.ndarray,
    ad_embs:      np.ndarray,
) -> np.ndarray:
    """
    Assemble the 9-dimensional feature matrix for all rows in query_df.

    Each feature is designed to be in a similar log-odds scale so that
    the linear model weights are comparable across features.

    Args:
        query_df:     DataFrame with SearchID, AdID, Position, HistCTR
        skncp_scores: pre-computed SKNCP scores (shape: len(query_df),)
        ctr_stats:    output of compute_ctr_stats()
        *mappings:    lookup dicts produced by load_data()

    Returns:
        X: np.ndarray of shape (len(query_df), 9)
    """
    GCT   = ctr_stats["GCT"]
    FL    = ctr_stats["FL"]
    ad_m  = ctr_stats["ad_m"]
    ct_m  = ctr_stats["ct_m"]
    pos_c = ctr_stats["pos_c"]
    ad_sc = ctr_stats["ad_sc"]

    nbhd    = build_neighbourhood_ctr(query_df, ad_embs, ad_sc, aid2row)
    rank_mp = build_within_search_rank(query_df)

    rows = []
    for i, row in enumerate(query_df.itertuples(index=False)):
        sid = int(row.SearchID)
        aid = int(row.AdID)
        pos = int(row.Position)
        h   = max(float(row.HistCTR), FL)

        # Feature 0: log HistCTR
        f0 = float(np.log(h))

        # Feature 1: log ad Laplace CTR
        f1 = float(np.log(max(ad_m.get(aid, GCT), FL)))

        # Feature 2: log category Laplace CTR
        f2 = float(np.log(max(ct_m.get(sid2cat.get(sid, -1), GCT), FL)))

        # Feature 3: not-logged-in indicator (used with fixed LOGIN_OFFSET)
        f3 = float(1 - sid2log.get(sid, 1))

        # Feature 4: position debiasing (IPS) — log(pos_ctr / global_ctr)
        f4 = float(np.log(pos_c.get(pos, GCT) / GCT + 1e-8))

        # Feature 5: semantic similarity — centred and scaled
        sr = sid2row.get(sid)
        ar = aid2row.get(aid)
        if sr is not None and ar is not None:
            sem = _cosine(search_embs[sr], ad_embs[ar])
        else:
            sem = 0.5
        f5 = (sem - 0.50) / 0.22

        # Feature 6: within-search rank (higher HistCTR → lower rank value)
        f6 = 0.5 - rank_mp.get((sid, aid), 0.5)

        # Feature 7: SKNCP — centred at empirical mean (~0.30)
        f7 = skncp_scores[i] - 0.30

        # Feature 8: neighbourhood CTR — log-odds relative to global CTR
        f8 = _logit(min(float(nbhd.get(aid, GCT)), 0.999)) - _logit(GCT)

        rows.append([f0, f1, f2, f3, f4, f5, f6, f7, f8])

    return np.array(rows, dtype=float)
