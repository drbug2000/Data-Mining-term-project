"""
data_utils.py - Data loading, CTR statistics, and chronological split
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATASET_DIR, TRAIN_RATIO, LAPLACE_K


def load_data(dataset_dir: Path = DATASET_DIR) -> dict:
    """
    Load all dataset files into memory.

    Returns a dict with keys:
        searchinfo, adinfo, train_df, val_q, val_a, test_q,
        search_embs, ad_embs, sid2row, aid2row, sid2cat, sid2log
    """
    si = pd.read_csv(dataset_dir / "searchinfo.csv")
    ai = pd.read_csv(dataset_dir / "adinfo.csv")
    tr = pd.read_csv(dataset_dir / "search_stream_training.csv")
    vq = pd.read_csv(dataset_dir / "click_validation_query.csv")
    va = pd.read_csv(dataset_dir / "click_validation_answer.csv")
    tq = pd.read_csv(dataset_dir / "click_test_query.csv")
    se = np.load(dataset_dir / "searchinfo_text_embs.npy")   # (N_search, 384)
    ae = np.load(dataset_dir / "adinfo_title_embs.npy")       # (N_ad, 384)

    return {
        "searchinfo":   si,
        "adinfo":       ai,
        "train_df":     tr,
        "val_q":        vq,
        "val_a":        va,
        "test_q":       tq,
        "search_embs":  se,
        "ad_embs":      ae,
        "sid2row":      dict(zip(si["SearchID"], range(len(si)))),
        "aid2row":      dict(zip(ai["AdID"],     range(len(ai)))),
        "sid2cat":      dict(zip(si["SearchID"], si["CategoryID"])),
        "sid2log":      dict(zip(si["SearchID"], si["IsUserLoggedOn"])),
    }


def chronological_split(
    train_df: pd.DataFrame,
    ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training data chronologically by SearchID.

    The dataset was collected chronologically, so sorting by SearchID
    approximates a temporal split.  The first `ratio` fraction of unique
    SearchIDs goes to internal train; the remainder becomes internal val.

    Args:
        train_df: full training DataFrame (320K rows)
        ratio:    fraction of unique SearchIDs assigned to internal train

    Returns:
        (internal_train_df, internal_val_df)
    """
    all_sids = np.sort(train_df["SearchID"].unique())
    cut = int(len(all_sids) * ratio)
    itr_sids = set(all_sids[:cut])
    iva_sids = set(all_sids[cut:])

    itr = train_df[train_df["SearchID"].isin(itr_sids)].reset_index(drop=True)
    iva = train_df[train_df["SearchID"].isin(iva_sids)].reset_index(drop=True)
    return itr, iva


def compute_ctr_stats(
    df: pd.DataFrame,
    aid2row: dict,
    sid2cat: dict,
    n_ads: int,
    k: float = LAPLACE_K,
) -> dict:
    """
    Compute Laplace-smoothed CTR statistics from a training DataFrame.

    Laplace smoothing formula:
        ctr(entity) = (n_clicks + k * global_ctr) / (n_impressions + k)

    Unseen entities fall back to `global_ctr`.

    Args:
        df:      training DataFrame (must contain IsClick, AdID, SearchID)
        aid2row: AdID → row index in adinfo
        sid2cat: SearchID → CategoryID
        n_ads:   total number of ads (for ad_cnt/ad_clk arrays)
        k:       Laplace smoothing strength

    Returns:
        dict with keys: GCT, FL, ad_sc, ad_m, ct_m, pos_c, clik_index
            GCT:        global click-through rate
            FL:         HistCTR floor value
            ad_sc:      Laplace-smoothed CTR array (indexed by aid2row)
            ad_m:       AdID → smoothed CTR (dict)
            ct_m:       CategoryID → smoothed CTR (dict)
            pos_c:      Position → observed CTR (dict)
            clik_index: SearchID → list of L2-normalised ad embeddings (for SKNCP)
    """
    GCT = float(df["IsClick"].mean())
    FL  = GCT * 0.15          # floor to avoid log(0)
    AP  = GCT * 10            # Laplace prior numerator
    BP  = (1 - GCT) * 10     # Laplace prior denominator

    ad_cnt = np.zeros(n_ads, dtype=np.int32)
    ad_clk = np.zeros(n_ads, dtype=np.int32)
    ad_cd: dict[int, int] = defaultdict(int)
    ac_cd: dict[int, int] = defaultdict(int)
    ct_c:  dict[int, int] = defaultdict(int)
    ct_k:  dict[int, int] = defaultdict(int)
    pos_c: dict[int, float] = {}
    clik_index: dict[int, list] = {}

    for row in df.itertuples(index=False):
        aid = int(row.AdID)
        clk = int(row.IsClick)
        sid = int(row.SearchID)

        if aid in aid2row:
            ar = aid2row[aid]
            ad_cnt[ar] += 1
            ad_clk[ar] += clk
            ad_cd[aid]  += 1
            ac_cd[aid]  += clk

        cat = sid2cat.get(sid, -1)
        if cat != -1:
            ct_c[cat] += 1
            ct_k[cat] += clk

    # Position CTR
    for p in range(1, 8):
        mask = (df["Position"] == p)
        if mask.sum() > 0:
            pos_c[p] = float(df[mask]["IsClick"].mean())

    # Laplace-smoothed CTR
    ad_sc = (ad_clk + AP) / (ad_cnt + AP + BP)  # array indexed by aid2row
    ad_m  = {a: (ac_cd[a] + k * GCT) / (ad_cd[a] + k) for a in ad_cd}
    ct_m  = {c: (ct_k[c]  + k * GCT) / (ct_c[c]  + k) for c in ct_c}

    return {
        "GCT":        GCT,
        "FL":         FL,
        "ad_sc":      ad_sc,
        "ad_m":       ad_m,
        "ct_m":       ct_m,
        "pos_c":      pos_c,
        "clik_index": clik_index,  # filled by build_skncp_index
    }


def build_skncp_index(
    df: pd.DataFrame,
    aid2row: dict,
    ad_embs: np.ndarray,
    clik_index: dict,
) -> None:
    """
    Populate clik_index with L2-normalised ad embeddings for clicked rows.

    clik_index[SearchID] = [normalised_ad_emb_1, normalised_ad_emb_2, ...]

    This index is used by SKNCP to retrieve clicked ads from similar queries.
    """
    for row in df.itertuples(index=False):
        if int(row.IsClick) == 1 and int(row.AdID) in aid2row:
            ar = aid2row[int(row.AdID)]
            emb = _l2norm(ad_embs[ar])
            clik_index.setdefault(int(row.SearchID), []).append(emb)


def _l2norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x) + eps)
