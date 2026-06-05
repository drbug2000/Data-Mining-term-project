"""
skncp.py - Semantic K-Nearest-Neighbor Click Prediction (SKNCP)

Algorithm:
    For a given (query q, ad a) pair:
    1. Find K training queries most similar to q  (cosine similarity on embeddings)
    2. Collect ad embeddings that were actually clicked in those K queries
    3. Return max cosine similarity between ad a and the collected clicked ads

Intuition:
    "Users who searched for something similar to me clicked these ads.
     How similar is the current ad to what they clicked?"

Why SKNCP instead of direct cosine similarity?
    Direct query-ad cosine similarity → Cohen's d = 0.04 (nearly useless)
    SKNCP (K=100, max aggregation) → Cohen's d = 0.44  (11× stronger)
"""
from __future__ import annotations

import numpy as np

from config import SKNCP_K, SKNCP_BATCH


def l2norm_matrix(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalisation."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def build_train_index(
    train_df,
    sid2row: dict,
    search_embs: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """
    Build the KNN search index from training SearchIDs.

    Returns:
        train_sids: list of SearchIDs that have an embedding
        train_enorm: L2-normalised embedding matrix (shape: N × 384)
    """
    train_sids = [s for s in train_df["SearchID"].unique() if s in sid2row]
    train_enorm = l2norm_matrix(
        np.stack([search_embs[sid2row[s]] for s in train_sids])
    )
    return train_sids, train_enorm


def compute_skncp(
    query_df,
    train_sids:  list[int],
    train_enorm: np.ndarray,
    clik_index:  dict,
    sid2row:     dict,
    aid2row:     dict,
    ad_embs:     np.ndarray,
    K:     int = SKNCP_K,
    batch: int = SKNCP_BATCH,
) -> np.ndarray:
    """
    Compute SKNCP scores for every row in query_df.

    Steps:
        1. For each unique query SearchID, find the K nearest training SearchIDs
           using cosine similarity on L2-normalised text embeddings.
        2. Aggregate clicked-ad embeddings from those K neighbours.
        3. For each (SearchID, AdID) pair, compute
               SKNCP = max cosine_similarity(ad_emb, clicked_ad_embs)
           If no clicked ads were found → score = 0.

    Args:
        query_df:    DataFrame with columns SearchID, AdID
        train_sids:  list returned by build_train_index()
        train_enorm: matrix returned by build_train_index()
        clik_index:  SearchID → list of L2-normalised clicked-ad embeddings
        sid2row:     SearchID → row index in searchinfo
        aid2row:     AdID     → row index in adinfo
        ad_embs:     raw ad embedding matrix (will be L2-normalised inside)
        K:           number of nearest neighbours
        batch:       query batch size for matrix multiplication

    Returns:
        np.ndarray of shape (len(query_df),) with SKNCP scores in [0, 1]
    """
    # Step 1: retrieve K nearest neighbours for each unique query SearchID
    val_sids = [s for s in query_df["SearchID"].unique() if s in sid2row]
    val_enorm = l2norm_matrix(
        np.stack([search_embs[sid2row[s]] for s in val_sids])
    )

    sid_to_clicked: dict[int, np.ndarray | None] = {}

    for i in range(0, len(val_sids), batch):
        b_sids = val_sids[i : i + batch]
        b_embs = val_enorm[i : i + batch]

        # cosine similarities: (B × N_train)
        sims = b_embs @ train_enorm.T
        k_eff = min(K, sims.shape[1] - 1)
        top   = np.argpartition(-sims, k_eff, axis=1)[:, :k_eff]

        for j, sid in enumerate(b_sids):
            clicked: list[np.ndarray] = []
            for idx in top[j]:
                tsid = train_sids[idx]
                if tsid in clik_index:
                    clicked.extend(clik_index[tsid])

            sid_to_clicked[sid] = np.stack(clicked[:300]) if clicked else None

    # Step 2 & 3: compute max cosine similarity per (query, ad) pair
    scores = np.zeros(len(query_df), dtype=float)

    for i, row in enumerate(query_df.itertuples(index=False)):
        V  = sid_to_clicked.get(int(row.SearchID))
        ar = aid2row.get(int(row.AdID))

        if V is not None and ar is not None:
            a_norm = l2norm_matrix(ad_embs[ar : ar + 1])[0]
            scores[i] = float((V @ a_norm).max())

    return scores


# make search_embs accessible from module-level (set by main.py)
search_embs: np.ndarray = None  # type: ignore
