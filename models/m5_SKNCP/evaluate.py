"""
evaluate.py - Official evaluation using shared/eval/predictor.py interface

Mirrors the evaluate_task_a() function from the shared evaluation harness.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from model import _rank_normalize, _threshold_f1, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Wilcoxon-Mann-Whitney AUC (exact, no approximation)."""
    s = np.array(scores, dtype=float)
    l = np.array(labels, dtype=int)
    pos = s[l == 1]
    neg = s[l == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    return float(
        (np.sum(pos[:, None] > neg[None, :])
         + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))
    )


def evaluate(
    log_odds:    np.ndarray,
    answers_df:  pd.DataFrame,
    threshold:   float,
) -> dict:
    """
    Evaluate model predictions against ground-truth labels.

    Compatible with shared/eval/predictor.py :: evaluate_task_a().

    Args:
        log_odds:   raw model scores (log-odds, will be rank-normalised)
        answers_df: DataFrame with column 'IsClick'
        threshold:  positive prediction boundary (score > 1-threshold → click)

    Returns:
        dict with keys: f1, precision, recall, auc, accuracy,
                        tp, fp, fn, n_pred_pos
    """
    sc     = _rank_normalize(log_odds)
    labels = np.array(answers_df["IsClick"].tolist(), dtype=int)

    preds = (sc > 1 - threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    pr  = tp / (tp + fp + 1e-8)
    rc  = tp / (tp + fn + 1e-8)
    f1  = 2 * pr * rc / (pr + rc + 1e-8)
    acc = (tp + tn) / len(labels)
    auc = binary_auc(sc, labels)

    return {
        "f1":        f1,
        "precision": pr,
        "recall":    rc,
        "auc":       auc,
        "accuracy":  acc,
        "tp": tp, "fp": fp, "fn": fn,
        "n_pred_pos": int(preds.sum()),
    }


def find_best_threshold(
    log_odds:  np.ndarray,
    labels_df: pd.DataFrame,
) -> float:
    """
    Grid-search for the threshold maximising F1 on a labelled dataset.
    Used only on the internal validation set (never on External Val).
    """
    sc     = _rank_normalize(log_odds)
    labels = np.array(labels_df["IsClick"].tolist(), dtype=int)

    best_f1, best_t = 0.0, 0.01
    for t in np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS):
        f1 = _threshold_f1(sc, labels, t)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
    return best_t
