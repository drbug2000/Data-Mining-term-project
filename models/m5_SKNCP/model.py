"""
model.py - Log-linear scoring model with F1-objective Coordinate Ascent

Scoring function (log-odds):
    lo(q, a, p) = w[0]*log_hctr
                + w[1]*log_adctr
                + w[2]*log_catctr
                + LOGIN_OFFSET * not_logged      ← fixed, not learned
                + w[3]*log_pos
                + w[4]*sem_sim
                + w[5]*rank
                + w[6]*skncp
                + w[7]*nbhd_ctr

    score = sigmoid(lo)

Why log-odds space?
    Under the Naive Bayes assumption, independent evidence signals add in
    log-odds space: logit(P(y=1|x1,x2,...)) = logit(prior) + Σ log-BF_i
    This avoids probabilities exceeding 1 and gives interpretable weights.

Why F1-objective Coordinate Ascent instead of log-loss?
    With CTR ≈ 1.1%, standard log-loss optimisation yields poor F1.
    Directly maximising Top-K F1 on the internal validation set aligns
    training with the evaluation metric.

Why w_min = 0.5?
    Without a lower bound, Coordinate Ascent occasionally assigns
    weight ≈ 0 to HistCTR (AUC = 0.669, the strongest signal) when
    the internal validation partition exhibits local distributional quirks.
    The floor of 0.5 prevents this degenerate solution.
"""
from __future__ import annotations

import numpy as np

from config import (
    LOGIN_OFFSET,
    WEIGHT_MIN, WEIGHT_MAX, WEIGHT_STEP,
    MAX_ITER,
    OOF_FOLDS,
    THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS,
)


# ── Scoring ───────────────────────────────────────────────────────

def score(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute log-odds scores for feature matrix X with weight vector w.

    X columns:  [log_hctr, log_adctr, log_catctr, not_logged,
                 log_pos, sem_sim, rank, skncp, nbhd_ctr]
    w shape:    (8,)   — one weight per feature except not_logged (fixed)

    Returns: log-odds array of shape (len(X),)
    """
    return (
        w[0] * X[:, 0] +   # log_hctr
        w[1] * X[:, 1] +   # log_adctr
        w[2] * X[:, 2] +   # log_catctr
        LOGIN_OFFSET * X[:, 3] +   # not_logged (fixed offset)
        w[3] * X[:, 4] +   # log_pos
        w[4] * (X[:, 5] - 0.5) / 0.22 +   # sem_sim (centred)
        w[5] * X[:, 6] +   # rank
        w[6] * X[:, 7] +   # skncp
        w[7] * X[:, 8]     # nbhd_ctr
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ── Evaluation metrics ────────────────────────────────────────────

def topk_f1(scores: np.ndarray, labels: np.ndarray, max_k: int = 3000) -> float:
    """
    Sweep K from 1 to max_k; return the maximum achievable F1.

    This is an oracle upper bound on F1 for a given ranking.
    Used during Coordinate Ascent on the internal validation set.
    """
    order = np.argsort(-scores)
    ls    = labels[order]
    npos  = labels.sum()
    if npos == 0:
        return 0.0

    best = 0.0
    cum  = 0
    for k in range(1, min(len(scores), max_k) + 1):
        cum += int(ls[k - 1])
        pr = cum / k
        rc = cum / npos
        f1 = 2 * pr * rc / (pr + rc + 1e-8)
        if f1 > best:
            best = f1
    return best


# ── Coordinate Ascent ─────────────────────────────────────────────

def coordinate_ascent(
    X:      np.ndarray,
    labels: np.ndarray,
    w_min:  float = WEIGHT_MIN,
    w_max:  float = WEIGHT_MAX,
    step:   float = WEIGHT_STEP,
    max_iter: int = MAX_ITER,
) -> np.ndarray:
    """
    Optimise feature weights by coordinate ascent on Top-K F1.

    Each iteration cycles through every weight, performing a 1-D grid
    search while holding all other weights fixed.  The grid runs from
    w_min to w_max in steps of `step`.  Stops when no weight update
    improves F1 (convergence).

    The lower bound w_min = 0.5 prevents HistCTR from being zeroed out
    due to local optima specific to the internal validation partition.

    Args:
        X:        feature matrix (n_samples × 9)
        labels:   binary click labels (n_samples,)
        w_min:    minimum allowed weight (default 0.5)
        w_max:    maximum allowed weight (default 3.0)
        step:     grid step size (default 0.1)
        max_iter: maximum number of full passes

    Returns:
        w: optimised weight vector (8,)
    """
    n_weights = 8
    w         = np.ones(n_weights)
    best_f1   = 0.0
    grid      = np.arange(w_min, w_max + step / 2, step)

    for iteration in range(max_iter):
        improved = False
        for j in range(n_weights):
            best_v = w[j]
            for v in grid:
                w[j] = v
                f1   = topk_f1(score(X, w), labels)
                if f1 > best_f1:
                    best_f1 = f1
                    best_v  = v
                    improved = True
            w[j] = best_v

        if not improved:
            break   # converged

    return w


# ── Threshold selection (OOF) ─────────────────────────────────────

def oof_threshold(
    lo_iva:    np.ndarray,
    iva_labels: np.ndarray,
    n_folds:   int = OOF_FOLDS,
    seed:      int = 42,
) -> float:
    """
    Estimate the optimal positive prediction rate using Out-of-Fold (OOF)
    5-fold cross-validation on the internal validation set.

    For each fold:
        rank-normalise scores → sweep threshold → find F1-maximising threshold
    Average the five threshold values.

    Applying the averaged threshold to unseen data avoids over-fitting
    the threshold to a single evaluation partition.

    Returns:
        oof_rate: fraction of samples to predict as positive
    """
    rng      = np.random.default_rng(seed)
    idx      = np.arange(len(lo_iva))
    rng.shuffle(idx)
    fold_sz  = len(idx) // n_folds
    rates    = []

    for fold in range(n_folds):
        vi       = idx[fold * fold_sz : (fold + 1) * fold_sz]
        sc_fold  = _rank_normalize(lo_iva[vi])
        lbl_fold = iva_labels[vi]

        best_f1, best_t = 0.0, 0.01
        for t in np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS):
            f1 = _threshold_f1(sc_fold, lbl_fold, t)
            if f1 > best_f1:
                best_f1 = f1
                best_t  = t
        rates.append(best_t)

    return float(np.mean(rates))


def _rank_normalize(lo: np.ndarray) -> np.ndarray:
    """Map log-odds to rank-based scores in [0, 1] (highest score → ~1)."""
    N    = len(lo)
    rank = np.argsort(np.argsort(-lo))   # 0 = best
    return (N - 1 - rank) / N


def _threshold_f1(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Compute F1 when predicting 1 for score > 1 - threshold."""
    preds = (scores > 1 - threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    pr = tp / (tp + fp + 1e-8)
    rc = tp / (tp + fn + 1e-8)
    return 2 * pr * rc / (pr + rc + 1e-8)


# ── Prediction ────────────────────────────────────────────────────

def predict(
    lo:       np.ndarray,
    oof_rate: float,
) -> np.ndarray:
    """
    Convert log-odds scores to binary predictions using the OOF threshold.

    Predicts the top `oof_rate` fraction of samples as positive (IsClick=1).

    Args:
        lo:       log-odds scores (n_samples,)
        oof_rate: positive prediction rate from oof_threshold()

    Returns:
        binary predictions (n_samples,)
    """
    k      = max(1, int(len(lo) * oof_rate))
    pred   = np.zeros(len(lo), dtype=int)
    top_k  = np.argsort(-lo)[:k]
    pred[top_k] = 1
    return pred
