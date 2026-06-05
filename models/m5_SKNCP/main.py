"""
main.py - End-to-end pipeline for Task 1: Click Prediction
===========================================================

Usage:
    python main.py

Output:
    - Prints validation F1 and AUC to stdout
    - Writes click_test_answer.csv (submission file for blind test)

Validation methodology:
    All parameter selection is performed on the INTERNAL validation set
    (Training data sorted by SearchID, last 20%).
    The EXTERNAL validation set (click_validation_query/answer.csv) is
    used ONLY for final performance reporting — never for fitting.

Performance (External Validation):
    F1 = 0.0945 ~ 0.0949   AUC = 0.710
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

import skncp as sk_module
from config   import DATASET_DIR
from data_utils import (
    load_data,
    chronological_split,
    compute_ctr_stats,
    build_skncp_index,
)
from skncp    import build_train_index, compute_skncp
from features import build_feature_matrix
from model    import (
    score,
    coordinate_ascent,
    oof_threshold,
    predict,
)
from evaluate import evaluate, find_best_threshold


def main():
    t_start = time.time()

    # ── 1. Load data ──────────────────────────────────────────────
    print("Loading data ...", flush=True)
    data = load_data(DATASET_DIR)

    tr      = data["train_df"]
    vq      = data["val_q"]
    va      = data["val_a"]
    tq      = data["test_q"]
    se      = data["search_embs"]
    ae      = data["ad_embs"]
    sid2row = data["sid2row"]
    aid2row = data["aid2row"]
    sid2cat = data["sid2cat"]
    sid2log = data["sid2log"]

    # make search_embs available to skncp.py
    sk_module.search_embs = se

    n_ads = len(data["adinfo"])

    # ── 2. Chronological 80/20 split ─────────────────────────────
    print("Splitting data (chronological 80/20) ...", flush=True)
    itr_df, iva_df = chronological_split(tr)
    iva_labels = iva_df["IsClick"].to_numpy()

    print(f"  Internal train: {len(itr_df):,} rows  "
          f"({itr_df['IsClick'].sum():,} clicks)")
    print(f"  Internal val:   {len(iva_df):,} rows  "
          f"({iva_df['IsClick'].sum():,} clicks)")
    print(f"  External val:   {len(vq):,} rows  "
          f"({va['IsClick'].sum():,} clicks)")

    # ── 3. CTR statistics (Internal Train only) ───────────────────
    print("Computing CTR statistics ...", flush=True)
    ctr = compute_ctr_stats(itr_df, aid2row, sid2cat, n_ads)
    build_skncp_index(itr_df, aid2row, ae, ctr["clik_index"])

    # ── 4. SKNCP scores ───────────────────────────────────────────
    print("Computing SKNCP (K=100) ...", flush=True)
    t0 = time.time()
    train_sids, train_enorm = build_train_index(itr_df, sid2row, se)

    sk_iva = compute_skncp(iva_df, train_sids, train_enorm,
                           ctr["clik_index"], sid2row, aid2row, ae)
    print(f"  Internal val SKNCP: {time.time()-t0:.1f}s  "
          f"coverage={(sk_iva > 0).mean()*100:.1f}%")

    # Full 320K index for External Val and Test (more coverage, no leakage)
    all_clik: dict = {}
    build_skncp_index(tr, aid2row, ae, all_clik)
    all_sids, all_enorm = build_train_index(tr, sid2row, se)

    sk_ext = compute_skncp(vq, all_sids, all_enorm, all_clik, sid2row, aid2row, ae)
    sk_tst = compute_skncp(tq, all_sids, all_enorm, all_clik, sid2row, aid2row, ae)
    print(f"  External val SKNCP coverage={(sk_ext > 0).mean()*100:.1f}%")

    # ── 5. Feature matrices ───────────────────────────────────────
    print("Building feature matrices ...", flush=True)
    X_iva = build_feature_matrix(iva_df, sk_iva, ctr,
                                  sid2row, aid2row, sid2cat, sid2log, se, ae)
    X_ext = build_feature_matrix(vq,     sk_ext, ctr,
                                  sid2row, aid2row, sid2cat, sid2log, se, ae)
    X_tst = build_feature_matrix(tq,     sk_tst, ctr,
                                  sid2row, aid2row, sid2cat, sid2log, se, ae)
    print(f"  Feature matrix shape: {X_iva.shape}")

    # ── 6. Coordinate Ascent (Internal Val only) ──────────────────
    print("Coordinate Ascent (w_min=0.5) ...", flush=True)
    w = coordinate_ascent(X_iva, iva_labels)

    FEAT_NAMES = ["hctr","adctr","catctr","pos","sem","rank","skncp","nbhd"]
    print(f"  Weights: { {n: round(float(v), 2) for n,v in zip(FEAT_NAMES, w)} }")
    print(f"  Internal val Top-K F1 = "
          f"{__import__('model').topk_f1(score(X_iva, w), iva_labels):.4f}")

    # ── 7. OOF threshold (Internal Val) ──────────────────────────
    print("OOF threshold (5-fold) ...", flush=True)
    lo_iva = score(X_iva, w)
    rate   = oof_threshold(lo_iva, iva_labels)
    print(f"  OOF positive rate = {rate:.4f}")

    # ── 8. External Validation — FINAL REPORT (no fitting here) ──
    lo_ext = score(X_ext, w)
    result = evaluate(lo_ext, va, rate)

    print()
    print("=" * 55)
    print("  External Validation Results (no parameter fitting)")
    print("=" * 55)
    print(f"  F1        = {result['f1']:.4f}")
    print(f"  Precision = {result['precision']:.4f}")
    print(f"  Recall    = {result['recall']:.4f}")
    print(f"  AUC       = {result['auc']:.4f}")
    print(f"  Accuracy  = {result['accuracy']:.4f}")
    print(f"  TP={result['tp']}  FP={result['fp']}  FN={result['fn']}")
    print()

    # ── 9. Test submission ────────────────────────────────────────
    lo_tst = score(X_tst, w)
    preds  = predict(lo_tst, rate)

    out = tq[["SearchID", "AdID", "Position", "HistCTR"]].copy()
    out["IsClick"] = preds
    out_path = Path("click_test_answer.csv")
    out.to_csv(out_path, index=False)

    print(f"  Predicted positives: {preds.sum()} / {len(preds)}")
    print(f"  Submission saved → {out_path.absolute()}")
    print(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
