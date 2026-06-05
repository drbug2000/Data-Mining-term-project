"""Reproduce Task 1 validation metrics (F1=0.1109, AUC=0.713).

Two modes:
  --frozen   Use the frozen content-score cache included with the submission.
             Exactly reproduces the reported F1=0.1109 (deterministic).
             Requires: ../../m06_skncp/results/cs_cache.npz
                     (relative to this file: report/code/task1/)

  --retrain  Retrain the content head (30 seeds) fresh, then score.
             Result will be in the bootstrap CI [0.067, 0.144]; point F1
             may differ slightly from 0.1109 due to stochastic training.

Run:
    python evaluate.py --frozen
    python evaluate.py --retrain
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import RecoDataset
from utils import (
    BLEND_ALPHA,
    FITTED_WEIGHTS,
    LOGIN_BOOST,
    binary_auc,
    boosted_score,
    bootstrap_f1,
    build_feature_matrix,
    build_skncp_index,
    cohens_d,
    collect_click_pairs,
    collect_train,
    compute_skncp,
    ctr_map,
    f1_at_rate,
)
from content_head import train_ensemble

K_SKNCP = 200
N_SEEDS = 30

DATASET_DIR = Path(__file__).resolve().parents[3] / "../datasets"
FROZEN_CACHE = Path(__file__).resolve().parent / "cs_cache.npz"


def run(dataset_dir: Path, use_frozen: bool) -> dict:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  mode={'frozen-cache' if use_frozen else 'retrain'}")

    ds = RecoDataset(dataset_dir).load()
    si = pd.read_csv(ds.dir / "searchinfo.csv")
    ui = pd.read_csv(ds.dir / "userinfo.csv")
    sid2ip = dict(zip(si["SearchID"], si["IPID"]))
    uid2dev = dict(zip(ui["UserID"], ui["UserDeviceID"]))

    print("Loading training data...")
    train = collect_train(ds, sid2ip, uid2dev)

    print("Loading external validation data...")
    val_pairs = ds.val_click_queries()
    y_ext = ds.val_click_answers()["IsClick"].to_numpy()[: len(val_pairs)].astype(np.int8)
    ext = collect_click_pairs(val_pairs, sid2ip, uid2dev, y_arr=y_ext)

    y = train["Y"]
    global_ctr = float(y.mean())
    print(f"train_rows={len(y):,}  global_ctr={global_ctr:.4%}  "
          f"ext_rows={len(y_ext):,}  ext_clicks={int(y_ext.sum())}")

    # Internal 80/20 split
    unique_sids = np.sort(np.unique(train["sid"]))
    val_sids = set(unique_sids[int(len(unique_sids) * 0.8):].tolist())
    mask_val = np.asarray([s in val_sids for s in train["sid"]], dtype=bool)
    mask_tr = ~mask_val
    val_rows = {k: v[mask_val] if isinstance(v, np.ndarray) and len(v) == len(y) else v
                for k, v in train.items()}

    # CTR maps
    ctr_keys = ("ad", "ip", "dev", "cat")
    ctr_tr = {k: ctr_map(train[k][mask_tr], y[mask_tr], global_ctr) for k in ctr_keys}
    ctr_full = {k: ctr_map(train[k], y, global_ctr) for k in ctr_keys}
    pos_tr = ctr_map(train["pos"][mask_tr], y[mask_tr], global_ctr)
    pos_full = ctr_map(train["pos"], y, global_ctr)

    # SKNCP
    print(f"Computing SKNCP (K={K_SKNCP})...")
    idx_tr = build_skncp_index(train, mask_tr, ctr_tr["ad"], global_ctr)
    idx_full = build_skncp_index(train, np.ones(len(y), dtype=bool), ctr_full["ad"], global_ctr)
    skncp_val = compute_skncp(train["Q"][mask_val], train["A"][mask_val], idx_tr, K_SKNCP)
    skncp_ext = compute_skncp(ext["Q"], ext["A"], idx_full, K_SKNCP)

    # Content scores
    if use_frozen:
        print(f"Loading frozen cache from {FROZEN_CACHE}")
        cache = np.load(FROZEN_CACHE)
        con_val = cache["sva"].astype(np.float64)
        con_ext = cache["se"].astype(np.float64)
        if len(con_val) != int(mask_val.sum()) or len(con_ext) != len(y_ext):
            raise SystemExit(
                f"Cache shape mismatch: sva={len(con_val)} (expected {mask_val.sum()}), "
                f"se={len(con_ext)} (expected {len(y_ext)}). Use --retrain."
            )
    else:
        print(f"Training content head ({N_SEEDS} seeds)...")
        [con_val, con_ext] = train_ensemble(
            train["Q"][mask_tr], train["A"][mask_tr], y[mask_tr].astype(np.int8),
            train["Q"][mask_val], train["A"][mask_val], y[mask_val].astype(np.int8),
            (train["Q"][mask_val], train["A"][mask_val]),
            (ext["Q"], ext["A"]),
            n_seeds=N_SEEDS,
            device=device,
        )

    # Feature matrices
    x_val = build_feature_matrix(val_rows, ctr_tr, pos_tr, global_ctr, skncp_val, con_val)
    x_ext = build_feature_matrix(ext, ctr_full, pos_full, global_ctr, skncp_ext, con_ext)

    base_val = LOGIN_BOOST * (1.0 - train["logged"][mask_val])
    base_ext = LOGIN_BOOST * (1.0 - ext["logged"])

    # Z-score params from internal-val
    z_content_mean = float(con_val.mean())
    z_content_sd = float(con_val.std() + 1e-9)
    ll_val = x_val @ FITTED_WEIGHTS + base_val
    z_ll_mean = float(ll_val.mean())
    z_ll_sd = float(ll_val.std() + 1e-9)

    scores_ext = boosted_score(
        x_ext, con_ext, base_ext,
        z_content_mean, z_content_sd,
        z_ll_mean, z_ll_sd,
    )

    result = {
        "auc": binary_auc(scores_ext, y_ext),
        "f1_prevalence": f1_at_rate(scores_ext, y_ext, global_ctr),
        "cohens_d": cohens_d(scores_ext, y_ext),
        "bootstrap": bootstrap_f1(scores_ext, y_ext, global_ctr),
        "mode": "frozen-cache" if use_frozen else "retrain",
        "elapsed_sec": time.time() - t0,
    }
    print("\n--- Results ---")
    print(f"AUC         : {result['auc']:.4f}   (expected 0.713)")
    print(f"F1 (1.11%)  : {result['f1_prevalence']['f1']:.4f}   (expected 0.1109)")
    print(f"Cohen's d   : {result['cohens_d']:.4f}   (expected 0.44)")
    print(f"Bootstrap   : mean={result['bootstrap']['mean']:.4f}  "
          f"CI95={result['bootstrap']['ci95']}")
    print(f"Elapsed     : {result['elapsed_sec']:.1f}s")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frozen", action="store_true",
                       help="Use frozen cs_cache.npz (deterministic)")
    group.add_argument("--retrain", action="store_true",
                       help="Retrain content head (stochastic, ~3 min on GPU)")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args()
    run(args.dataset_dir, use_frozen=args.frozen)
