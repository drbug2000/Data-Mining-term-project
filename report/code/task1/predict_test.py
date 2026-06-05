"""Generate click_test_answer.csv for Task 1 submission.

Pipeline (mirrors the submitted model exactly):
  1. Train the content head ensemble on the internal-train split.
  2. Build SKNCP index from full training data (K=200).
  3. Compute SKNCP + 9-feature matrix for test rows.
  4. Apply fitted weights (coordinate-ascent F1 objective, internal-val only).
  5. Boost: score = z(content-strong) + 0.2 * z(s_LL)   [Eq. 2 in report]
  6. Threshold: top 1.11% = train click rate.

Output: click_test_answer.csv at the submission root
        (columns: SearchID, AdID, IsClick)

Run:
    python predict_test.py --dataset-dir /path/to/datasets
"""

from __future__ import annotations

import argparse
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
    boosted_score,
    build_feature_matrix,
    build_skncp_index,
    collect_click_pairs,
    collect_train,
    compute_skncp,
    ctr_map,
)
from content_head import train_ensemble

K_SKNCP = 200
N_SEEDS = 30

SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = SUBMISSION_ROOT / "click_test_answer.csv"


def main(dataset_dir: Path | None = None) -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = RecoDataset(dataset_dir) if dataset_dir is not None else RecoDataset()
    ds = ds.load()
    import pandas as _pd
    si = _pd.read_csv(ds.dir / "searchinfo.csv")
    ui = _pd.read_csv(ds.dir / "userinfo.csv")
    sid2ip = dict(zip(si["SearchID"], si["IPID"]))
    uid2dev = dict(zip(ui["UserID"], ui["UserDeviceID"]))

    print("Loading training data...")
    train = collect_train(ds, sid2ip, uid2dev)
    print("Loading test data...")
    test_pairs = ds.test_click_queries()
    test = collect_click_pairs(test_pairs, sid2ip, uid2dev)

    y = train["Y"]
    global_ctr = float(y.mean())
    n_test = len(test["sid"])
    print(f"train_rows={len(y):,}  global_ctr={global_ctr:.4%}  test_rows={n_test:,}")

    # Internal 80/20 split (for z-score normalization params only)
    unique_sids = np.sort(np.unique(train["sid"]))
    val_sids = set(unique_sids[int(len(unique_sids) * 0.8):].tolist())
    mask_val = np.asarray([s in val_sids for s in train["sid"]], dtype=bool)
    mask_tr = ~mask_val

    val_rows = {k: v[mask_val] if isinstance(v, np.ndarray) and len(v) == len(y) else v
                for k, v in train.items()}

    # CTR maps: internal-train for val z-score params; full-train for test scoring
    ctr_keys = ("ad", "ip", "dev", "cat")
    ctr_tr = {k: ctr_map(train[k][mask_tr], y[mask_tr], global_ctr) for k in ctr_keys}
    ctr_full = {k: ctr_map(train[k], y, global_ctr) for k in ctr_keys}
    pos_tr = ctr_map(train["pos"][mask_tr], y[mask_tr], global_ctr)
    pos_full = ctr_map(train["pos"], y, global_ctr)

    # SKNCP index
    print(f"Building SKNCP index (K={K_SKNCP})...")
    idx_tr = build_skncp_index(train, mask_tr, ctr_tr["ad"], global_ctr)
    idx_full = build_skncp_index(train, np.ones(len(y), dtype=bool), ctr_full["ad"], global_ctr)
    print(f"  clicked in full index: {idx_full['n_clicks']:,}")

    skncp_val = compute_skncp(train["Q"][mask_val], train["A"][mask_val], idx_tr, K_SKNCP)
    skncp_test = compute_skncp(test["Q"], test["A"], idx_full, K_SKNCP)

    # Content head ensemble.
    print(f"Training content head ({N_SEEDS} seeds)...")
    [con_val, con_test] = train_ensemble(
        train["Q"][mask_tr], train["A"][mask_tr], y[mask_tr].astype(np.int8),
        train["Q"][mask_val], train["A"][mask_val], y[mask_val].astype(np.int8),
        (train["Q"][mask_val], train["A"][mask_val]),
        (test["Q"], test["A"]),
        n_seeds=N_SEEDS,
        device=device,
    )

    # Feature matrices
    x_val = build_feature_matrix(val_rows, ctr_tr, pos_tr, global_ctr, skncp_val, con_val)
    x_test = build_feature_matrix(test, ctr_full, pos_full, global_ctr, skncp_test, con_test)

    base_val = LOGIN_BOOST * (1.0 - train["logged"][mask_val])
    base_test = LOGIN_BOOST * (1.0 - test["logged"])

    # Z-score normalization params from internal-val
    z_content_mean, z_content_sd = float(con_val.mean()), float(con_val.std() + 1e-9)
    ll_val = x_val @ FITTED_WEIGHTS + base_val
    z_ll_mean, z_ll_sd = float(ll_val.mean()), float(ll_val.std() + 1e-9)

    scores_test = boosted_score(
        x_test, con_test, base_test,
        z_content_mean, z_content_sd,
        z_ll_mean, z_ll_sd,
    )

    # Threshold: top train_prevalence fraction
    k_predict = max(1, int(round(global_ctr * n_test)))
    pred = np.zeros(n_test, dtype=np.int8)
    pred[np.argsort(-scores_test)[:k_predict]] = 1
    print(f"Predicting {int(pred.sum())} clicks / {n_test} rows "
          f"(rate={int(pred.sum())/n_test:.4%})")

    df_out = pd.DataFrame({
        "SearchID": test["SearchID"],
        "AdID": test["AdID"],
        "IsClick": pred.astype(int),
    })
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Directory containing the provided dataset files. "
                             "If omitted, DATASET_DIR env var and nearby datasets/ folders are tried.")
    args = parser.parse_args()
    main(args.dataset_dir)
