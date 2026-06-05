"""
task1_skncp_boosted.py — boosted model6 with SKNCP + content score.

This script keeps SKNCP in the scoring schema and adds the cached m04
content-strong score as a high-AUC semantic signal. It uses only the sorted
SearchID 80/20 internal split for fitting weights. The reported primary F1 uses
the train-prevalence positive rate, so external labels are not used for
threshold selection.

Prerequisite:
    python -X utf8 models/m04_gated/experiments/exp_content_strong.py

Run:
    python -X utf8 models/m06_skncp/experiments/task1_skncp_boosted.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.m04_gated.gated_ctr import _best_f1_topk, _binary_auc, _fit_f1_exponents
from models.m06_skncp.experiments.task1_skncp_model6 import (
    DATASET_DIR,
    OUT_DIR,
    _bootstrap_f1,
    _build_skncp_index,
    _collect_external,
    _collect_train,
    _ctr_series,
    _f1_at_rate,
    _lookup,
    _rank_feature,
    _skncp_scores,
)
from shared.data.dataset import RecoDataset

CONTENT_SCORE_CACHE = Path("/tmp/cs_cache.npz")
OUT_JSON = OUT_DIR / "model6_boosted_task1_results.json"

K_SKNN = 200
BLEND_ALPHA = 0.2
KS = 20
LOGIN_BOOST = np.log(1.7)
FEATURES = (
    "logHist",
    "ad_ctr",
    "ip_ctr",
    "dev_ctr",
    "cat_ctr",
    "IPS_pos",
    "rank",
    "SKNCP",
    "content_score",
)


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), 1e-6))


def _z_params(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std() + 1e-9)


def _z(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return (x - mean) / sd


def _cv_positive_rate(scores: np.ndarray, y: np.ndarray) -> dict:
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y))
    fold_rates = [
        _best_f1_topk(scores[fold], y[fold])[1] / len(fold)
        for fold in np.array_split(idx, 5)
    ]
    return {
        "rate": float(np.mean(fold_rates)),
        "fold_rates": [float(x) for x in fold_rates],
    }


def _build_feature_matrix(
    rows: dict,
    ctr_maps: dict,
    pos_ctr: pd.Series,
    global_ctr: float,
    skncp: np.ndarray,
    content_score: np.ndarray,
) -> np.ndarray:
    ad_ctr = _lookup(rows["ad"], ctr_maps["ad"], global_ctr)
    ip_ctr = _lookup(rows["ip"], ctr_maps["ip"], global_ctr)
    dev_ctr = _lookup(rows["dev"], ctr_maps["dev"], global_ctr)
    cat_ctr = _lookup(rows["cat"], ctr_maps["cat"], global_ctr)
    pos_rate = _lookup(rows["pos"], pos_ctr, global_ctr)
    rank = _rank_feature(rows["sid"], rows["hist"])
    return np.column_stack(
        [
            _safe_log(rows["hist"]),
            _safe_log(ad_ctr),
            _safe_log(ip_ctr),
            _safe_log(dev_ctr),
            _safe_log(cat_ctr),
            _safe_log(pos_rate) - np.log(global_ctr),
            rank,
            skncp,
            content_score,
        ]
    ).astype(np.float64)


def main() -> None:
    if not CONTENT_SCORE_CACHE.exists():
        raise SystemExit(
            "Missing /tmp/cs_cache.npz. Run "
            "`python -X utf8 models/m04_gated/experiments/exp_content_strong.py` first."
        )

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = RecoDataset(DATASET_DIR).load()
    train = _collect_train(ds)
    external = _collect_external(ds)

    searchinfo = pd.read_csv(ds.dir / "searchinfo.csv")
    userinfo = pd.read_csv(ds.dir / "userinfo.csv")
    sid2ip = dict(zip(searchinfo["SearchID"], searchinfo["IPID"]))
    uid2dev = dict(zip(userinfo["UserID"], userinfo["UserDeviceID"]))
    for rows in (train, external):
        rows["ip"] = np.asarray([sid2ip.get(int(s), -1) for s in rows["sid"]])
        rows["dev"] = np.asarray([uid2dev.get(int(u), -1) for u in rows["uid"]])

    y = train["Y"]
    y_external = external["Y"]
    global_ctr = float(y.mean())

    unique_sids = np.sort(np.unique(train["sid"]))
    val_sids = set(unique_sids[int(len(unique_sids) * 0.8):].tolist())
    mask_val = np.asarray([sid in val_sids for sid in train["sid"]], dtype=bool)
    mask_train = ~mask_val
    y_val = y[mask_val]
    train_val_rows = {
        key: value[mask_val] if isinstance(value, np.ndarray) and len(value) == len(y) else value
        for key, value in train.items()
    }

    cache = np.load(CONTENT_SCORE_CACHE)
    content_val = cache["sva"].astype(np.float64)
    content_external = cache["se"].astype(np.float64)
    if len(content_val) != int(mask_val.sum()) or len(content_external) != len(y_external):
        raise SystemExit("Content cache shape does not match current dataset split.")

    keys = ("ad", "ip", "dev", "cat")
    ctr_train = {k: _ctr_series(train[k][mask_train], y[mask_train], global_ctr) for k in keys}
    ctr_full = {k: _ctr_series(train[k], y, global_ctr) for k in keys}
    pos_train = _ctr_series(train["pos"][mask_train], y[mask_train], global_ctr)
    pos_full = _ctr_series(train["pos"], y, global_ctr)

    index_train = _build_skncp_index(train, mask_train, ctr_train["ad"], global_ctr)
    index_full = _build_skncp_index(train, np.ones(len(y), dtype=bool), ctr_full["ad"], global_ctr)
    skncp_val, _ = _skncp_scores(train["Q"][mask_val], train["A"][mask_val], index_train, K_SKNN)
    skncp_external, _ = _skncp_scores(external["Q"], external["A"], index_full, K_SKNN)

    x_val = _build_feature_matrix(
        train_val_rows,
        ctr_train,
        pos_train,
        global_ctr,
        skncp_val,
        content_val,
    )
    x_external = _build_feature_matrix(
        external,
        ctr_full,
        pos_full,
        global_ctr,
        skncp_external,
        content_external,
    )
    base_val = LOGIN_BOOST * (1.0 - train["logged"][mask_val])
    base_external = LOGIN_BOOST * (1.0 - external["logged"])

    weights = _fit_f1_exponents(
        x_val,
        y_val,
        base=base_val,
        grid=np.arange(0.0, 3.05, 0.5),
        n_pass=5,
    )
    model6_val = x_val @ weights + base_val
    model6_external = x_external @ weights + base_external

    content_mean, content_sd = _z_params(content_val)
    model6_mean, model6_sd = _z_params(model6_val)
    boosted_val = _z(content_val, content_mean, content_sd) + BLEND_ALPHA * _z(
        model6_val, model6_mean, model6_sd
    )
    boosted_external = _z(
        content_external, content_mean, content_sd
    ) + BLEND_ALPHA * _z(model6_external, model6_mean, model6_sd)

    cv_rate = _cv_positive_rate(boosted_val, y_val)
    split_rate = _best_f1_topk(boosted_val, y_val)[1] / len(y_val)
    result = {
        "approach": "model6_boosted_skncp_content",
        "k_skncp": K_SKNN,
        "blend_alpha": BLEND_ALPHA,
        "features": list(FEATURES),
        "weights": {FEATURES[i]: float(weights[i]) for i in range(len(FEATURES))},
        "internal": {
            "auc": _binary_auc(boosted_val, y_val),
            "oracle_f1": _best_f1_topk(boosted_val, y_val)[0],
            "oracle_k": _best_f1_topk(boosted_val, y_val)[1],
            "cv_rate": cv_rate["rate"],
            "fold_rates": cv_rate["fold_rates"],
            "split_rate": float(split_rate),
        },
        "external": {
            "auc": _binary_auc(boosted_external, y_external),
            "oracle_f1": _best_f1_topk(boosted_external, y_external)[0],
            "oracle_k": _best_f1_topk(boosted_external, y_external)[1],
            "f1_prevalence": _f1_at_rate(boosted_external, y_external, global_ctr),
            "f1_cv_rate": _f1_at_rate(boosted_external, y_external, cv_rate["rate"]),
            "f1_split_rate": _f1_at_rate(boosted_external, y_external, split_rate),
            "bootstrap_prevalence": _bootstrap_f1(boosted_external, y_external, global_ctr),
        },
        "diagnostics": {
            "train_rows": int(len(y)),
            "train_clicks": int(y.sum()),
            "train_ctr": global_ctr,
            "external_rows": int(len(y_external)),
            "external_clicks": int(y_external.sum()),
            "external_ctr": float(y_external.mean()),
            "skncp_internal_auc": _binary_auc(skncp_val, y_val),
            "skncp_external_auc": _binary_auc(skncp_external, y_external),
        },
        "leak_audit": (
            "Weights use sorted internal 80/20 only. Internal CTR/SKNCP use "
            "internal-train counts/clicks. External CTR/SKNCP use full-train "
            "counts/clicks. Primary threshold uses train prevalence; external "
            "labels are used only for final validation and bootstrap."
        ),
        "elapsed_sec": float(time.time() - t0),
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
