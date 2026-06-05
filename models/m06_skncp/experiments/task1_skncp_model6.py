"""
task1_skncp_model6.py — model6 SKNCP Task1 EDA, tuning, fitting, validation.

Task1 is interpreted as click prediction over (SearchID, AdID) rows. Model
selection uses only a sorted SearchID 80/20 internal split. External validation
labels are used only for final scoring, bootstrap CI, and reporting.

Run:
    python -X utf8 models/m06_skncp/experiments/task1_skncp_model6.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.m04_gated.gated_ctr import (  # noqa: E402
    _best_f1_topk,
    _binary_auc,
    _fit_f1_exponents,
    _l2_normalize,
)
from shared.data.dataset import RecoDataset  # noqa: E402

DATASET_DIR = ROOT / "../datasets"
OUT_DIR = ROOT / "models/m06_skncp/results"
OUT_JSON = OUT_DIR / "model6_task1_results.json"

KS = 20
LOGIN_BOOST = np.log(1.7)
K_CANDIDATES = (20, 50, 100, 200, 400)
FEATURE_NAMES = (
    "logHist",
    "ad_ctr",
    "ip_ctr",
    "dev_ctr",
    "cat_ctr",
    "IPS_pos",
    "rank",
    "SKNCP",
    "raw_cos",
    "knn_adctr_mean",
)
SUBSETS = {
    "skncp_only": ("SKNCP",),
    "hist_only": ("logHist",),
    "5ctr": ("logHist", "ad_ctr", "ip_ctr", "dev_ctr", "cat_ctr"),
    "koskncp_core": (
        "logHist",
        "ad_ctr",
        "ip_ctr",
        "dev_ctr",
        "cat_ctr",
        "IPS_pos",
        "rank",
        "SKNCP",
    ),
    "model6_all_regularized": FEATURE_NAMES,
    "model6_no_rawcos": tuple(f for f in FEATURE_NAMES if f != "raw_cos"),
}


def _section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), 1e-6))


def _cohens_d(scores: np.ndarray, y: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y)
    sd = float(scores.std() + 1e-12)
    return float((scores[y == 1].mean() - scores[y == 0].mean()) / sd)


def _f1_at_rate(scores: np.ndarray, y: np.ndarray, rate: float) -> dict:
    k = max(1, int(round(float(rate) * len(scores))))
    order = np.argsort(-scores)
    pred = np.zeros(len(scores), dtype=np.int8)
    pred[order[:k]] = 1
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "k": int(k),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _bootstrap_f1(
    scores: np.ndarray,
    y: np.ndarray,
    rate: float,
    seed: int = 1,
    n_boot: int = 2000,
) -> dict:
    rng = np.random.RandomState(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        vals.append(_f1_at_rate(scores[idx], y[idx], rate)["f1"])
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci95": [
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
        ],
        "n_boot": int(n_boot),
    }


def _collect_train(ds: RecoDataset) -> dict:
    q, a, y = [], [], []
    ad, cat, uid, sid, hist, logged, pos, price = [], [], [], [], [], [], [], []
    for ev in ds.training_stream():
        for rec in ev.ads:
            q.append(ev.search_emb)
            a.append(rec.ad_emb)
            y.append(int(rec.is_click))
            ad.append(int(rec.ad_id))
            cat.append(int(rec.category_id))
            uid.append(int(ev.user_id))
            sid.append(int(ev.search_id))
            hist.append(float(rec.hist_ctr) if rec.hist_ctr is not None else 0.0)
            logged.append(float(ev.is_logged_on))
            pos.append(int(rec.position))
            price.append(float(getattr(rec, "price", 0.0) or 0.0))
    return {
        "Q": _l2_normalize(np.asarray(q, np.float32)),
        "A": _l2_normalize(np.asarray(a, np.float32)),
        "Y": np.asarray(y, np.int8),
        "ad": np.asarray(ad),
        "cat": np.asarray(cat),
        "uid": np.asarray(uid),
        "sid": np.asarray(sid),
        "hist": np.asarray(hist, np.float64),
        "logged": np.asarray(logged, np.float64),
        "pos": np.asarray(pos, np.float64),
        "price": np.asarray(price, np.float64),
    }


def _collect_external(ds: RecoDataset) -> dict:
    pairs = ds.val_click_queries()
    y = ds.val_click_answers()["IsClick"].to_numpy()[: len(pairs)].astype(np.int8)
    return {
        "Q": _l2_normalize(np.asarray([ev.search_emb for ev, _ in pairs], np.float32)),
        "A": _l2_normalize(np.asarray([rec.ad_emb for _, rec in pairs], np.float32)),
        "Y": y,
        "ad": np.asarray([int(rec.ad_id) for _, rec in pairs]),
        "cat": np.asarray([int(rec.category_id) for _, rec in pairs]),
        "uid": np.asarray([int(ev.user_id) for ev, _ in pairs]),
        "sid": np.asarray([int(ev.search_id) for ev, _ in pairs]),
        "hist": np.asarray([float(rec.hist_ctr) for _, rec in pairs], np.float64),
        "logged": np.asarray([float(ev.is_logged_on) for ev, _ in pairs], np.float64),
        "pos": np.asarray([int(rec.position) for _, rec in pairs], np.float64),
        "price": np.asarray([float(getattr(rec, "price", 0.0) or 0.0) for _, rec in pairs], np.float64),
    }


def _ctr_series(keys: np.ndarray, y: np.ndarray, global_ctr: float) -> pd.Series:
    df = pd.DataFrame({"k": keys, "y": y})
    gp = df.groupby("k")["y"].agg(["sum", "count"])
    return (gp["sum"] + KS * global_ctr) / (gp["count"] + KS)


def _lookup(keys: np.ndarray, series: pd.Series, default: float) -> np.ndarray:
    return pd.Series(keys).map(series).fillna(default).to_numpy(dtype=np.float64)


def _rank_feature(search_ids: np.ndarray, hist: np.ndarray) -> np.ndarray:
    rank = pd.DataFrame({"sid": search_ids, "hist": hist}).groupby("sid")["hist"].rank(pct=True)
    return 0.5 - rank.to_numpy(dtype=np.float64)


def _build_skncp_index(train: dict, mask: np.ndarray, ad_ctr_map: pd.Series, global_ctr: float) -> dict:
    click_idx = np.flatnonzero(mask & (train["Y"] == 1))
    clicked_ad_ctr = _lookup(train["ad"][click_idx], ad_ctr_map, global_ctr)
    return {
        "Q": train["Q"][click_idx].astype(np.float32),
        "A": train["A"][click_idx].astype(np.float32),
        "ad_ctr": clicked_ad_ctr.astype(np.float32),
        "n_clicks": int(len(click_idx)),
    }


def _skncp_scores(
    q: np.ndarray,
    a: np.ndarray,
    index: dict,
    k: int,
    block_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (SKNCP max clicked-ad cosine, mean ad_ctr of K nearest clicked ads)."""
    n = len(q)
    if index["n_clicks"] == 0:
        return np.zeros(n, np.float32), np.zeros(n, np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_all = torch.tensor(q, dtype=torch.float32, device=device)
    a_all = torch.tensor(a, dtype=torch.float32, device=device)
    tq = torch.tensor(index["Q"], dtype=torch.float32, device=device)
    ta = torch.tensor(index["A"], dtype=torch.float32, device=device)
    t_ctr = torch.tensor(index["ad_ctr"], dtype=torch.float32, device=device)

    out = np.zeros(n, np.float32)
    nbr = np.zeros(n, np.float32)
    k_eff = min(int(k), tq.shape[0])
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        sims = q_all[start:end] @ tq.T
        top = sims.topk(k_eff, dim=1).indices
        pools = ta[top]
        target = a_all[start:end].unsqueeze(1)
        out[start:end] = (pools * target).sum(dim=2).max(dim=1).values.detach().cpu().numpy()
        nbr[start:end] = t_ctr[top].mean(dim=1).detach().cpu().numpy()
    return out, nbr


def _build_features(
    rows: dict,
    ctr_maps: dict,
    pos_ctr: pd.Series,
    global_ctr: float,
    skncp: np.ndarray,
    knn_adctr_mean: np.ndarray,
) -> np.ndarray:
    raw_cos = (rows["Q"] * rows["A"]).sum(axis=1)
    ad_ctr = _lookup(rows["ad"], ctr_maps["ad"], global_ctr)
    ip_ctr = _lookup(rows["ip"], ctr_maps["ip"], global_ctr)
    dev_ctr = _lookup(rows["dev"], ctr_maps["dev"], global_ctr)
    cat_ctr = _lookup(rows["cat"], ctr_maps["cat"], global_ctr)
    pos_rate = _lookup(rows["pos"], pos_ctr, global_ctr)
    ips_pos = _safe_log(pos_rate) - np.log(global_ctr)
    rank = _rank_feature(rows["sid"], rows["hist"])
    feats = np.column_stack(
        [
            _safe_log(rows["hist"]),
            _safe_log(ad_ctr),
            _safe_log(ip_ctr),
            _safe_log(dev_ctr),
            _safe_log(cat_ctr),
            ips_pos,
            rank,
            skncp,
            raw_cos,
            _safe_log(knn_adctr_mean),
        ]
    )
    return feats.astype(np.float64)


def _subset_indices(names: tuple[str, ...]) -> list[int]:
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    return [idx[name] for name in names]


def _fit_eval_subset(
    name: str,
    cols: list[int],
    x_va: np.ndarray,
    y_va: np.ndarray,
    base_va: np.ndarray,
    x_ex: np.ndarray,
    y_ex: np.ndarray,
    base_ex: np.ndarray,
    global_ctr: float,
    grid: np.ndarray,
) -> dict:
    w = _fit_f1_exponents(x_va[:, cols], y_va, base=base_va, grid=grid, n_pass=5)
    s_va = x_va[:, cols] @ w + base_va
    s_ex = x_ex[:, cols] @ w + base_ex

    best_internal_f1, best_internal_k = _best_f1_topk(s_va, y_va)
    split_rate = best_internal_k / len(y_va)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y_va))
    fold_rates = [
        _best_f1_topk(s_va[fold], y_va[fold])[1] / len(fold)
        for fold in np.array_split(idx, 5)
    ]
    cv_rate = float(np.mean(fold_rates))
    oracle_f1, oracle_k = _best_f1_topk(s_ex, y_ex)

    return {
        "name": name,
        "features": [FEATURE_NAMES[i] for i in cols],
        "grid_min": float(np.min(grid)),
        "weights": {FEATURE_NAMES[cols[i]]: float(w[i]) for i in range(len(cols))},
        "internal": {
            "f1": float(best_internal_f1),
            "auc": _binary_auc(s_va, y_va),
            "best_k": int(best_internal_k),
            "split_rate": float(split_rate),
            "cv_rate": cv_rate,
            "fold_rates": [float(x) for x in fold_rates],
        },
        "external": {
            "f1_prevalence": _f1_at_rate(s_ex, y_ex, global_ctr),
            "f1_cv_rate": _f1_at_rate(s_ex, y_ex, cv_rate),
            "f1_split_rate": _f1_at_rate(s_ex, y_ex, split_rate),
            "oracle_f1": float(oracle_f1),
            "oracle_k": int(oracle_k),
            "auc": _binary_auc(s_ex, y_ex),
        },
        "scores_external": s_ex,
    }


def _as_jsonable_result(result: dict) -> dict:
    cleaned = {}
    for key, value in result.items():
        if key == "scores_external":
            continue
        if isinstance(value, dict):
            cleaned[key] = _as_jsonable_result(value)
        elif isinstance(value, np.ndarray):
            cleaned[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("MKL_NUM_THREADS", "6")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _section("Load")
    ds = RecoDataset(DATASET_DIR).load()
    train = _collect_train(ds)
    external = _collect_external(ds)
    sinfo = pd.read_csv(ds.dir / "searchinfo.csv")
    uinfo = pd.read_csv(ds.dir / "userinfo.csv")
    sid2ip = dict(zip(sinfo["SearchID"], sinfo["IPID"]))
    uid2dev = dict(zip(uinfo["UserID"], uinfo["UserDeviceID"]))

    for rows in (train, external):
        rows["ip"] = np.asarray([sid2ip.get(int(s), -1) for s in rows["sid"]])
        rows["dev"] = np.asarray([uid2dev.get(int(u), -1) for u in rows["uid"]])

    y = train["Y"]
    y_ex = external["Y"]
    global_ctr = float(y.mean())
    uniq_sid = np.sort(np.unique(train["sid"]))
    cut = int(len(uniq_sid) * 0.8)
    val_sids = set(uniq_sid[cut:].tolist())
    mask_va = np.asarray([sid in val_sids for sid in train["sid"]], dtype=bool)
    mask_tr = ~mask_va
    y_va = y[mask_va]
    print(f"train_rows={len(y):,} train_ctr={global_ctr:.6f} clicks={int(y.sum())}")
    print(f"internal_train={int(mask_tr.sum()):,} internal_val={int(mask_va.sum()):,}")
    print(f"external_rows={len(y_ex):,} external_ctr={float(y_ex.mean()):.6f} clicks={int(y_ex.sum())}")

    train_users = set(train["uid"].tolist())
    external_unseen_user = float(np.mean([u not in train_users for u in external["uid"]]))
    external_sid_overlap = len(set(train["sid"]) & set(external["sid"]))
    ads_per_search = pd.Series(train["sid"]).value_counts().value_counts(normalize=True).sort_index()

    _section("EDA")
    hist_auc_va = _binary_auc(train["hist"][mask_va], y_va)
    hist_auc_ex = _binary_auc(external["hist"], y_ex)
    raw_va = (train["Q"][mask_va] * train["A"][mask_va]).sum(axis=1)
    raw_ex = (external["Q"] * external["A"]).sum(axis=1)
    raw_auc_va = _binary_auc(raw_va, y_va)
    raw_auc_ex = _binary_auc(raw_ex, y_ex)
    pos_ctr_train = pd.DataFrame({"pos": train["pos"], "y": y}).groupby("pos")["y"].mean()
    login_ctr = pd.DataFrame({"logged": train["logged"], "y": y}).groupby("logged")["y"].mean()
    hist_bins = pd.cut(train["hist"], [0, 0.005, 0.02, 0.05, 0.1, 1.0])
    hist_bin_ctr = (
        pd.DataFrame({"bin": hist_bins, "y": y})
        .groupby("bin", observed=True)["y"]
        .mean()
        .astype(float)
    )
    eda = {
        "train_rows": int(len(y)),
        "train_clicks": int(y.sum()),
        "train_ctr": global_ctr,
        "external_rows": int(len(y_ex)),
        "external_clicks": int(y_ex.sum()),
        "external_ctr": float(y_ex.mean()),
        "external_searchid_overlap_count": int(external_sid_overlap),
        "external_unseen_user_rate": external_unseen_user,
        "ads_per_search_distribution": {str(int(k)): float(v) for k, v in ads_per_search.items()},
        "hist_auc_internal_val": float(hist_auc_va),
        "hist_auc_external": float(hist_auc_ex),
        "raw_cos_auc_internal_val": float(raw_auc_va),
        "raw_cos_auc_external": float(raw_auc_ex),
        "raw_cos_d_internal_val": _cohens_d(raw_va, y_va),
        "raw_cos_d_external": _cohens_d(raw_ex, y_ex),
        "position_ctr_train": {str(int(k)): float(v) for k, v in pos_ctr_train.items()},
        "login_ctr_train": {str(int(k)): float(v) for k, v in login_ctr.items()},
        "hist_bin_ctr_train": {str(k): float(v) for k, v in hist_bin_ctr.items()},
    }
    print(json.dumps(eda, ensure_ascii=False, indent=2))

    _section("Leak-safe priors")
    keys = ("ad", "ip", "dev", "cat")
    ctr_tr = {k: _ctr_series(train[k][mask_tr], y[mask_tr], global_ctr) for k in keys}
    ctr_full = {k: _ctr_series(train[k], y, global_ctr) for k in keys}
    pos_tr = _ctr_series(train["pos"][mask_tr], y[mask_tr], global_ctr)
    pos_full = _ctr_series(train["pos"], y, global_ctr)
    idx_internal = _build_skncp_index(train, mask_tr, ctr_tr["ad"], global_ctr)
    idx_full = _build_skncp_index(train, np.ones(len(y), dtype=bool), ctr_full["ad"], global_ctr)
    print(f"internal_clicked_index={idx_internal['n_clicks']:,}")
    print(f"full_clicked_index={idx_full['n_clicks']:,}")

    base_va = LOGIN_BOOST * (1.0 - train["logged"][mask_va])
    base_ex = LOGIN_BOOST * (1.0 - external["logged"])

    _section("SKNCP K sweep and model fitting")
    k_results = {}
    all_results = []
    for k in K_CANDIDATES:
        t_k = time.time()
        sk_va, nbr_va = _skncp_scores(train["Q"][mask_va], train["A"][mask_va], idx_internal, k)
        sk_ex, nbr_ex = _skncp_scores(external["Q"], external["A"], idx_full, k)
        x_va = _build_features(
            {key: value[mask_va] if isinstance(value, np.ndarray) and len(value) == len(y) else value
             for key, value in train.items()},
            ctr_tr,
            pos_tr,
            global_ctr,
            sk_va,
            nbr_va,
        )
        x_ex = _build_features(external, ctr_full, pos_full, global_ctr, sk_ex, nbr_ex)
        sk_metrics = {
            "auc_internal_val": _binary_auc(sk_va, y_va),
            "auc_external": _binary_auc(sk_ex, y_ex),
            "d_internal_val": _cohens_d(sk_va, y_va),
            "d_external": _cohens_d(sk_ex, y_ex),
            "external_prevalence_f1": _f1_at_rate(sk_ex, y_ex, global_ctr),
        }

        subset_results = []
        for subset_name, subset_features in SUBSETS.items():
            cols = _subset_indices(subset_features)
            grid = np.arange(0.5, 3.05, 0.5)
            if subset_name == "model6_all_regularized":
                grid = np.arange(0.0, 3.05, 0.5)
            result = _fit_eval_subset(
                subset_name,
                cols,
                x_va,
                y_va,
                base_va,
                x_ex,
                y_ex,
                base_ex,
                global_ctr,
                grid,
            )
            result["k"] = int(k)
            subset_results.append(result)
            all_results.append(result)

        best_internal = max(subset_results, key=lambda r: r["internal"]["f1"])
        best_external_prev = max(
            subset_results,
            key=lambda r: r["external"]["f1_prevalence"]["f1"],
        )
        k_results[str(k)] = {
            "skncp": sk_metrics,
            "best_by_internal": _as_jsonable_result(best_internal),
            "best_by_external_prevalence_report_only": _as_jsonable_result(best_external_prev),
            "subsets": [_as_jsonable_result(r) for r in subset_results],
            "elapsed_sec": float(time.time() - t_k),
        }
        print(
            f"K={k:3d} sk_auc_va={sk_metrics['auc_internal_val']:.4f} "
            f"sk_auc_ex={sk_metrics['auc_external']:.4f} "
            f"best_internal={best_internal['name']}:{best_internal['internal']['f1']:.4f} "
            f"ext_prev={best_internal['external']['f1_prevalence']['f1']:.4f} "
            f"best_ext_prev_report={best_external_prev['name']}:"
            f"{best_external_prev['external']['f1_prevalence']['f1']:.4f} "
            f"({time.time() - t_k:.1f}s)"
        )

    selected = max(all_results, key=lambda r: r["internal"]["f1"])
    best_validation_cv = max(all_results, key=lambda r: r["external"]["f1_cv_rate"]["f1"])
    best_validation_prev = max(all_results, key=lambda r: r["external"]["f1_prevalence"]["f1"])

    selected_clean = _as_jsonable_result(selected)
    selected_clean["selection_rule"] = "max internal-val F1 over K/subset; external labels not used"
    selected_clean["bootstrap_prevalence_rate"] = _bootstrap_f1(
        selected["scores_external"], y_ex, global_ctr
    )
    selected_clean["bootstrap_cv_rate"] = _bootstrap_f1(
        selected["scores_external"], y_ex, selected["internal"]["cv_rate"]
    )

    best_validation_cv_clean = _as_jsonable_result(best_validation_cv)
    best_validation_cv_clean["selection_rule"] = (
        "report-only: max external F1 at internal CV-derived positive rate"
    )
    best_validation_cv_clean["bootstrap_cv_rate"] = _bootstrap_f1(
        best_validation_cv["scores_external"],
        y_ex,
        best_validation_cv["internal"]["cv_rate"],
    )
    best_validation_cv_clean["bootstrap_split_rate"] = _bootstrap_f1(
        best_validation_cv["scores_external"],
        y_ex,
        best_validation_cv["internal"]["split_rate"],
    )
    best_validation_cv_clean["bootstrap_prevalence_rate"] = _bootstrap_f1(
        best_validation_cv["scores_external"], y_ex, global_ctr
    )

    best_validation_prev_clean = _as_jsonable_result(best_validation_prev)
    best_validation_prev_clean["selection_rule"] = (
        "report-only: max external F1 at train-prevalence positive rate"
    )
    best_validation_prev_clean["bootstrap_prevalence_rate"] = _bootstrap_f1(
        best_validation_prev["scores_external"], y_ex, global_ctr
    )

    result_json = {
        "approach": "model6_skncp_task1",
        "task": "Task1 click prediction",
        "dataset_dir": str(DATASET_DIR),
        "eda": eda,
        "split": {
            "mode": "sorted SearchID 80/20",
            "internal_train_rows": int(mask_tr.sum()),
            "internal_val_rows": int(mask_va.sum()),
            "internal_train_clicks": int(y[mask_tr].sum()),
            "internal_val_clicks": int(y_va.sum()),
            "global_ctr_threshold_rate": global_ctr,
        },
        "k_results": k_results,
        "selected_by_internal": selected_clean,
        "best_validation_cv_rate": best_validation_cv_clean,
        "best_validation_prevalence_rate": best_validation_prev_clean,
        "selected": selected_clean,
        "leak_audit": (
            "Feature/K/subset/weight selection uses only sorted internal 80/20. "
            "Internal-val CTR/SKNCP indexes use internal-train clicks/counts. "
            "External scoring uses full-train clicks/counts. "
            "External labels are used only for final validation metrics and bootstrap CI."
        ),
        "elapsed_sec": float(time.time() - t0),
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    _section("Selected")
    print(json.dumps(selected_clean, ensure_ascii=False, indent=2))
    _section("Best validation CV-rate (report-only)")
    print(json.dumps(best_validation_cv_clean, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(f"elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
