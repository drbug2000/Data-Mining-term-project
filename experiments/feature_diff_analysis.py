"""
feature_diff_analysis.py — n_layers=2 vs 4, 유저 그룹별 F1 + sim_sa Gap 비교.

sim 없는 구성 (top_k_sim=0, click_weight=5) 기준.
실행: python -X utf8 experiments/feature_diff_analysis.py
"""
from __future__ import annotations
import sys, numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import RecoDataset
from data.graph import build_graph
from model.gnn import GNNConfig, GNNModel
from model.ctr_mlp import CTRConfig, CTRPredictor, _unit

DATASET_DIR = ROOT / "../datasets"
SEED = 42

FEAT_NAMES = [
    "sim_sa", "log_pos", "inv_pos", "is_top1", "hist_ctr",
    "cat_match", "search_cat_id", "ad_cat_id", "log_price", "is_logged_on",
]

def binary_auc(sc, lb):
    sc, lb = np.array(sc, dtype=float), np.array(lb, dtype=float)
    pos = sc[lb == 1]; neg = sc[lb == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(
        (np.sum(pos[:, None] > neg[None, :]) +
         0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))
    )

def best_f1_metrics(sc, lb):
    sc, lb = np.array(sc, dtype=float), np.array(lb, dtype=float)
    best_f1, best_thr, best_prec, best_rec = 0.0, 0.5, 0.0, 0.0
    for thr in np.arange(0.05, 0.96, 0.05):
        preds = (sc > 1 - thr).astype(int)
        tp = int(((preds == 1) & (lb == 1)).sum())
        fp = int(((preds == 1) & (lb == 0)).sum())
        fn = int(((preds == 0) & (lb == 1)).sum())
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        if f > best_f1:
            best_f1, best_thr, best_prec, best_rec = f, float(thr), p, r
    return best_f1, best_thr, best_prec, best_rec

def ugroup(uid, train_users, train_click_users):
    if uid not in train_users:        return "U0"
    if uid not in train_click_users:  return "U1"
    return                                   "U2"

GROUP_LABEL = {
    "U0": "① 신규 (cold-start)",
    "U1": "② 기존, 검색만",
    "U2": "③ 기존, 클릭 있음",
}

def sep(title=""):
    print(f"\n{'='*66}")
    if title: print(f"  {title}"); print(f"{'='*66}")


def run_experiment(ds, graph, val_pairs, val_ans, labels,
                   train_users, train_click_users,
                   n_layers: int) -> dict:
    """n_layers 설정으로 GNN+MLP 실행 후 그룹별 결과 반환."""
    np.random.seed(SEED)
    gnn = GNNModel(GNNConfig(n_layers=n_layers, agg_fn="mean",
                             gamma=0.7, gamma_search=0.5, click_weight=5.0))
    gnn.fit(graph)

    ctr = CTRPredictor(CTRConfig(hidden_dim=128, n_epochs=30, batch_size=1024,
                                  lr=3e-4, focal_gamma=2.0, smooth_prior=10))
    ctr.fit(ds, gnn, val_pairs=val_pairs, val_answers_df=val_ans)

    scores = ctr.score_pairs(val_pairs)

    # 그룹별 피처 + 점수 집계
    feat_g  = {"U0": [], "U1": [], "U2": []}
    score_g = {"U0": [], "U1": [], "U2": []}
    label_g = {"U0": [], "U1": [], "U2": []}
    drift_g = {"U0": [], "U1": [], "U2": []}

    for (ev, ad), s, l in zip(val_pairs, scores, labels):
        g = ugroup(ev.user_id, train_users, train_click_users)
        f = ctr._feat_raw(ev, ad)
        feat_g[g].append(f if f is not None else np.zeros(len(FEAT_NAMES), np.float32))
        score_g[g].append(s)
        label_g[g].append(int(l))

        # drift 계산
        si = gnn._search_id_to_idx.get(ev.search_id)
        if si is not None:
            h_raw = _unit(gnn._search_feat_raw[si])
            h_gnn = _unit(gnn._search_repr[si])
            drift_g[g].append(float(1 - h_raw @ h_gnn))

    # 그룹별 메트릭 계산
    out = {}
    for gk in ["U0", "U1", "U2"]:
        X  = np.array(feat_g[gk],  dtype=np.float32)
        sc = score_g[gk]; lb = label_g[gk]
        n  = len(lb); pos = sum(lb)

        f1, thr, prec, rec = best_f1_metrics(sc, lb)
        auc = binary_auc(sc, lb)

        # sim_sa gap
        sa  = X[:, 0]; lba = np.array(lb, dtype=float)
        mc  = sa[lba == 1].mean() if pos > 0 else float("nan")
        mn  = sa[lba == 0].mean()
        gap = mc - mn if pos > 0 else float("nan")

        out[gk] = {
            "n": n, "pos": pos,
            "f1": f1, "thr": thr, "prec": prec, "rec": rec, "auc": auc,
            "sim_sa_click": mc, "sim_sa_noclk": mn, "sim_sa_gap": gap,
            "drift_mean": np.mean(drift_g[gk]) if drift_g[gk] else float("nan"),
        }

    # 전체
    all_sc = [s for g in score_g.values() for s in g]
    all_lb = [l for g in label_g.values() for l in g]
    f1_all, thr_all, prec_all, rec_all = best_f1_metrics(all_sc, all_lb)
    out["ALL"] = {
        "n": len(all_lb), "pos": sum(all_lb),
        "f1": f1_all, "thr": thr_all, "prec": prec_all, "rec": rec_all,
        "auc": binary_auc(all_sc, all_lb),
        "sim_sa_gap": float("nan"), "drift_mean": float("nan"),
    }
    return out


def main():
    np.random.seed(SEED)
    print("[1] 데이터 로드...")
    ds = RecoDataset(DATASET_DIR).load()

    train_users, train_click_users = set(), set()
    for ev in ds.training_stream():
        train_users.add(ev.user_id)
        for ad in ev.ads:
            if ad.is_click: train_click_users.add(ev.user_id)

    graph     = build_graph(ds, verbose=False, transductive=True,
                            include_test=True, top_k_sim=0)
    val_pairs = ds.val_click_queries()
    val_ans   = ds.val_click_answers()
    labels    = val_ans["IsClick"].tolist()[:len(val_pairs)]

    print(f"\n  훈련 유저 {len(train_users):,}  "
          f"(클릭:{len(train_click_users):,} / 검색만:{len(train_users-train_click_users):,})")

    # ── 실험 실행 ──────────────────────────────────────────────────────
    results = {}
    for nl in [2, 4]:
        print(f"\n[2] n_layers={nl} 실행...")
        results[nl] = run_experiment(ds, graph, val_pairs, val_ans, labels,
                                     train_users, train_click_users, nl)

    # ── 결과 테이블 ────────────────────────────────────────────────────
    sep("그룹별 F1 / AUC / sim_sa Gap 비교 (L=2 vs L=4)")

    keys = ["U0", "U1", "U2", "ALL"]
    col  = 18

    # 헤더
    print(f"\n  {'그룹':<26}  {'--- L=2 ---':^44}  {'--- L=4 ---':^44}")
    print(f"  {'':26}  "
          f"{'F1':>7} {'AUC':>7} {'Prec':>7} {'Rec':>7} {'Gap':>8}  "
          f"{'F1':>7} {'AUC':>7} {'Prec':>7} {'Rec':>7} {'Gap':>8}")
    print("  " + "─" * 115)

    for gk in keys:
        label = GROUP_LABEL.get(gk, "전체")
        r2, r4 = results[2][gk], results[4][gk]
        n, pos = r2["n"], r2["pos"]

        def fmt(v): return f"{v:.4f}" if not (v != v) else "  nan "

        # F1이 개선된 경우 표시
        mark2 = ""
        mark4 = " ◀" if r4["f1"] > r2["f1"] else ""

        print(f"  {label:<26}  "
              f"{fmt(r2['f1']):>7} {fmt(r2['auc']):>7} "
              f"{fmt(r2['prec']):>7} {fmt(r2['rec']):>7} {fmt(r2['sim_sa_gap']):>8}  "
              f"{fmt(r4['f1']):>7} {fmt(r4['auc']):>7} "
              f"{fmt(r4['prec']):>7} {fmt(r4['rec']):>7} {fmt(r4['sim_sa_gap']):>8}"
              f"{mark4}")

        if gk == "U2":
            print("  " + "─" * 115)

    # ── sim_sa drift 비교 ──────────────────────────────────────────────
    sep("h_search drift (원본 text emb에서 변형 정도)")
    print(f"\n  {'그룹':<26}  {'L=2 drift':>12}  {'L=4 drift':>12}  {'변화':>8}")
    print("  " + "─" * 64)
    for gk in ["U0", "U1", "U2"]:
        d2 = results[2][gk]["drift_mean"]
        d4 = results[4][gk]["drift_mean"]
        delta = d4 - d2 if not (d2 != d2 or d4 != d4) else float("nan")
        print(f"  {GROUP_LABEL[gk]:<26}  {d2:>12.4f}  {d4:>12.4f}  {delta:>+8.4f}")

    print("""
  drift = 1 - cosine(h_search_raw, h_search_gnn)
    0 ≈ GNN 전파 후에도 원본 text emb 유지
    1 ≈ 원본에서 크게 멀어짐 (click/user 정보 강하게 반영)
""")


if __name__ == "__main__":
    main()
