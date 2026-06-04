"""
group_analysis.py — 유저 그룹별 Task A 성능 분석.

그룹 정의:
  U0: 훈련에 등장하지 않은 신규 유저 (cold-start)
  U1: 훈련에 검색 이력만 있는 유저 (클릭 없음)
  U2: 훈련에 클릭 이력까지 있는 유저

실행: python -X utf8 experiments/group_analysis.py
"""
from __future__ import annotations
import sys, numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.data.dataset import RecoDataset
from shared.data.graph import build_graph
from models.m02_gnn import GNNConfig, GNNModel
from models.m03_ctr_mlp import CTRConfig, CTRPredictor

DATASET_DIR = ROOT / "../datasets"
SEED = 42

# ── 헬퍼 ──────────────────────────────────────────────────────────────────

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
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        if f > best_f1:
            best_f1, best_thr, best_prec, best_rec = f, float(thr), p, r
    return best_f1, best_thr, best_prec, best_rec


def sep(title=""):
    print(f"\n{'='*62}")
    if title:
        print(f"  {title}")
        print(f"{'='*62}")


# ── 메인 ──────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)

    # 1. 데이터 로드
    print("[1] 데이터 로드 및 훈련 통계 집계...")
    ds = RecoDataset(DATASET_DIR).load()

    train_users       : set[int] = set()
    train_click_users : set[int] = set()

    for ev in ds.training_stream():
        train_users.add(ev.user_id)
        for ad in ev.ads:
            if ad.is_click:
                train_click_users.add(ev.user_id)

    print(f"  전체 훈련 유저     : {len(train_users):,}")
    print(f"  클릭 이력 있는 유저: {len(train_click_users):,}")
    print(f"  검색만 있는 유저   : {len(train_users - train_click_users):,}")

    # 2. 그래프 + GNN + MLP (현재 최고 baseline)
    print("\n[2] 그래프 구축 + GNN + MLP 학습 (baseline 구성)...")
    graph = build_graph(ds, verbose=False, transductive=True,
                        include_test=True, top_k_sim=0)

    val_pairs = ds.val_click_queries()
    val_ans   = ds.val_click_answers()

    gnn_cfg = GNNConfig(n_layers=2, agg_fn="mean", gamma=0.7,
                        gamma_search=0.5, click_weight=5.0,
                        residual_alpha=0.0, user_click_init=False)
    gnn = GNNModel(gnn_cfg)
    np.random.seed(SEED)
    gnn.fit(graph)

    ctr_cfg = CTRConfig(hidden_dim=128, n_epochs=30, batch_size=1024,
                        lr=3e-4, focal_gamma=2.0, smooth_prior=10)
    ctr = CTRPredictor(ctr_cfg)
    ctr.fit(ds, gnn, val_pairs=val_pairs, val_answers_df=val_ans)

    # 3. 점수 계산
    print("\n[3] 평가...")
    scores = ctr.score_pairs(val_pairs)
    labels = val_ans["IsClick"].tolist()[:len(val_pairs)]

    # 4. 유저 그룹 분류
    def user_group(uid):
        if uid not in train_users:       return "U0"
        if uid not in train_click_users: return "U1"
        return                                   "U2"

    GROUP_LABEL = {
        "U0": "① 신규 유저     (훈련 미등장)",
        "U1": "② 기존 유저     (검색만, 클릭 없음)",
        "U2": "③ 기존 유저     (클릭 이력 있음)",
    }

    groups: dict[str, dict] = {
        g: {"scores": [], "labels": []} for g in ["U0", "U1", "U2"]
    }
    for (ev, ad), s, l in zip(val_pairs, scores, labels):
        g = user_group(ev.user_id)
        groups[g]["scores"].append(s)
        groups[g]["labels"].append(int(l))

    # 5. 출력
    sep("유저 그룹별 Task A 성능 (sim 엣지 없음, click_weight=5)")
    total   = len(labels)
    total_p = sum(labels)

    fmt = lambda v: f"{v:.4f}"

    print(f"\n  {'그룹':<38} {'샘플':>6} {'비율':>6}  {'클릭':>4}  {'CTR':>6}  "
          f"{'F1':>7}  {'thr':>5}  {'Prec':>7}  {'Rec':>7}  {'AUC':>7}")
    print("  " + "─" * 98)

    for gk in ["U0", "U1", "U2"]:
        d   = groups[gk]
        n   = len(d["labels"])
        pos = sum(d["labels"])
        ctr_v = pos / max(n, 1)
        auc   = binary_auc(d["scores"], d["labels"])
        f1, thr, prec, rec = best_f1_metrics(d["scores"], d["labels"])

        print(f"  {GROUP_LABEL[gk]:<38} {n:>6,} {n/total:>6.1%}  {pos:>4}  "
              f"{ctr_v:>6.2%}  {fmt(f1):>7}  {thr:>5.2f}  "
              f"{fmt(prec):>7}  {fmt(rec):>7}  {fmt(auc):>7}")

    print("  " + "─" * 98)
    auc_all = binary_auc(scores, labels)
    f1_all, thr_all, prec_all, rec_all = best_f1_metrics(scores, labels)
    print(f"  {'전체':<38} {total:>6,} {'100%':>6}  {total_p:>4}  "
          f"{total_p/total:>6.2%}  {fmt(f1_all):>7}  {thr_all:>5.2f}  "
          f"{fmt(prec_all):>7}  {fmt(rec_all):>7}  {fmt(auc_all):>7}")

    # 6. 추가 인사이트: 그룹별 score 분포
    sep("그룹별 score 분포 (클릭 vs 비클릭)")
    for gk in ["U0", "U1", "U2"]:
        d  = groups[gk]
        sc = np.array(d["scores"]); lb = np.array(d["labels"])
        pos_sc = sc[lb == 1]; neg_sc = sc[lb == 0]
        print(f"\n  {GROUP_LABEL[gk]}")
        if len(pos_sc) > 0:
            print(f"    클릭(n={len(pos_sc):3d})  mean={pos_sc.mean():.4f}  "
                  f"std={pos_sc.std():.4f}  min={pos_sc.min():.4f}  max={pos_sc.max():.4f}")
        else:
            print(f"    클릭 샘플 없음")
        print(f"    비클릭(n={len(neg_sc):,}) mean={neg_sc.mean():.4f}  "
              f"std={neg_sc.std():.4f}  min={neg_sc.min():.4f}  max={neg_sc.max():.4f}")
        if len(pos_sc) > 0:
            gap = pos_sc.mean() - neg_sc.mean()
            print(f"    Gap (click - noclk): {gap:+.4f}")


if __name__ == "__main__":
    main()
