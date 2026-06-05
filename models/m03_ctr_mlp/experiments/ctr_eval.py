"""
ctr_eval.py — GNN + CTR MLP 파이프라인 평가.

파이프라인
─────────────────────────────────────────────────────────────
  1. 데이터 로드
  2. 그래프 구축    (transductive=True)
  3. GNN 학습      (node repr 전파)
  4. CTR MLP 학습  (GNN repr + 스칼라 feature → BCE loss)
  5. Task A 평가   (CTR MLP score → F1)
  6. Task B 평가   (GNN score_ad_candidates → NDCG@3)
  7. 비교          Interest vs GNN-only vs GNN+MLP

실행:
    python -X utf8 experiments/ctr_eval.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.data.dataset import RecoDataset
from shared.data.graph import build_graph
from models.m02_gnn import GNNConfig, GNNModel
from models.m03_ctr_mlp import CTRConfig, CTRPredictor
from models.m01_interest import ModelConfig, MultiInterestModel
from shared.eval.predictor import (
    evaluate_task_a, evaluate_task_b_ndcg,
    score_task_a, score_task_b, train,
)

DATASET_DIR = ROOT / "../datasets"
SEED = 42


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.2f}s"


def fmt(v: float) -> str:
    return f"{v:.4f}"


def sweep_f1(scores, answers_df):
    """threshold sweep으로 best F1 반환."""
    best_f1, best_thr, best_m = 0.0, 0.5, None
    for thr in np.arange(0.05, 0.96, 0.05):
        m = evaluate_task_a(scores, answers_df, float(thr))
        if m["f1"] > best_f1:
            best_f1, best_thr, best_m = m["f1"], float(thr), m
    return best_f1, best_thr, best_m


def main() -> None:
    np.random.seed(SEED)

    # ── 1. 데이터 로드 ───────────────────────────────────────────────────
    section("1. 데이터 로드")
    ds = RecoDataset(DATASET_DIR).load()
    print(ds.summary())
    candidate_embs, candidate_ids = ds.all_ad_embs()
    val_ad_q    = ds.val_ad_queries()
    val_ad_ans  = ds.val_ad_answers()
    val_clk_q   = ds.val_click_queries()
    val_clk_ans = ds.val_click_answers()

    # ── 2. 그래프 구축 ───────────────────────────────────────────────────
    section("2. 그래프 구축  (transductive=True, include_test=True, top_k_sim=10)")
    t0 = time.time()
    graph = build_graph(ds, verbose=True, transductive=True,
                        include_test=True, top_k_sim=5)
    print(f"\n  build_graph: {elapsed(t0)}")

    # ── 3~5. click_weight sweep ──────────────────────────────────────────
    section("3. GNN + MLP (click_weight sweep)")
    ctr_cfg = CTRConfig(hidden_dim=128, n_epochs=30,
                        batch_size=1024, lr=3e-4,
                        focal_gamma=2.0, smooth_prior=10)

    # 최고 baseline: L=2, cw=5, top_k_sim=5 (hist_ctr 제외 9d 피처)
    experiments = [
        ("baseline-no-histctr (9d)",
         dict(n_layers=2, agg_fn="mean", click_weight=5.0,
              residual_alpha=0.0, user_click_init=False)),
    ]

    sweep_results = []
    for tag, gnn_kwargs in experiments:
        n_layers = gnn_kwargs["n_layers"]
        cw       = gnn_kwargs["click_weight"]
        print(f"\n  ── {tag} ──")
        gnn_cfg = GNNConfig(gamma=0.7, gamma_search=0.5, **gnn_kwargs)
        gnn = GNNModel(gnn_cfg)
        np.random.seed(SEED)
        gnn.fit(graph)

        ctr = CTRPredictor(ctr_cfg)
        ctr.fit(ds, gnn, val_pairs=val_clk_q, val_answers_df=val_clk_ans)

        sc_a  = ctr.score_pairs(val_clk_q)
        f1, thr, m_a = sweep_f1(sc_a, val_clk_ans)

        sc_b = score_task_b(gnn, val_ad_q, candidate_embs)
        m_b  = evaluate_task_b_ndcg(sc_b, val_ad_ans, candidate_ids)

        sweep_results.append((tag, f1, thr, m_a["auc"], m_b["ndcg@3"]))
        print(f"  [{tag}]  Task A F1={fmt(f1)} AUC={fmt(m_a['auc'])}  "
              f"Task B NDCG@3={fmt(m_b['ndcg@3'])}")

    # 마지막 실행된 gnn/ctr을 이후 비교용으로 유지
    gnn_time = "—"
    ctr_time = "—"

    # ── 6. Interest 모델 비교 ────────────────────────────────────────────
    section("6. 비교: Interest 모델")
    int_cfg   = ModelConfig(k=5, alpha_search=0.01, alpha_click=0.5,
                            gamma=0.7, gamma_search=0.5, threshold=0.5)
    int_model = MultiInterestModel(int_cfg)
    np.random.seed(SEED)
    train(int_model, ds.training_stream())
    sc_int_a = score_task_a(int_model, val_clk_q)
    f1_int, thr_int, m_int_a = sweep_f1(sc_int_a, val_clk_ans)
    sc_int_b = score_task_b(int_model, val_ad_q, candidate_embs)
    m_int_b  = evaluate_task_b_ndcg(sc_int_b, val_ad_ans, candidate_ids)

    # ── 최종 비교 테이블 ──────────────────────────────────────────────────
    section("7. 최종 결과 비교")
    print(f"\n  {'모델':<32}  {'Task A':^30}  {'Task B':^10}")
    print(f"  {'':32}  {'F1':>8} {'thr':>5} {'AUC':>8}  {'NDCG@3':>10}")
    print("  " + "─" * 78)

    best_f1 = max(r[1] for r in sweep_results)
    for tag, f1, thr, auc, ndcg in sweep_results:
        marker = " ◀" if f1 == best_f1 else ""
        print(f"  {tag:<32}  "
              f"{fmt(f1):>8} {thr:>5.2f} {fmt(auc):>8}  {fmt(ndcg):>10}{marker}")

    print(f"  {'Interest (γ=0.7/0.5)':<32}  "
          f"{fmt(f1_int):>8} {thr_int:>5.2f} {fmt(m_int_a['auc']):>8}  "
          f"{fmt(m_int_b['ndcg@3']):>10}")
    print(f"\n  HistCTR baseline  Task A F1=0.0436 / Task B NDCG@3=0.0211")


if __name__ == "__main__":
    main()
