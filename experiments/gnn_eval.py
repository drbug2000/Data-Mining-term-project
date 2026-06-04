"""
gnn_eval.py — GNN 모델 학습 및 평가.

파이프라인
──────────────────────────────────────────────────────────────
  1. 데이터 로드         : RecoDataset
  2. 그래프 구축         : build_graph(ds)  → HeteroGraph
  3. GNN 학습 단계       : GNNModel.fit(graph)
       - hop-1: search repr ← clicked ad embeddings (mean/max/sum)
       - hop-2: user repr   ← search reprs
  4. 추론 단계           : score_task_b(model, val_ad_q, candidate_embs)
  5. 평가                : evaluate_task_b_ndcg
  6. 비교                : Interest 모델과 나란히 비교

실행:
    python -X utf8 experiments/gnn_eval.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import RecoDataset
from data.graph import build_graph
from model.gnn import GNNConfig, GNNModel
from model import ModelConfig, MultiInterestModel
from model.predictor import (
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
    section("2. 그래프 구축 (transductive=True: click_val search 포함)")
    t0 = time.time()
    graph = build_graph(ds, verbose=True, transductive=True)
    print(f"\n  build_graph: {elapsed(t0)}")

    # ── 3. GNN 학습 (그래프 전파) ────────────────────────────────────────
    section("3. GNN 학습 단계 — 그래프 전파 (message passing)")

    gnn_configs = [
        GNNConfig(n_layers=2, agg_fn="mean", gamma=0.7, gamma_search=0.5),
        GNNConfig(n_layers=3, agg_fn="mean", gamma=0.7, gamma_search=0.5),
        GNNConfig(n_layers=4, agg_fn="mean", gamma=0.7, gamma_search=0.5),
        GNNConfig(n_layers=3, agg_fn="max",  gamma=0.7, gamma_search=0.5),
    ]

    gnn_results = []
    for cfg in gnn_configs:
        print(f"\n  >> {cfg}")
        model_gnn = GNNModel(cfg)
        t0 = time.time()
        model_gnn.fit(graph)
        fit_time = elapsed(t0)

        # ── 4. 추론 & 평가 ───────────────────────────────────────────────
        # Task B
        sc_b = score_task_b(model_gnn, val_ad_q, candidate_embs)
        m_b  = evaluate_task_b_ndcg(sc_b, val_ad_ans, candidate_ids)

        # Task A 방법 1: link prediction — sim(h_search, h_ad)
        sc_a_link = [
            model_gnn.score_link(ev.search_id, ad.ad_id)
            for ev, ad in val_clk_q
        ]
        best_f1_link, best_thr_link, best_m_a_link = 0.0, 0.5, None
        for thr in np.arange(0.05, 0.96, 0.05):
            m_a = evaluate_task_a(sc_a_link, val_clk_ans, float(thr))
            if m_a["f1"] > best_f1_link:
                best_f1_link, best_thr_link, best_m_a_link = m_a["f1"], float(thr), m_a

        # Task A 방법 2: user repr 기반 — (1-γ)*sim(q,a) + γ*sim(h_user,a)
        sc_a_user = [
            model_gnn.score_click(ev.user_id, ev.search_emb, ad.ad_emb)
            for ev, ad in val_clk_q
        ]
        best_f1_user, best_thr_user, best_m_a_user = 0.0, 0.5, None
        for thr in np.arange(0.05, 0.96, 0.05):
            m_a = evaluate_task_a(sc_a_user, val_clk_ans, float(thr))
            if m_a["f1"] > best_f1_user:
                best_f1_user, best_thr_user, best_m_a_user = m_a["f1"], float(thr), m_a

        gnn_results.append((cfg, m_b,
                            best_m_a_link, best_thr_link,
                            best_m_a_user, best_thr_user,
                            fit_time))
        rd = m_b["rank_dist"]
        print(f"     [L={cfg.n_layers} {cfg.agg_fn}]  "
              f"Task B NDCG@3={fmt(m_b['ndcg@3'])} (rank1={rd[1]})  "
              f"Task A(link) F1={fmt(best_f1_link)} thr={best_thr_link:.2f} AUC={fmt(best_m_a_link['auc'])}  "
              f"Task A(user) F1={fmt(best_f1_user)} thr={best_thr_user:.2f} AUC={fmt(best_m_a_user['auc'])}  "
              f"fit={fit_time}")

    # ── 6. Interest 모델 (비교 기준) ─────────────────────────────────────
    section("6. 비교: Interest 모델")
    interest_cfg = ModelConfig(
        k=5, alpha_search=0.01, alpha_click=0.5,
        gamma=0.7, gamma_search=0.5, threshold=0.5,
    )
    model_int = MultiInterestModel(interest_cfg)
    np.random.seed(SEED)
    t0 = time.time()
    train(model_int, ds.training_stream())
    train_time = elapsed(t0)

    sc_int_b = score_task_b(model_int, val_ad_q, candidate_embs)
    m_int_b  = evaluate_task_b_ndcg(sc_int_b, val_ad_ans, candidate_ids)

    sc_int_a = score_task_a(model_int, val_clk_q)
    best_f1_int, best_thr_int, best_m_int_a = 0.0, 0.5, None
    for thr in np.arange(0.05, 0.96, 0.05):
        m_a = evaluate_task_a(sc_int_a, val_clk_ans, float(thr))
        if m_a["f1"] > best_f1_int:
            best_f1_int, best_thr_int, best_m_int_a = m_a["f1"], float(thr), m_a

    rd_int = m_int_b["rank_dist"]
    print(f"\n  Interest (gamma=0.7/0.5)  "
          f"Task B NDCG@3={fmt(m_int_b['ndcg@3'])}  "
          f"Task A F1={fmt(best_f1_int)} (thr={best_thr_int:.2f})  "
          f"AUC={fmt(best_m_int_a['auc'])}  train={train_time}")

    # ── 7. 최종 비교 테이블 ──────────────────────────────────────────────
    section("7. 최종 결과 비교")

    hdr_w = 36
    print(f"\n  {'모델':<{hdr_w}}  {'TaskB':^18}  {'TaskA(link)':^24}  {'TaskA(user)':^24}")
    print(f"  {'':>{hdr_w}}  {'NDCG@3':>8} {'R1':>4} {'>R3':>4}  "
          f"{'F1':>8} {'thr':>5} {'AUC':>8}  "
          f"{'F1':>8} {'thr':>5} {'AUC':>8}")
    print("  " + "─" * 108)

    for cfg, m_b, m_a_link, thr_link, m_a_user, thr_user, fit_t in gnn_results:
        rd  = m_b["rank_dist"]
        tag = f"GNN L={cfg.n_layers} {cfg.agg_fn}"
        print(f"  {tag:<{hdr_w}}  "
              f"{fmt(m_b['ndcg@3']):>8} {rd[1]:>4} {rd['>3']:>4}  "
              f"{fmt(m_a_link['f1']):>8} {thr_link:>5.2f} {fmt(m_a_link['auc']):>8}  "
              f"{fmt(m_a_user['f1']):>8} {thr_user:>5.2f} {fmt(m_a_user['auc']):>8}")

    rd = m_int_b["rank_dist"]
    print(f"  {'Interest (γ=0.7/γ_s=0.5)':<{hdr_w}}  "
          f"{fmt(m_int_b['ndcg@3']):>8} {rd[1]:>4} {rd['>3']:>4}  "
          f"{'N/A':>8} {'':>5} {'':>8}  "
          f"{fmt(best_f1_int):>8} {best_thr_int:>5.2f} {fmt(best_m_int_a['auc']):>8}")

    print(f"\n  HistCTR baseline  Task B NDCG@3=0.0211 / Task A F1=0.0436")
    print(f"  Query-only        Task B NDCG@3=0.0989")


if __name__ == "__main__":
    main()
