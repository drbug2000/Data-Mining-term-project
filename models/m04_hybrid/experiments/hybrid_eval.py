"""
hybrid_eval.py — HybridModel v2 평가 (Task별 최강 모델 결합).

  Task B: m01 다중 interest (GNN = Option A 입력 보강만)
  Task A: (1-α)·m01_click_score + α·gnn.score_link(h_search, h_ad)

실행:
    python -X utf8 models/m04_hybrid/experiments/hybrid_eval.py
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
from shared.eval.predictor import (
    evaluate_task_a, evaluate_task_b_ndcg,
    score_task_a, score_task_b, train,
)
from models.m01_interest import ModelConfig, MultiInterestModel
from models.m02_gnn import GNNConfig, GNNModel
from models.m04_hybrid import HybridConfig, HybridModel

DATASET_DIR = ROOT / "../datasets"
SEED = 42


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


def fmt(v: float) -> str:
    return f"{v:.4f}"


def best_threshold_sweep(scores: list[float], answers_df) -> tuple[float, float, dict]:
    best_f1, best_thr, best_m = 0.0, 0.5, None
    for thr in np.arange(0.05, 0.96, 0.05):
        m = evaluate_task_a(scores, answers_df, float(thr))
        if m["f1"] > best_f1:
            best_f1, best_thr, best_m = m["f1"], float(thr), m
    return best_f1, best_thr, best_m


# ── Hybrid 전용 ID-aware 함수 ─────────────────────────────────────────────────

def train_hybrid(model: HybridModel, stream) -> None:
    """search_id / ad_id 를 함께 전달 (Option A 활성화)."""
    for event in stream:
        model.update_search(event.user_id, event.search_emb,
                            search_id=event.search_id)
        for ad in event.ads:
            model.update_click(event.user_id, ad.ad_emb,
                               clicked=bool(ad.is_click),
                               ad_id=ad.ad_id)


def score_task_b_hybrid(model, queries, candidate_embs, candidate_ids):
    """candidate_ids 전달 → GNN-enriched 후보 repr 활성화."""
    return {
        ev.search_id: model.score_ad_candidates(
            ev.user_id, ev.search_emb, candidate_embs, candidate_ids
        )
        for ev in queries
    }


def score_task_a_hybrid(model, pairs):
    """search_id + ad_id 전달 → GNN link prediction 활성화."""
    return [
        model.score_click(
            ev.user_id, ev.search_emb, ad.ad_emb,
            search_id=ev.search_id,
            ad_id=ad.ad_id,
        )
        for ev, ad in pairs
    ]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(SEED)

    # ── 1. 데이터 ─────────────────────────────────────────────────────────────
    section("1. 데이터 로드")
    ds = RecoDataset(DATASET_DIR).load()
    print(ds.summary())
    candidate_embs, candidate_ids = ds.all_ad_embs()
    val_ad_q    = ds.val_ad_queries()
    val_ad_ans  = ds.val_ad_answers()
    val_clk_q   = ds.val_click_queries()
    val_clk_ans = ds.val_click_answers()

    # ── 2. 그래프 ─────────────────────────────────────────────────────────────
    # base: Option A (Task B interest 업데이트) — sim 엣지 없이 embedding 품질 유지
    # sim : score_link (Task A GNN 신호) — sim 엣지로 h_search 전파를 풍부하게
    section("2. 그래프 구축")
    t0 = time.time()
    graph_base = build_graph(ds, verbose=False, transductive=True, top_k_sim=0)
    print(f"  base graph: {elapsed(t0)}")
    t0 = time.time()
    graph_sim  = build_graph(ds, verbose=False, transductive=True, top_k_sim=10)
    print(f"  sim  graph: {elapsed(t0)}")

    # ── 3. Baseline: m01 ──────────────────────────────────────────────────────
    section("3. Baseline — MultiInterest (m01)")
    m01 = MultiInterestModel(ModelConfig(
        k=5, alpha_search=0.01, alpha_click=0.5,
        gamma=0.7, gamma_search=0.5, threshold=0.5,
    ))
    np.random.seed(SEED)
    t0 = time.time()
    train(m01, ds.training_stream())
    m01_train_t = elapsed(t0)

    sc_b_m01  = score_task_b(m01, val_ad_q, candidate_embs)
    met_b_m01 = evaluate_task_b_ndcg(sc_b_m01, val_ad_ans, candidate_ids)
    f1_m01, thr_m01, met_a_m01 = best_threshold_sweep(
        score_task_a(m01, val_clk_q), val_clk_ans
    )
    rd = met_b_m01["rank_dist"]
    print(f"  TaskB NDCG@3={fmt(met_b_m01['ndcg@3'])}  R1={rd[1]} R2={rd[2]} R3={rd[3]}")
    print(f"  TaskA F1={fmt(f1_m01)} thr={thr_m01:.2f}  "
          f"P={fmt(met_a_m01['precision'])} R={fmt(met_a_m01['recall'])}  train={m01_train_t}")

    # ── 4. Baseline: m02 GNN — score_link (Task A 전용) ───────────────────────
    section("4. Baseline — GNN (m02, L=2)")
    m02 = GNNModel(GNNConfig(n_layers=2, agg_fn="mean", gamma=0.7, gamma_search=0.5))
    t0 = time.time()
    m02.fit(graph_sim)   # +sim 그래프: score_link 품질 최대화
    m02_fit_t = elapsed(t0)

    sc_b_m02  = score_task_b(m02, val_ad_q, candidate_embs)
    met_b_m02 = evaluate_task_b_ndcg(sc_b_m02, val_ad_ans, candidate_ids)

    # GNN Task A: score_link (h_search vs h_ad) — 구조 기반 link prediction
    sc_a_link = [
        m02.score_link(ev.search_id, ad.ad_id)
        for ev, ad in val_clk_q
    ]
    f1_link, thr_link, met_a_link = best_threshold_sweep(sc_a_link, val_clk_ans)

    # GNN Task A: score_click (h_user 기반) — 비교용
    sc_a_user = [
        m02.score_click(ev.user_id, ev.search_emb, ad.ad_emb)
        for ev, ad in val_clk_q
    ]
    f1_user, thr_user, met_a_user = best_threshold_sweep(sc_a_user, val_clk_ans)

    rd = met_b_m02["rank_dist"]
    print(f"  TaskB NDCG@3={fmt(met_b_m02['ndcg@3'])}  R1={rd[1]} R2={rd[2]} R3={rd[3]}")
    print(f"  TaskA(score_link)  F1={fmt(f1_link)} thr={thr_link:.2f}  "
          f"P={fmt(met_a_link['precision'])} R={fmt(met_a_link['recall'])}")
    print(f"  TaskA(score_click) F1={fmt(f1_user)} thr={thr_user:.2f}  "
          f"P={fmt(met_a_user['precision'])} R={fmt(met_a_user['recall'])}")
    print(f"  fit={m02_fit_t}")

    # ── 5. HybridModel — link_alpha sweep ────────────────────────────────────
    section("5. HybridModel v2 — link_alpha sweep (Task A)")
    print("  Task B: 항상 m01 다중 interest  (link_alpha 무관)")
    print("  Task A: (1-α)·m01_click + α·gnn.score_link\n")

    link_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hybrid_results = []

    for α in link_alphas:
        cfg = HybridConfig(
            k=5, alpha_search=0.01, alpha_click=0.5,
            gamma=0.7, gamma_search=0.5, threshold=0.5,
            n_layers=2, agg_fn="mean",
            link_alpha=α,
        )
        model = HybridModel(cfg)
        model.fit(graph_base)           # Task B: base graph (embedding 품질 유지)
        model.fit_task_a(graph_sim)     # Task A: +sim graph (score_link 품질 최대화)
        np.random.seed(SEED)
        train_hybrid(model, ds.training_stream())

        sc_b  = score_task_b_hybrid(model, val_ad_q, candidate_embs, candidate_ids)
        met_b = evaluate_task_b_ndcg(sc_b, val_ad_ans, candidate_ids)

        sc_a  = score_task_a_hybrid(model, val_clk_q)
        f1, thr, met_a = best_threshold_sweep(sc_a, val_clk_ans)

        hybrid_results.append((α, met_b, f1, thr, met_a))

        rd = met_b["rank_dist"]
        print(f"  alpha={α:.1f}  TaskB={fmt(met_b['ndcg@3'])} (R1={rd[1]})  "
              f"TaskA F1={fmt(f1)} thr={thr:.2f}  "
              f"P={fmt(met_a['precision'])} R={fmt(met_a['recall'])}")

    # ── 6. 최종 비교 테이블 ──────────────────────────────────────────────────
    section("6. 최종 결과 비교")

    # best link_alpha by F1
    best_α, best_b, best_f1, best_thr, best_a = max(hybrid_results, key=lambda x: x[2])

    W = 38
    print(f"\n  {'모델':<{W}}  {'TaskB NDCG@3':>12}  {'TaskA F1':>9}  {'P':>7}  {'R':>7}")
    print("  " + "-" * 78)
    print(f"  {'MultiInterest m01':<{W}}  "
          f"{fmt(met_b_m01['ndcg@3']):>12}  {fmt(f1_m01):>9}  "
          f"{fmt(met_a_m01['precision']):>7}  {fmt(met_a_m01['recall']):>7}")
    print(f"  {'GNN m02 +sim (TaskA=score_link)':<{W}}  "
          f"{fmt(met_b_m02['ndcg@3']):>12}  {fmt(f1_link):>9}  "
          f"{fmt(met_a_link['precision']):>7}  {fmt(met_a_link['recall']):>7}")
    print(f"  {'GNN m02 +sim (TaskA=score_click)':<{W}}  "
          f"{fmt(met_b_m02['ndcg@3']):>12}  {fmt(f1_user):>9}  "
          f"{fmt(met_a_user['precision']):>7}  {fmt(met_a_user['recall']):>7}")
    print("  " + "·" * 78)
    for α, met_b, f1, thr, met_a in hybrid_results:
        tag = f"Hybrid v2 link_alpha={α:.1f}"
        marker = " <-- best F1" if α == best_α else ""
        print(f"  {tag:<{W}}  "
              f"{fmt(met_b['ndcg@3']):>12}  {fmt(f1):>9}  "
              f"{fmt(met_a['precision']):>7}  {fmt(met_a['recall']):>7}{marker}")

    print(f"\n  참고:  HistCTR TaskB=0.0211  QueryOnly TaskB=0.0989  HistCTR TaskA F1=0.0436")


if __name__ == "__main__":
    main()
