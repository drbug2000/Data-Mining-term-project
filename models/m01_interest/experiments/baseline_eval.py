"""
baseline_eval.py — 기본 성능 테스트.

Task A : click_validation_*  |  IsClick 예측  |  F1
Task B : ad_validation_*     |  AdID 추천     |  NDCG@3

실행 (프로젝트 루트에서):
    python experiments/baseline_eval.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.data.dataset import RecoDataset
from models.m01_interest import ModelConfig, MultiInterestModel
from shared.eval.predictor import (
    evaluate_task_a,
    evaluate_task_b_ndcg,
    score_task_a,
    score_task_b,
    train,
)

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATASET_DIR = ROOT / "../datasets"
SEED        = 42

CONFIG = ModelConfig(
    k             = 5,
    alpha_search  = 0.01,
    alpha_click   = 0.5,
    alpha_neg     = 0.0,
    temperature   = 1.0,
    gamma         = 0.7,   # click 이력 유저의 interest 가중치
    gamma_search  = 0.5,   # search 이력만 있는 유저의 interest 가중치 (tiered adaptive gamma)
    threshold     = 0.5,
)


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────
def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.2f}s"


def fmt(v: float) -> str:
    return f"{v:.4f}"


def print_confusion_a(per_class: dict) -> None:
    """Task A 2x2 confusion matrix."""
    c0 = per_class.get(0, {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    c1 = per_class.get(1, {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    TP = c1["tp"]; FP = c1["fp"]; FN = c1["fn"]; TN = c0["tp"]
    print(f"\n  Confusion Matrix (Task A)  [[TN, FP], [FN, TP]]")
    print(f"                  Pred 0      Pred 1")
    print(f"    True 0  :  {TN:>8}   {FP:>8}    (support={c0['support']})")
    print(f"    True 1  :  {FN:>8}   {TP:>8}    (support={c1['support']})")


def print_per_class(per_class: dict, label: str = "class") -> None:
    """Per-class TP / FP / FN / Precision / Recall / F1 / Support 테이블."""
    print(f"\n  Per-class metrics")
    print(f"  {'':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'─'*10}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*6}")
    for c, m in sorted(per_class.items()):
        name = str(c) if label == "class" else f"{label} {c}"
        print(f"  {name:>10} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} "
              f"{fmt(m['precision']):>10} {fmt(m['recall']):>8} "
              f"{fmt(m['f1']):>8} {m['support']:>8}")


# ──────────────────────────────────────────────
# 베이스라인 스코어 생성
# ──────────────────────────────────────────────
def random_scores_b(queries, n_candidates, rng) -> dict[int, np.ndarray]:
    return {ev.search_id: rng.random(n_candidates).astype(np.float32) for ev in queries}


def constant_scores_a(pairs, value: float) -> list[float]:
    return [value] * len(pairs)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main() -> None:
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # ── 1. 데이터 로드 ────────────────────────
    section("1. Dataset")
    t0 = time.time()
    ds = RecoDataset(DATASET_DIR).load()
    print(ds.summary())
    print(f"\n  load: {elapsed(t0)}")

    candidate_embs, candidate_ids = ds.all_ad_embs()

    # Task A : click_validation (IsClick 예측, F1)
    val_clk_q   = ds.val_click_queries()
    val_clk_ans = ds.val_click_answers()

    # Task B : ad_validation (AdID 추천, NDCG@3)
    val_ad_q    = ds.val_ad_queries()
    val_ad_ans  = ds.val_ad_answers()

    # ── 2. 훈련 ──────────────────────────────
    section("2. Training")
    print(CONFIG)
    model = MultiInterestModel(CONFIG)
    t0 = time.time()
    train(model, ds.training_stream())
    print(f"\n  training: {elapsed(t0)}")

    # ── 3. Task A  |  Click Prediction (F1) ──
    section("3. Task A  |  Click Prediction  (20,000 pairs, F1)")
    n_clicks = val_clk_ans["IsClick"].sum()
    print(f"  정답 분포: click={n_clicks} ({n_clicks/len(val_clk_ans):.2%})  "
          f"no-click={len(val_clk_ans)-n_clicks}\n")

    rows_a: list[tuple[str, dict, str]] = []
    thr = CONFIG.threshold

    t0 = time.time()
    sc = score_task_a(model, val_clk_q)
    m_ours_a = evaluate_task_a(sc, val_clk_ans, thr)
    rows_a.append(("MultiInterest (ours)", m_ours_a, elapsed(t0)))

    sc = constant_scores_a(val_clk_q, 0.0)
    rows_a.append(("Always-0 (no click)", evaluate_task_a(sc, val_clk_ans, thr), " -"))

    sc = constant_scores_a(val_clk_q, 1.0)
    rows_a.append(("Always-1 (all click)", evaluate_task_a(sc, val_clk_ans, thr), " -"))

    header = (f"  {'모델':<24} {'Accuracy':>10} {'Precision':>10} "
              f"{'Recall':>8} {'F1':>8} {'AUC':>8}  {'시간':>6}")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for name, m, t in rows_a:
        print(
            f"  {name:<24} {fmt(m['accuracy']):>10} {fmt(m['precision']):>10} "
            f"{fmt(m['recall']):>8} {fmt(m['f1']):>8} {fmt(m['auc']):>8}  {t:>6}"
        )
    print("  * Precision / Recall / F1 : IsClick=1 (클릭) 클래스 기준")

    print_per_class(m_ours_a["per_class"], label="IsClick")
    print_confusion_a(m_ours_a["per_class"])

    # ── 4. Task B  |  Ad Recommendation (NDCG@3) ──
    section("4. Task B  |  Ad Recommendation  (214 queries, 17,518 candidates, NDCG@3)")

    rows_b: list[tuple[str, dict, str]] = []

    t0 = time.time()
    sc = score_task_b(model, val_ad_q, candidate_embs)
    m_ours_b = evaluate_task_b_ndcg(sc, val_ad_ans, candidate_ids)
    rows_b.append(("MultiInterest (ours)", m_ours_b, elapsed(t0)))

    # Query-only 베이스라인
    sc_qonly = score_task_b(MultiInterestModel(ModelConfig(gamma=0.0)), val_ad_q, candidate_embs)
    rows_b.append(("Query-only (gamma=0)", evaluate_task_b_ndcg(sc_qonly, val_ad_ans, candidate_ids), " -"))

    # Random 베이스라인
    sc_rand = random_scores_b(val_ad_q, len(candidate_ids), rng)
    rows_b.append(("Random", evaluate_task_b_ndcg(sc_rand, val_ad_ans, candidate_ids), " -"))

    header = f"  {'모델':<24} {'NDCG@3':>8} {'Queries':>8}  {'시간':>6}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for name, m, t in rows_b:
        print(f"  {name:<24} {fmt(m['ndcg@3']):>8} {m['n_queries']:>8}  {t:>6}")

    rd = m_ours_b["rank_dist"]
    print(f"\n  Rank distribution (MultiInterest, n={m_ours_b['n_queries']})")
    print(f"    Rank 1 : {rd[1]:>4}  (NDCG = 1.000)")
    print(f"    Rank 2 : {rd[2]:>4}  (NDCG = 0.631)")
    print(f"    Rank 3 : {rd[3]:>4}  (NDCG = 0.500)")
    print(f"    Rank >3: {rd['>3']:>4}  (NDCG = 0.000)")

    # ── 5. 요약 ───────────────────────────────
    section("5. Summary")
    ma = m_ours_a
    mb = m_ours_b
    print(f"  Task A  F1={fmt(ma['f1'])}  Precision={fmt(ma['precision'])}  "
          f"Recall={fmt(ma['recall'])}  AUC={fmt(ma['auc'])}")
    print(f"          (IsClick=1 기준, threshold={thr})")
    print(f"  Task B  NDCG@3={fmt(mb['ndcg@3'])}  "
          f"(rank1={rd[1]}, rank2={rd[2]}, rank3={rd[3]}, >3={rd['>3']})")
    print(f"\n  Config : {CONFIG.to_dict()}")


if __name__ == "__main__":
    main()
