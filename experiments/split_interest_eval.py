"""
split_interest_eval.py — Click-only interest 분리 실험.

변경 사항:
  - _interests       : 검색+클릭 → Task B (ad recommendation)
  - _click_interests : 클릭/비클릭 전용 → Task A (click prediction)
  - score_click()이 _click_interests를 사용함으로써 검색 오염을 제거

실행:
    python experiments/split_interest_eval.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import RecoDataset
from model import ModelConfig, MultiInterestModel
from model.predictor import (
    evaluate_task_a,
    evaluate_task_b_ndcg,
    score_task_b,
    score_task_a,
    train,
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


def sweep_threshold(scores, answers_df):
    """F1이 최대가 되는 threshold와 최대 F1을 반환한다."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.96, 0.05):
        m = evaluate_task_a(scores, answers_df, float(thr))
        if m["f1"] > best_f1:
            best_f1, best_thr = m["f1"], float(thr)
    return best_f1, best_thr


def main() -> None:
    np.random.seed(SEED)

    section("데이터 로드")
    ds = RecoDataset(DATASET_DIR).load()
    candidate_embs, candidate_ids = ds.all_ad_embs()
    val_clk_q   = ds.val_click_queries()
    val_clk_ans = ds.val_click_answers()
    val_ad_q    = ds.val_ad_queries()
    val_ad_ans  = ds.val_ad_answers()
    print(ds.summary())

    # ── 실험 그리드 ──
    # alpha_neg : click_interests에 negative feedback 강도
    # gamma     : Task A에서 click interest 혼합 비율
    alpha_negs = [0.0, 0.05, 0.1, 0.3]
    gammas     = [0.3, 0.5, 0.7, 1.0]

    section("Sweep: alpha_neg × gamma  (Task A F1 / Task B NDCG@3)")
    header = (f"  {'alpha_neg':>10} {'gamma':>6} {'F1(best)':>10} "
              f"{'thr':>6} {'AUC':>8} {'NDCG@3':>8}  {'시간':>6}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    best_combo = None
    best_f1_global = 0.0

    for alpha_neg in alpha_negs:
        for gamma in gammas:
            cfg = ModelConfig(
                k=5,
                alpha_search=0.01,
                alpha_click=0.5,
                alpha_neg=alpha_neg,
                temperature=1.0,
                gamma=gamma,
                threshold=0.5,
            )
            model = MultiInterestModel(cfg)
            t0 = time.time()
            train(model, ds.training_stream())

            # Task A: click_validation, score_task_a → F1
            sc_a = score_task_a(model, val_clk_q)
            best_f1, best_thr = sweep_threshold(sc_a, val_clk_ans)
            m_a = evaluate_task_a(sc_a, val_clk_ans, best_thr)

            # Task B: ad_validation, score_task_b → NDCG@3
            sc_b = score_task_b(model, val_ad_q, candidate_embs)
            m_b = evaluate_task_b_ndcg(sc_b, val_ad_ans, candidate_ids)

            t = elapsed(t0)
            print(
                f"  {alpha_neg:>10.3f} {gamma:>6.2f} {fmt(best_f1):>10} "
                f"{best_thr:>6.2f} {fmt(m_a['auc']):>8} {fmt(m_b['ndcg@3']):>8}  {t:>6}"
            )

            if best_f1 > best_f1_global:
                best_f1_global = best_f1
                best_combo = (alpha_neg, gamma, best_thr, m_a, m_b)

    # ── 최적 설정 상세 출력 ──
    if best_combo:
        an, gm, thr, ma, mb = best_combo
        section(f"최적 설정 : alpha_neg={an}, gamma={gm}, threshold={thr:.2f}")
        print(f"  Task A  F1={fmt(ma['f1'])}  Precision={fmt(ma['precision'])}  "
              f"Recall={fmt(ma['recall'])}  AUC={fmt(ma['auc'])}")
        rd = mb["rank_dist"]
        print(f"  Task B  NDCG@3={fmt(mb['ndcg@3'])}  "
              f"(rank1={rd[1]}, rank2={rd[2]}, rank3={rd[3]}, >3={rd['>3']})")
        print(f"\n  HistCTR baseline : Task A F1=0.0436, Task B NDCG@3=0.0211")


if __name__ == "__main__":
    main()
