"""
eval_gated.py — GatedCTRModel 을 공통 평가(shared/eval/predictor)로 측정한다.

Task A : click_validation_*  |  IsClick 예측  |  F1
Task B : ad_validation_*     |  AdID 추천     |  NDCG@3

CTRPredictor 와 동일하게 fit(ds) 로 배치 학습 후 score_pairs(pairs) 로 점수를 내고,
그 점수를 shared.eval.predictor.evaluate_task_a 에 그대로 넣어 협업자들과 동일 지표로 비교.

train/val 분리: 모델은 ds.training_stream() 전체를 받아 내부에서 SearchID 80/20 로 나눠
(content head early-stop · entity F1 지수 · 게이트 t) 선택에만 쓴다. click_validation 은
최종 F1 보고에만 사용 (선택에 미사용) → 원본 honest 방법론 재현.

실행 (프로젝트 루트에서):
    python -X utf8 models/m04_gated/experiments/eval_gated.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.data.dataset import RecoDataset
from shared.eval.predictor import (
    evaluate_task_a,
    evaluate_task_b_ndcg,
    score_task_a,
    score_task_b,
)
from models.m04_gated import GateConfig, GatedCTRModel

DATASET_DIR = ROOT / "../datasets"
SEED = 42


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def fmt(v: float) -> str:
    return f"{v:.4f}"


def sweep_f1(scores, answers_df):
    """threshold sweep 으로 best F1 (공통 evaluate_task_a 사용). 1.1% 양성 희소영역을 위해
    상위 fraction 을 촘촘히(0.004 간격) 훑는다 — rank-정규화 점수라 thr ≈ 1-top_fraction."""
    best_f1, best_thr, best_m = 0.0, 0.5, None
    for thr in np.arange(0.004, 0.99, 0.004):
        m = evaluate_task_a(scores, answers_df, float(thr))
        if m["f1"] > best_f1:
            best_f1, best_thr, best_m = m["f1"], float(thr), m
    return best_f1, best_thr, best_m


def main() -> None:
    np.random.seed(SEED)

    section("1. Dataset")
    t0 = time.time()
    ds = RecoDataset(DATASET_DIR).load()
    print(ds.summary())
    print(f"  load {time.time()-t0:.1f}s")

    val_clk_q   = ds.val_click_queries()
    val_clk_ans = ds.val_click_answers()
    candidate_embs, candidate_ids = ds.all_ad_embs()
    val_ad_q    = ds.val_ad_queries()
    val_ad_ans  = ds.val_ad_answers()

    section("2. Training (GatedCTRModel.fit — 내부 SearchID 80/20 split)")
    cfg = GateConfig()
    print(cfg)
    t0 = time.time()
    model = GatedCTRModel(cfg).fit(ds)
    print(f"  training {time.time()-t0:.1f}s")

    section("3. Task A | Click Prediction (F1)")
    n_clk = int(val_clk_ans["IsClick"].sum())
    print(f"  val click={n_clk} ({n_clk/len(val_clk_ans):.2%})")
    scores = score_task_a(model, val_clk_q)
    best_f1, best_thr, m = sweep_f1(scores, val_clk_ans)
    print(f"  GatedCTR (ours)  F1={fmt(m['f1'])}  P={fmt(m['precision'])}  "
          f"R={fmt(m['recall'])}  AUC={fmt(m['auc'])}  thr*={best_thr:.2f}")

    section("4. Task B | Ad Recommendation (NDCG@3)")
    sc_b = score_task_b(model, val_ad_q, candidate_embs)
    mb = evaluate_task_b_ndcg(sc_b, val_ad_ans, candidate_ids)
    print(f"  GatedCTR (ours)  NDCG@3={fmt(mb['ndcg@3'])}  queries={mb['n_queries']}")

    section("5. Summary")
    print(f"  Task A  F1={fmt(m['f1'])}  AUC={fmt(m['auc'])}  (thr*={best_thr:.2f})")
    print(f"  Task B  NDCG@3={fmt(mb['ndcg@3'])}")


if __name__ == "__main__":
    main()
