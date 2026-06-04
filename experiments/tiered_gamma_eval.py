"""
tiered_gamma_eval.py — 3단계 adaptive gamma 실험.

  cold        → gamma_eff = 0
  search-only → gamma_eff = gamma_search  (낮은 가중치)
  click       → gamma_eff = gamma         (높은 가중치)

gamma × gamma_search 그리드 sweep → Task B NDCG@3
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import RecoDataset
from model import ModelConfig, MultiInterestModel
from model.predictor import evaluate_task_b_ndcg, score_task_b, train

DATASET_DIR = ROOT / "../datasets"
SEED = 42


def main():
    np.random.seed(SEED)

    ds = RecoDataset(DATASET_DIR).load()
    candidate_embs, candidate_ids = ds.all_ad_embs()
    val_ad_q   = ds.val_ad_queries()
    val_ad_ans = ds.val_ad_answers()

    # gamma (click용) × gamma_search (search-only용) sweep
    gammas        = [0.3, 0.5, 0.7]
    gamma_searches = [0.1, 0.2, 0.3, 0.5]

    print(f"\n  gamma_search\\gamma  ", end="")
    for g in gammas:
        print(f"  γ={g:.1f}", end="")
    print()
    print("  " + "─" * (20 + 8 * len(gammas)))

    best_ndcg = 0.0
    best_cfg  = None

    def run(g, gs):
        np.random.seed(SEED)               # ← 매 실험 동일 seed 보장
        cfg = ModelConfig(
            k=5, alpha_search=0.01, alpha_click=0.5,
            alpha_neg=0.0, temperature=1.0,
            gamma=g, gamma_search=gs, threshold=0.5,
        )
        model = MultiInterestModel(cfg)
        train(model, ds.training_stream())
        sc = score_task_b(model, val_ad_q, candidate_embs)
        return evaluate_task_b_ndcg(sc, val_ad_ans, candidate_ids)

    for gs in gamma_searches:
        print(f"  γ_search={gs:.2f}        ", end="")
        for g in gammas:
            if gs >= g:
                print(f"  {'(skip)':>6}", end="")
                continue
            m  = run(g, gs)
            nd = m["ndcg@3"]
            print(f"  {nd:.4f}", end="", flush=True)
            if nd > best_ndcg:
                best_ndcg = nd
                best_cfg  = ModelConfig(k=5, alpha_search=0.01, alpha_click=0.5,
                                        gamma=g, gamma_search=gs)
        print()

    # 비교용 베이스라인 (동일 seed)
    print("\n  --- 비교 베이스라인 ---")
    baselines = [
        ("이전 최고 (click만 활성, γ=0.5, γ_s=0.0)", 0.5, 0.0),
        ("query-only (γ=0, γ_s=0)",                  0.0, 0.0),
    ]
    for label, g, gs in baselines:
        m = run(g, gs)
        print(f"  {label:<48}  NDCG@3={m['ndcg@3']:.4f}")

    print(f"\n  ★ 최적: gamma={best_cfg.gamma}, gamma_search={best_cfg.gamma_search}"
          f"  →  NDCG@3={best_ndcg:.4f}")
    print(f"  HistCTR baseline NDCG@3 = 0.0211")


if __name__ == "__main__":
    main()
