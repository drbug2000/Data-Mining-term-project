"""
temperature_sweep.py — soft assignment temperature τ sweep (Task B NDCG@3)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.data.dataset import RecoDataset
from models.m01_interest import ModelConfig, MultiInterestModel
from shared.eval.predictor import evaluate_task_b_ndcg, score_task_b, train

DATASET_DIR = ROOT / "../datasets"
SEED = 42

SOTA = dict(k=5, alpha_search=0.01, alpha_click=0.5,
            alpha_neg=0.0, gamma=0.7, gamma_search=0.5, threshold=0.5)

TEMPS = [0.001, 0.005, 0.01, 0.02, 0.05,
         0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

def main():
    np.random.seed(SEED)
    ds = RecoDataset(DATASET_DIR).load()
    cand_embs, cand_ids = ds.all_ad_embs()
    val_ad_q   = ds.val_ad_queries()
    val_ad_ans = ds.val_ad_answers()

    print(f"\n  {'τ':>8}  {'NDCG@3':>8}  {'vs τ=1.0':>9}  "
          f"{'r1':>4} {'r2':>4} {'r3':>4} {'>3':>4}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*4} {'─'*4} {'─'*4} {'─'*4}")

    results = []
    sota_ndcg = None

    for tau in TEMPS:
        np.random.seed(SEED)
        cfg   = ModelConfig(**SOTA, temperature=tau)
        model = MultiInterestModel(cfg)
        train(model, ds.training_stream())
        sc = score_task_b(model, val_ad_q, cand_embs)
        m  = evaluate_task_b_ndcg(sc, val_ad_ans, cand_ids)
        nd = m["ndcg@3"]
        rd = m["rank_dist"]
        results.append((tau, nd, rd))

        if tau == 1.0:
            sota_ndcg = nd

    # print (sota_ndcg 확정 후)
    best_tau, best_nd = max(results, key=lambda x: x[1])[:2]
    for tau, nd, rd in results:
        delta = f"{nd - sota_ndcg:+.4f}" if sota_ndcg else "  -"
        marker = " ★" if tau == best_tau else ""
        print(f"  {tau:>8.3f}  {nd:.4f}    {delta:>9}  "
              f"{rd[1]:>4} {rd[2]:>4} {rd[3]:>4} {rd['>3']:>4}{marker}")

    print(f"\n  Best: τ={best_tau}  NDCG@3={best_nd:.4f}")


if __name__ == "__main__":
    main()
