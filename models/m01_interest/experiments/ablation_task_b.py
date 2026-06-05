"""
ablation_task_b.py — Task B NDCG@3 Ablation Study

각 모듈을 하나씩 제거/비활성화했을 때 Task B NDCG@3 변화를 측정한다.

Ablation 목록:
  FULL            : SOTA  (γ=0.7, γ_s=0.5, k=5, α_s=0.01, α_c=0.5, τ=0.1)
  ── interest 전체 제거 ──
  w/o interest    : γ=0, γ_s=0                         (query-only baseline)
  ── tiered gamma ──
  w/o tiered γ    : γ_s=0                               (search-only → cold-start)
  ── interest 구성 신호 ──
  w/o search sig  : α_s=0                               (클릭 신호만으로 interest 구축)
  w/o click sig   : α_c=0                               (검색 신호만으로 interest 구축)
  ── interest vector 수 ──
  k=1             : k=1                                 (single interest vector)
  k=3             : k=3
  k=10            : k=10
  ── soft assignment ──
  hard (τ=0.01)   : temperature=0.01                    (nearest 1개에 집중)
  uniform (τ=100) : temperature=100.0                   (모든 vector 균등 업데이트)

Run from project root:
    python models/m01_interest/experiments/ablation_task_b.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.data.dataset import RecoDataset
from models.m01_interest import ModelConfig, MultiInterestModel
from shared.eval.predictor import evaluate_task_b_ndcg, score_task_b, train

DATASET_DIR = ROOT / "../datasets"
SEED = 42
HISTCTR_NDCG = 0.0211

SOTA = dict(k=5, alpha_search=0.01, alpha_click=0.5,
            alpha_neg=0.0, temperature=0.1,
            gamma=0.7, gamma_search=0.5, threshold=0.5)


def run_once(ds, val_ad_q, val_ad_ans, cand_embs, cand_ids, **overrides):
    np.random.seed(SEED)
    cfg = ModelConfig(**{**SOTA, **overrides})
    model = MultiInterestModel(cfg)
    train(model, ds.training_stream())
    sc = score_task_b(model, val_ad_q, cand_embs)
    return evaluate_task_b_ndcg(sc, val_ad_ans, cand_ids)


def main():
    np.random.seed(SEED)
    ds = RecoDataset(DATASET_DIR).load()
    cand_embs, cand_ids = ds.all_ad_embs()
    val_ad_q   = ds.val_ad_queries()
    val_ad_ans = ds.val_ad_answers()

    ablations = [
        # (label, overrides, description)
        ("FULL (SOTA)",           {},                                        "기준선"),
        ("─── interest 전체 ───", None, ""),
        ("w/o interest",          dict(gamma=0.0, gamma_search=0.0),        "query-only (γ=0)"),
        ("─── tiered gamma ────", None, ""),
        ("w/o tiered γ",          dict(gamma_search=0.0),                   "search-only → cold-start"),
        ("─── interest 신호 ────", None, ""),
        ("w/o search signal",     dict(alpha_search=0.0),                   "클릭 신호만으로 interest 구축"),
        ("w/o click signal",      dict(alpha_click=0.0),                    "검색 신호만으로 interest 구축"),
        ("─── k (벡터 수) ──────", None, ""),
        ("k=1",                   dict(k=1),                                "single interest vector"),
        ("k=3",                   dict(k=3),                                ""),
        ("k=5  (SOTA)",           {},                                        ""),
        ("k=10",                  dict(k=10),                               ""),
        ("─── soft assignment ──", None, ""),
        ("SOTA   (τ=0.1)",        {},                                        "SOTA"),
        ("sharp  (τ=0.01)",       dict(temperature=0.01),                   "최고점 (과적합 위험)"),
        ("warm   (τ=1.0)",        dict(temperature=1.0),                    "이전 기본값"),
        ("uniform (τ=100)",       dict(temperature=100.0),                  "모든 vector 균등 업데이트"),
    ]

    sota_ndcg = None

    print(f"\n  {'Ablation':<24}  {'NDCG@3':>8}  {'vs SOTA':>8}  {'r1':>4} {'r2':>4} {'r3':>4}  Description")
    print(f"  {'─'*24}  {'─'*8}  {'─'*8}  {'─'*4} {'─'*4} {'─'*4}  {'─'*30}")

    for label, overrides, desc in ablations:
        if overrides is None:          # 구분선
            print(f"  {label}")
            continue

        t0 = time.time()
        m  = run_once(ds, val_ad_q, val_ad_ans, cand_embs, cand_ids, **overrides)
        nd = m["ndcg@3"]
        rd = m["rank_dist"]

        if label.startswith("FULL"):
            sota_ndcg = nd
            delta_str = "  (base)"
        else:
            delta = nd - sota_ndcg
            delta_str = f"{delta:+.4f}"

        print(f"  {label:<24}  {nd:.4f}    {delta_str:>8}  "
              f"{rd[1]:>4} {rd[2]:>4} {rd[3]:>4}  {desc}")

    print(f"\n  HistCTR baseline NDCG@3 = {HISTCTR_NDCG}")
    print(f"  SOTA vs HistCTR = x{sota_ndcg/HISTCTR_NDCG:.1f}")


if __name__ == "__main__":
    main()
