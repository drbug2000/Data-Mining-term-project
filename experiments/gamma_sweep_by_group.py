"""
gamma_sweep_by_group.py
─────────────────────────────────────────────────────────────────────
γ 값을 바꾸면서 Task B NDCG@3을 유저 그룹별로 측정.

유저 분류
  cold        : 훈련 데이터에 전혀 등장하지 않음
  search-only : 검색 기록 있음, 클릭 없음
  click       : 클릭 기록 1건 이상

γ 적용 방식
  - "uniform γ"  : 그룹 구분 없이 동일한 γ를 모든 유저에게 적용
                   (cold 포함 전원에게 interest 반영)
  - "adaptive γ" : cold=0, search-only=γ_s, click=γ 로 차등 적용

실행:
    python -X utf8 experiments/gamma_sweep_by_group.py
"""

from __future__ import annotations

import math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import RecoDataset
from model import ModelConfig, MultiInterestModel
from model.predictor import train

DATASET_DIR = ROOT / "../datasets"
SEED = 42

GAMMAS = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0]
GROUPS = ["cold", "search-only", "click", "ALL"]


# ── 유틸 ──────────────────────────────────────────────────────────────────
def ndcg3(rank: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank <= 3 else 0.0


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def score_all(model, val_ad_q, candidate_embs):
    """score_task_b와 동일하되, uniform γ를 강제하기 위해 내부 구현."""
    return {
        ev.search_id: model.score_ad_candidates(ev.user_id, ev.search_emb, candidate_embs)
        for ev in val_ad_q
    }


def ndcg_by_group(scores_dict, val_ad_q, val_ad_ans, candidate_ids, classify_fn):
    """그룹별 NDCG@3 리스트를 반환."""
    id_to_idx = {aid: i for i, aid in enumerate(candidate_ids)}
    group_vals: dict[str, list[float]] = defaultdict(list)

    for ev in val_ad_q:
        sid = ev.search_id
        correct_aid = val_ad_ans.get(sid)
        if correct_aid is None or correct_aid not in id_to_idx:
            continue
        if sid not in scores_dict:
            continue

        sc   = scores_dict[sid]
        rank = int((sc > sc[id_to_idx[correct_aid]]).sum()) + 1
        nd   = ndcg3(rank)
        grp  = classify_fn(ev.user_id)

        group_vals[grp].append(nd)
        group_vals["ALL"].append(nd)

    return {g: (np.mean(v) if v else 0.0, len(v))
            for g, v in group_vals.items()}


# ── 모델 생성 헬퍼 ─────────────────────────────────────────────────────────
def make_model(gamma: float, gamma_search: float = 0.0) -> MultiInterestModel:
    np.random.seed(SEED)
    cfg = ModelConfig(
        k=5, alpha_search=0.01, alpha_click=0.5,
        alpha_neg=0.0, temperature=1.0,
        gamma=gamma, gamma_search=gamma_search, threshold=0.5,
    )
    return MultiInterestModel(cfg)


def fmt(v: float) -> str:
    return f"{v:.4f}"


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    ds = RecoDataset(DATASET_DIR).load()
    candidate_embs, candidate_ids = ds.all_ad_embs()
    val_ad_q   = ds.val_ad_queries()
    val_ad_ans = ds.val_ad_answers()

    # 유저 분류 사전 구축
    user_search_cnt: dict[int, int] = defaultdict(int)
    user_click_cnt:  dict[int, int] = defaultdict(int)
    for ev in ds.training_stream():
        user_search_cnt[ev.user_id] += 1
        for ad in ev.ads:
            if ad.is_click:
                user_click_cnt[ev.user_id] += 1

    def classify(uid: int) -> str:
        if user_click_cnt[uid] > 0:        return "click"
        if user_search_cnt[uid] > 0:       return "search-only"
        return "cold"

    # 유저 분포 출력
    dist = defaultdict(int)
    for ev in val_ad_q:
        dist[classify(ev.user_id)] += 1
    print(f"\n  val_ad_queries 유저 분포  "
          f"cold={dist['cold']}  search-only={dist['search-only']}  click={dist['click']}")

    # ── 실험 A: uniform γ (모든 유저에게 동일 γ 적용) ─────────────────────
    print(f"\n{'─'*72}")
    print(f"  실험 A: uniform γ  (cold/search/click 모두 같은 γ)")
    print(f"{'─'*72}")

    # 헤더
    header = f"  {'γ':>5}  "
    for g in GROUPS:
        header += f"  {g:>12}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    # γ=0일 때 모델 한 번만 훈련 (query-only는 interest 무관)
    base_model = make_model(0.0)
    train(base_model, ds.training_stream())

    results_uniform: dict[float, dict] = {}

    for gamma in GAMMAS:
        if gamma == 0.0:
            model = base_model
        else:
            model = make_model(gamma, gamma_search=gamma)  # search도 동일 γ 적용
            train(model, ds.training_stream())

        sc = score_all(model, val_ad_q, candidate_embs)
        res = ndcg_by_group(sc, val_ad_q, val_ad_ans, candidate_ids, classify)
        results_uniform[gamma] = res

        row = f"  {gamma:>5.1f}  "
        for g in GROUPS:
            nd, n = res.get(g, (0.0, 0))
            row += f"  {fmt(nd):>12}"
        print(row)

    # ── 실험 B: tiered adaptive γ ─────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  실험 B: tiered adaptive γ")
    print(f"  cold=0  /  search-only=γ_search  /  click=γ")
    print(f"{'─'*72}")

    tiered_configs = [
        (0.5, 0.0, "기존 adaptive (click만)"),
        (0.7, 0.5, "최적 tiered (현재 설정)"),
        (0.7, 0.3, "tiered γ=0.7/0.3"),
        (1.0, 0.5, "tiered γ=1.0/0.5"),
        (1.0, 1.0, "tiered γ=1.0/1.0 (interest-only)"),
    ]

    header2 = f"  {'설정':<30}  "
    for g in GROUPS:
        header2 += f"  {g:>12}"
    print(header2)
    print("  " + "─" * (len(header2) - 2))

    for gamma, gs, label in tiered_configs:
        model = make_model(gamma, gs)
        train(model, ds.training_stream())
        sc  = score_all(model, val_ad_q, candidate_embs)
        res = ndcg_by_group(sc, val_ad_q, val_ad_ans, candidate_ids, classify)

        row = f"  {label:<30}  "
        for g in GROUPS:
            nd, n = res.get(g, (0.0, 0))
            row += f"  {fmt(nd):>12}"
        print(row)

    # ── 핵심 비교 테이블 ────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  핵심 비교: γ=0 vs γ=1 vs 최적 (uniform), 그룹별")
    print(f"{'─'*72}")
    print(f"\n  {'그룹':<14} {'n':>5}  {'γ=0.0':>8}  {'γ=1.0':>8}  "
          f"{'최적 uniform':>14}  {'최적 tiered':>12}")
    print("  " + "─" * 68)

    model_opt_u = make_model(0.7)
    train(model_opt_u, ds.training_stream())
    sc_opt_u = score_all(model_opt_u, val_ad_q, candidate_embs)
    res_opt_u = ndcg_by_group(sc_opt_u, val_ad_q, val_ad_ans, candidate_ids, classify)

    model_opt_t = make_model(0.7, 0.5)
    train(model_opt_t, ds.training_stream())
    sc_opt_t = score_all(model_opt_t, val_ad_q, candidate_embs)
    res_opt_t = ndcg_by_group(sc_opt_t, val_ad_q, val_ad_ans, candidate_ids, classify)

    for g in GROUPS:
        nd0, n  = results_uniform[0.0].get(g, (0.0, 0))
        nd1, _  = results_uniform[1.0].get(g, (0.0, 0))
        ndu, _  = res_opt_u.get(g, (0.0, 0))
        ndt, _  = res_opt_t.get(g, (0.0, 0))
        print(f"  {g:<14} {n:>5}  {fmt(nd0):>8}  {fmt(nd1):>8}  "
              f"{fmt(ndu):>14}  {fmt(ndt):>12}")

    print(f"\n  HistCTR baseline NDCG@3 = 0.0211")


if __name__ == "__main__":
    main()
