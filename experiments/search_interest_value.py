"""
search_interest_value.py
"검색 기반 interest vector가 Task B에 도움이 되는가?" 검증.

현재 adaptive gamma는 클릭 이력이 있는 유저에게만 gamma>0을 적용한다.
search-only 유저에게도 gamma>0을 적용하면 성능이 오르는가, 내려가는가?

실험:
  A) adaptive gamma 기준 = click  (현재 설계)
  B) adaptive gamma 기준 = search (검색만 있어도 gamma 활성화)
  C) gamma=0 완전 비활성 (query-only baseline)

각 설정을 click / search-only / cold 그룹별로 평가.
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


def ndcg3(rank: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank <= 3 else 0.0


def score_with_gamma_rule(
    model: MultiInterestModel,
    val_ad_q,
    candidate_embs: np.ndarray,
    id_to_idx: dict,
    gamma_rule: str,                    # "click" | "search" | "none"
    user_search_cnt: dict,
    user_click_cnt: dict,
) -> dict[int, np.ndarray]:
    """gamma_rule에 따라 effective_gamma를 직접 제어해 스코어를 계산한다."""
    from model.interest import _l2_normalize

    C = _l2_normalize(candidate_embs)          # (N, dim)
    scores_dict = {}

    for ev in val_ad_q:
        uid = ev.user_id
        q   = _l2_normalize(ev.search_emb)     # (dim,)
        query_ad_sim = C @ q                   # (N,)

        # gamma 결정
        if gamma_rule == "none":
            eff_gamma = 0.0
        elif gamma_rule == "click":
            eff_gamma = model.config.gamma if user_click_cnt[uid] > 0 else 0.0
        elif gamma_rule == "search":
            eff_gamma = model.config.gamma if user_search_cnt[uid] > 0 else 0.0
        else:
            eff_gamma = 0.0

        if eff_gamma == 0.0:
            scores_dict[ev.search_id] = query_ad_sim
        else:
            V = _l2_normalize(model._get_or_init_store(model._interests, uid))
            interest_ad_sim = (V @ C.T).max(axis=0)
            scores_dict[ev.search_id] = (
                (1 - eff_gamma) * query_ad_sim + eff_gamma * interest_ad_sim
            )

    return scores_dict


def evaluate_groups(scores_dict, val_ad_q, val_ad_ans, id_to_idx,
                    user_search_cnt, user_click_cnt):
    def classify(uid):
        if user_search_cnt[uid] == 0:  return "cold"
        if user_click_cnt[uid]  == 0:  return "search-only"
        return "click"

    group_ndcg = defaultdict(list)
    for ev in val_ad_q:
        sid = ev.search_id
        correct_aid = val_ad_ans.get(sid)
        if correct_aid is None or correct_aid not in id_to_idx:
            continue
        sc    = scores_dict[sid]
        rank  = int((sc > sc[id_to_idx[correct_aid]]).sum()) + 1
        grp   = classify(ev.user_id)
        group_ndcg[grp].append(ndcg3(rank))
        group_ndcg["ALL"].append(ndcg3(rank))

    return {g: (np.mean(v), len(v)) for g, v in group_ndcg.items()}


def main():
    np.random.seed(SEED)

    print("\n데이터 로드 중...")
    ds = RecoDataset(DATASET_DIR).load()
    candidate_embs, candidate_ids = ds.all_ad_embs()
    id_to_idx = {aid: i for i, aid in enumerate(candidate_ids)}
    val_ad_q   = ds.val_ad_queries()
    val_ad_ans = ds.val_ad_answers()

    # 유저별 훈련 기록 집계
    user_search_cnt: dict[int, int] = defaultdict(int)
    user_click_cnt:  dict[int, int] = defaultdict(int)
    for ev in ds.training_stream():
        user_search_cnt[ev.user_id] += 1
        for ad in ev.ads:
            if ad.is_click:
                user_click_cnt[ev.user_id] += 1

    # 모델 훈련 (공통)
    print("모델 훈련 중...")
    cfg   = ModelConfig(k=5, alpha_search=0.01, alpha_click=0.5,
                        alpha_neg=0.0, gamma=0.5, threshold=0.5)
    model = MultiInterestModel(cfg)
    train(model, ds.training_stream())

    # 세 가지 gamma 규칙으로 스코어 계산
    rules = {
        "A: gamma 기준=click  (현재)": "click",
        "B: gamma 기준=search (실험)": "search",
        "C: gamma=0, query-only":      "none",
    }

    print("\n")
    print(f"  {'설정':<32} {'cold':>10} {'search-only':>12} {'click':>8} {'ALL':>8}")
    print("  " + "─" * 74)

    results = {}
    for label, rule in rules.items():
        sc = score_with_gamma_rule(
            model, val_ad_q, candidate_embs, id_to_idx,
            rule, user_search_cnt, user_click_cnt
        )
        res = evaluate_groups(sc, val_ad_q, val_ad_ans, id_to_idx,
                              user_search_cnt, user_click_cnt)
        results[label] = res

        def fmt(g):
            if g not in res: return "   -  "
            v, n = res[g]
            return f"{v:.4f}({n})"

        print(f"  {label:<32} {fmt('cold'):>10} {fmt('search-only'):>12} "
              f"{fmt('click'):>8} {fmt('ALL'):>8}")

    # ── 핵심 비교: search-only 유저에서 A vs B vs C ──────────────────────
    print("\n\n  [search-only 유저 집중 비교]")
    print(f"  검색 기록만 있는 유저 (n=74)에게 interest vector를 쓰면?")
    print()

    ndcg_A = results["A: gamma 기준=click  (현재)"]["search-only"][0]
    ndcg_B = results["B: gamma 기준=search (실험)"]["search-only"][0]
    ndcg_C = results["C: gamma=0, query-only"]["search-only"][0]

    print(f"    C (query-only, gamma=0):           NDCG@3 = {ndcg_C:.4f}")
    print(f"    A (현재: search-only는 gamma=0):   NDCG@3 = {ndcg_A:.4f}  (= C, 확인)")
    print(f"    B (실험: search-only에 gamma=0.5): NDCG@3 = {ndcg_B:.4f}  "
          f"{'↑ 향상' if ndcg_B > ndcg_C + 0.001 else ('↓ 저하' if ndcg_B < ndcg_C - 0.001 else '= 동일')}")
    print()

    delta = ndcg_B - ndcg_C
    if delta < -0.002:
        print("  → 검색 기반 interest vector를 적용하면 성능이 저하됨")
        print("    = '검색으로 쌓인 embedding은 Task B user profile로는 노이즈'")
    elif delta > 0.002:
        print("  → 검색 기반 interest vector를 적용하면 성능이 향상됨")
        print("    = '검색 embedding도 의미있는 user preference 신호를 담고 있음'")
    else:
        print("  → 검색 기반 interest vector는 성능에 거의 영향 없음")
        print("    = 'query-ad similarity가 이미 검색 신호를 충분히 커버함'")

    # ── cold 유저 비교: 아무 기록 없는 경우 ──────────────────────────────
    ndcg_cold_A = results["A: gamma 기준=click  (현재)"]["cold"][0]
    ndcg_cold_C = results["C: gamma=0, query-only"]["cold"][0]
    print(f"\n  [cold 유저 확인]")
    print(f"    A = C = {ndcg_cold_A:.4f}  (아무 기록 없으므로 모든 설정에서 동일)")

    print("\n  HistCTR baseline NDCG@3 = 0.0211")


if __name__ == "__main__":
    main()
