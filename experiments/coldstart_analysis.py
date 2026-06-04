"""
coldstart_analysis.py — Task B (ad recommendation, NDCG@3)에서
훈련 기록이 있는 유저 vs cold-start 유저의 성능 비교.

유저 분류:
  - cold      : 훈련 데이터에 전혀 등장하지 않음
  - search    : 훈련 데이터에 검색 기록만 있음 (클릭 없음)
  - click     : 훈련 데이터에 클릭 기록 1건 이상 있음

실행:
    python -X utf8 experiments/coldstart_analysis.py
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import RecoDataset
from model import ModelConfig, MultiInterestModel
from model.predictor import score_task_b, train

DATASET_DIR = ROOT / "../datasets"
SEED = 42


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def ndcg3(rank: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank <= 3 else 0.0


def evaluate_by_group(
    scores_dict: dict[int, np.ndarray],
    answers: dict[int, int],
    candidate_ids: list[int],
    group_of: dict[int, str],        # user_id → group label
) -> dict[str, dict]:
    """그룹별 NDCG@3, rank 분포를 계산한다."""
    id_to_idx = {aid: i for i, aid in enumerate(candidate_ids)}
    common = set(scores_dict) & set(answers)

    results: dict[str, list] = defaultdict(list)
    rank_dists: dict[str, dict] = defaultdict(lambda: {1: 0, 2: 0, 3: 0, ">3": 0})

    # user_id를 얻기 위해 val_ad_q의 search_id → user_id 매핑 필요
    # scores_dict의 key는 search_id, group_of는 user_id 기준
    # → search_id를 user_id로 변환하는 dict를 밖에서 넘겨받는다
    return results, rank_dists


def main() -> None:
    np.random.seed(SEED)

    section("1. 데이터 로드 및 유저 분류")
    ds = RecoDataset(DATASET_DIR).load()
    candidate_embs, candidate_ids = ds.all_ad_embs()
    id_to_idx = {aid: i for i, aid in enumerate(candidate_ids)}

    val_ad_q   = ds.val_ad_queries()
    val_ad_ans = ds.val_ad_answers()   # dict: SearchID → AdID

    # ── 훈련 데이터에서 유저별 검색 횟수 / 클릭 횟수 집계 ──────────────
    user_search_cnt: dict[int, int] = defaultdict(int)
    user_click_cnt:  dict[int, int] = defaultdict(int)

    for ev in ds.training_stream():
        user_search_cnt[ev.user_id] += 1
        for ad in ev.ads:
            if ad.is_click:
                user_click_cnt[ev.user_id] += 1

    # ── 유저 분류 ────────────────────────────────────────────────────────
    def classify(uid: int) -> str:
        if user_search_cnt[uid] == 0:
            return "cold"
        if user_click_cnt[uid] == 0:
            return "search-only"
        return "click"

    # val_ad_q 유저 분류 현황
    group_counts: dict[str, int] = defaultdict(int)
    sid_to_uid = {ev.search_id: ev.user_id for ev in val_ad_q}
    for ev in val_ad_q:
        group_counts[classify(ev.user_id)] += 1

    print(f"\n  val_ad_queries 214개 유저 분류:")
    for g in ["cold", "search-only", "click"]:
        n = group_counts[g]
        print(f"    {g:<14} : {n:>4}개 ({n/214:.1%})")

    # 훈련 데이터 통계
    print(f"\n  훈련 데이터 통계 (전체 유저 {len(user_search_cnt):,}명):")
    n_click_users  = sum(1 for v in user_click_cnt.values() if v > 0)
    n_search_only  = sum(1 for uid, v in user_search_cnt.items()
                         if v > 0 and user_click_cnt[uid] == 0)
    print(f"    클릭 이력 유저  : {n_click_users:,}명  "
          f"(평균 클릭 {np.mean([v for v in user_click_cnt.values() if v>0]):.2f}회)")
    print(f"    검색만 있는 유저: {n_search_only:,}명  "
          f"(평균 검색 {np.mean([v for uid,v in user_search_cnt.items() if v>0 and user_click_cnt[uid]==0]):.1f}회)")

    # ── 모델 훈련 ────────────────────────────────────────────────────────
    section("2. 모델 훈련 (adaptive gamma=0.5)")
    cfg = ModelConfig(k=5, alpha_search=0.01, alpha_click=0.5,
                      alpha_neg=0.0, gamma=0.5, threshold=0.5)
    model = MultiInterestModel(cfg)
    train(model, ds.training_stream())

    # ── 스코어 계산 ──────────────────────────────────────────────────────
    section("3. Task B NDCG@3 — 그룹별 성능 비교")
    scores_dict = score_task_b(model, val_ad_q, candidate_embs)

    # 그룹별 NDCG 누적
    group_ndcg:  dict[str, list[float]] = defaultdict(list)
    group_ranks: dict[str, dict]        = {
        g: {1: 0, 2: 0, 3: 0, ">3": 0}
        for g in ["cold", "search-only", "click", "ALL"]
    }

    for ev in val_ad_q:
        sid = ev.search_id
        uid = ev.user_id
        correct_aid = val_ad_ans.get(sid)
        if correct_aid is None or correct_aid not in id_to_idx:
            continue
        if sid not in scores_dict:
            continue

        scores = scores_dict[sid]
        rank = int((scores > scores[id_to_idx[correct_aid]]).sum()) + 1
        nd   = ndcg3(rank)
        grp  = classify(uid)

        group_ndcg[grp].append(nd)
        group_ndcg["ALL"].append(nd)

        rd_key = rank if rank <= 3 else ">3"
        group_ranks[grp][rd_key] += 1
        group_ranks["ALL"][rd_key] += 1

    # ── 결과 출력 ────────────────────────────────────────────────────────
    print(f"\n  {'그룹':<14} {'n':>5} {'NDCG@3':>9} {'Rank1':>7} {'Rank2':>7} "
          f"{'Rank3':>7} {'>Rank3':>7}")
    print("  " + "─" * 60)

    for grp in ["cold", "search-only", "click", "ALL"]:
        vals = group_ndcg[grp]
        if not vals:
            continue
        rd = group_ranks[grp]
        n  = len(vals)
        print(f"  {grp:<14} {n:>5} {np.mean(vals):>9.4f} "
              f"{rd[1]:>7} {rd[2]:>7} {rd[3]:>7} {rd['>3']:>7}")

    # ── 세부 분석: 검색 횟수 구간별 ─────────────────────────────────────
    section("4. 검색 횟수 구간별 NDCG@3 (search-only + click 유저)")
    buckets = [(1, 1), (2, 3), (4, 9), (10, 29), (30, 999)]
    print(f"\n  {'검색 횟수':^12} {'n':>5} {'NDCG@3':>9} {'Rank1':>7} {'>3':>7}")
    print("  " + "─" * 45)
    for lo, hi in buckets:
        vals = []
        r1 = 0
        rn = 0
        for ev in val_ad_q:
            sid = ev.search_id
            uid = ev.user_id
            sc = user_search_cnt[uid]
            if not (lo <= sc <= hi):
                continue
            correct_aid = val_ad_ans.get(sid)
            if correct_aid is None or correct_aid not in id_to_idx:
                continue
            if sid not in scores_dict:
                continue
            scores = scores_dict[sid]
            rank = int((scores > scores[id_to_idx[correct_aid]]).sum()) + 1
            nd   = ndcg3(rank)
            vals.append(nd)
            if rank == 1:
                r1 += 1
            elif rank > 3:
                rn += 1
        if vals:
            label = f"{lo}" if lo == hi else f"{lo}~{hi}"
            print(f"  {label:^12} {len(vals):>5} {np.mean(vals):>9.4f} "
                  f"{r1:>7} {rn:>7}")

    # ── Query-only (gamma=0) 비교 ─────────────────────────────────────
    section("5. MultiInterest vs Query-only — 그룹별 비교")
    from model import MultiInterestModel as MIM
    model_qonly = MIM(ModelConfig(k=5, alpha_search=0.01, alpha_click=0.5,
                                  gamma=0.0))
    train(model_qonly, ds.training_stream())
    scores_qonly = score_task_b(model_qonly, val_ad_q, candidate_embs)

    print(f"\n  {'그룹':<14} {'n':>5}  {'MultiInterest':>14}  {'Query-only':>12}  {'Delta':>8}")
    print("  " + "─" * 60)
    for grp in ["cold", "search-only", "click", "ALL"]:
        vals_mi = group_ndcg[grp]
        if not vals_mi:
            continue
        vals_qo = []
        for ev in val_ad_q:
            sid = ev.search_id
            uid = ev.user_id
            if classify(uid) != grp and grp != "ALL":
                continue
            if grp == "ALL" or True:
                correct_aid = val_ad_ans.get(sid)
                if correct_aid is None or correct_aid not in id_to_idx:
                    continue
                if sid not in scores_qonly:
                    continue
                sc = scores_qonly[sid]
                rank = int((sc > sc[id_to_idx[correct_aid]]).sum()) + 1
                vals_qo.append(ndcg3(rank))

        if not vals_qo:
            continue
        mi  = np.mean(vals_mi)
        qo  = np.mean(vals_qo)
        print(f"  {grp:<14} {len(vals_mi):>5}  {mi:>14.4f}  {qo:>12.4f}  "
              f"{mi-qo:>+8.4f}")

    section("요약")
    print("""
  cold start 유저  : 훈련 데이터에 검색 기록 없음 → interest vector = 랜덤
  search-only 유저 : 검색 기록 있으나 클릭 없음   → interest vector = 검색 방향 편향
  click 유저       : 클릭 기록 1건 이상           → interest vector = 클릭 선호 반영
  ALL              : 전체 214 queries 평균

  HistCTR baseline NDCG@3 = 0.0211
    """)


if __name__ == "__main__":
    main()
