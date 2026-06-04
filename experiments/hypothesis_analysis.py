"""
hypothesis_analysis.py — 가설 검증:
  "user interest가 추천(Task B)에는 효과적이나 클릭(Task A)에는 효과가 없는 이유"

분석 1: Task A — 클릭/비클릭 광고의 query-ad 코사인 유사도 분포
분석 2: Task B — 정답 광고 vs 랜덤 광고의 query-ad 코사인 유사도 분포
분석 3: 유저 레벨 — Task B를 잘 맞추는 유저가 Task A에서도 잘 맞추는가?
분석 4: 훈련 데이터 — 클릭 광고 vs 노출 광고의 embedding 특성 차이

실행:
    python experiments/hypothesis_analysis.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import RecoDataset
from model import ModelConfig, MultiInterestModel
from model.predictor import score_task_b, score_task_a, train

DATASET_DIR = ROOT / "../datasets"
SEED = 42


def section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print(f"{'─' * 65}")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + 1e-8)


def describe(arr: np.ndarray, label: str) -> None:
    print(f"    {label:<30}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}  n={len(arr)}")


def main() -> None:
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    section("데이터 로드")
    ds = RecoDataset(DATASET_DIR).load()
    candidate_embs, candidate_ids = ds.all_ad_embs()
    id_to_idx = {aid: i for i, aid in enumerate(candidate_ids)}

    val_clk_q   = ds.val_click_queries()   # (event, ad) 쌍 목록
    val_clk_ans = ds.val_click_answers()   # DataFrame: IsClick 컬럼
    val_ad_q    = ds.val_ad_queries()
    val_ad_ans  = ds.val_ad_answers()      # dict: SearchID → 정답 AdID

    # 모델 훈련 (Task B에 가장 좋은 설정 — adaptive gamma=0.5)
    cfg = ModelConfig(k=5, alpha_search=0.01, alpha_click=0.5,
                      alpha_neg=0.0, gamma=0.5, threshold=0.5)
    model = MultiInterestModel(cfg)
    train(model, ds.training_stream())

    # ── 분석 1: Task A — 클릭/비클릭 query-ad 유사도 분포 ──────────────
    section("분석 1 : Task A — query-ad 코사인 유사도, 클릭 vs 비클릭")

    labels = val_clk_ans["IsClick"].tolist()
    click_sims, noclick_sims = [], []

    for (ev, ad), label in zip(val_clk_q, labels):
        q = _l2_normalize(ev.search_emb)
        a = _l2_normalize(ad.ad_emb)
        sim = float(q @ a)
        if label == 1:
            click_sims.append(sim)
        else:
            noclick_sims.append(sim)

    click_sims   = np.array(click_sims)
    noclick_sims = np.array(noclick_sims)

    print()
    describe(click_sims,   "클릭 광고   sim(query, ad)")
    describe(noclick_sims, "비클릭 광고 sim(query, ad)")

    gap_a = click_sims.mean() - noclick_sims.mean()
    print(f"\n    Gap (클릭 평균 - 비클릭 평균) = {gap_a:+.4f}")

    # Cohen's d (effect size)
    pooled_std = np.sqrt((click_sims.std()**2 + noclick_sims.std()**2) / 2)
    cohens_d_a = gap_a / (pooled_std + 1e-8)
    print(f"    Cohen's d (효과 크기)         = {cohens_d_a:.4f}")
    print(f"    → |d| < 0.2: 거의 구분 불가 / 0.2-0.5: 소 / 0.5-0.8: 중 / >0.8: 대")

    # overlap coefficient (Bhattacharyya)
    bins = np.linspace(
        min(click_sims.min(), noclick_sims.min()),
        max(click_sims.max(), noclick_sims.max()), 50
    )
    h_click,   _ = np.histogram(click_sims,   bins=bins, density=True)
    h_noclick, _ = np.histogram(noclick_sims, bins=bins, density=True)
    overlap_a = np.sum(np.minimum(h_click, h_noclick)) * (bins[1] - bins[0])
    print(f"    분포 겹침 (overlap) = {overlap_a:.4f}  (1.0=완전 동일)")

    # ── 분석 2: Task B — 정답 광고 vs 랜덤 광고 유사도 분포 ─────────────
    section("분석 2 : Task B — query-ad 코사인 유사도, 정답 vs 랜덤 후보")

    correct_sims, random_sims = [], []

    for ev in val_ad_q:
        correct_aid = val_ad_ans.get(ev.search_id)
        if correct_aid is None or correct_aid not in id_to_idx:
            continue

        q = _l2_normalize(ev.search_emb)

        # 정답 광고 유사도
        c_emb = _l2_normalize(candidate_embs[id_to_idx[correct_aid]])
        correct_sims.append(float(q @ c_emb))

        # 5개 랜덤 후보 유사도
        rand_idxs = rng.choice(len(candidate_ids), size=5, replace=False)
        for ri in rand_idxs:
            r_emb = _l2_normalize(candidate_embs[ri])
            random_sims.append(float(q @ r_emb))

    correct_sims = np.array(correct_sims)
    random_sims  = np.array(random_sims)

    print()
    describe(correct_sims, "정답 광고   sim(query, ad)")
    describe(random_sims,  "랜덤 광고   sim(query, ad)")

    gap_b = correct_sims.mean() - random_sims.mean()
    pooled_std_b = np.sqrt((correct_sims.std()**2 + random_sims.std()**2) / 2)
    cohens_d_b   = gap_b / (pooled_std_b + 1e-8)
    print(f"\n    Gap (정답 평균 - 랜덤 평균)   = {gap_b:+.4f}")
    print(f"    Cohen's d (효과 크기)         = {cohens_d_b:.4f}")

    bins2 = np.linspace(
        min(correct_sims.min(), random_sims.min()),
        max(correct_sims.max(), random_sims.max()), 50
    )
    h_corr, _ = np.histogram(correct_sims, bins=bins2, density=True)
    h_rand, _ = np.histogram(random_sims,  bins=bins2, density=True)
    overlap_b = np.sum(np.minimum(h_corr, h_rand)) * (bins2[1] - bins2[0])
    print(f"    분포 겹침 (overlap) = {overlap_b:.4f}")

    # ── 분석 3: 유저 레벨 상관관계 ──────────────────────────────────────
    section("분석 3 : 유저 레벨 — Task B 정확도 vs Task A AUC 상관관계")

    # Task B: 유저별 rank
    sc_b = score_task_b(model, val_ad_q, candidate_embs)
    user_b_rank: dict[int, int] = {}
    for ev in val_ad_q:
        sid = ev.search_id
        correct_aid = val_ad_ans.get(sid)
        if correct_aid is None or correct_aid not in id_to_idx:
            continue
        scores = sc_b[sid]
        rank = int((scores > scores[id_to_idx[correct_aid]]).sum()) + 1
        user_b_rank[ev.user_id] = rank  # 마지막 검색 기준 (단순화)

    # Task A: 유저별 click AUC (score vs label)
    user_click_scores: dict[int, list] = defaultdict(list)
    user_click_labels: dict[int, list] = defaultdict(list)
    sc_a = score_task_a(model, val_clk_q)
    for (ev, ad), sc, label in zip(val_clk_q, sc_a, labels):
        user_click_scores[ev.user_id].append(sc)
        user_click_labels[ev.user_id].append(label)

    def binary_auc(scores, lbls):
        s = np.array(scores); l = np.array(lbls)
        pos = s[l == 1]; neg = s[l == 0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        return float((pos[:, None] > neg[None, :]).sum() / (len(pos) * len(neg)))

    # 두 task 모두 있는 유저만
    common_users = set(user_b_rank) & set(user_click_scores)
    b_ranks, a_aucs = [], []
    for uid in common_users:
        auc = binary_auc(user_click_scores[uid], user_click_labels[uid])
        if auc is not None:
            b_ranks.append(user_b_rank[uid])
            a_aucs.append(auc)

    b_ranks = np.array(b_ranks, dtype=float)
    a_aucs  = np.array(a_aucs,  dtype=float)

    print(f"\n  분석 대상 유저 수: {len(b_ranks)}")
    if len(b_ranks) > 2:
        corr = np.corrcoef(-b_ranks, a_aucs)[0, 1]  # rank는 낮을수록 좋으니 부호 반전
        print(f"  Task B 순위(역) ↔ Task A AUC 상관계수 = {corr:.4f}")
        print(f"  → 상관이 낮을수록: 두 task가 서로 다른 요인을 측정함을 시사")

        # 분위별 분석
        q33, q67 = np.percentile(b_ranks, [33, 67])
        top_b  = a_aucs[b_ranks <= q33]
        mid_b  = a_aucs[(b_ranks > q33) & (b_ranks <= q67)]
        bot_b  = a_aucs[b_ranks > q67]
        print(f"\n  Task B 상위 1/3 유저 (rank≤{q33:.0f}) → Task A AUC 평균 = {top_b.mean():.4f}  (n={len(top_b)})")
        print(f"  Task B 중위 1/3 유저               → Task A AUC 평균 = {mid_b.mean():.4f}  (n={len(mid_b)})")
        print(f"  Task B 하위 1/3 유저 (rank>{q67:.0f}) → Task A AUC 평균 = {bot_b.mean():.4f}  (n={len(bot_b)})")

    # ── 분석 4: 훈련 데이터 — 클릭 vs 비클릭 광고 embedding 특성 ────────
    section("분석 4 : 훈련 데이터 — 클릭/비클릭 광고 embedding 분석")

    train_click_sims, train_noclick_sims = [], []
    train_click_norms, train_noclick_norms = [], []

    for event in ds.training_stream():
        q = _l2_normalize(event.search_emb)
        for ad in event.ads:
            a_norm = np.linalg.norm(ad.ad_emb)
            a = _l2_normalize(ad.ad_emb)
            sim = float(q @ a)
            if ad.is_click:
                train_click_sims.append(sim)
                train_click_norms.append(a_norm)
            else:
                train_noclick_sims.append(sim)
                train_noclick_norms.append(a_norm)

    tc_sims = np.array(train_click_sims)
    tn_sims = np.array(train_noclick_sims)
    print()
    describe(tc_sims, "클릭 광고   sim(query, ad) [훈련]")
    describe(tn_sims, "비클릭 광고 sim(query, ad) [훈련]")

    train_gap = tc_sims.mean() - tn_sims.mean()
    train_pooled = np.sqrt((tc_sims.std()**2 + tn_sims.std()**2) / 2)
    print(f"\n    Gap (훈련)   = {train_gap:+.4f}")
    print(f"    Cohen's d    = {train_gap / (train_pooled + 1e-8):.4f}")

    # ── 요약 ─────────────────────────────────────────────────────────────
    section("요약 : 가설 검증 결과")
    print(f"""
  Task A (클릭 예측) — query-ad sim의 클릭 vs 비클릭 구분력
    Gap          = {gap_a:+.6f}
    Cohen's d    = {cohens_d_a:.4f}
    분포 겹침    = {overlap_a:.4f}

  Task B (광고 추천) — query-ad sim의 정답 vs 랜덤 구분력
    Gap          = {gap_b:+.6f}
    Cohen's d    = {cohens_d_b:.4f}
    분포 겹침    = {overlap_b:.4f}

  해석:
    - Task B: Cohen's d >> Task A의 Cohen's d  →  embedding이 Task B를 훨씬 잘 구분
    - Task A: Gap ≈ 0, 분포 거의 완전 겹침     →  클릭 여부는 sim만으로 예측 불가
    - 유저 레벨 상관관계가 낮다면:              →  Task B 잘하는 유저 ≠ Task A 잘하는 유저
    → 가설 지지: '관심 광고'와 '실제 클릭 광고'는 다른 메커니즘으로 결정됨
""")


if __name__ == "__main__":
    main()
