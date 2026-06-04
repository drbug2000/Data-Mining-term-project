"""
batch_eval.py — 스트리밍 방식 vs 배치 방식 interest vector 비교.

스트리밍(원본):
  - 검색/클릭 이벤트가 올 때마다 soft-assignment로 interest vector를 즉시 업데이트.

배치(신규):
  - 모든 이벤트를 수집한 뒤 클러스터링으로 interest vector를 일괄 구축.
  - 4가지 방법 비교:
      kmeans  : Spherical K-Means (k-means++, 코사인 유사도)
      svd     : top-k 우측 특이벡터 (최대 분산 방향, 서로 직교)
      mean    : 가중 평균 단일 벡터를 k개 복제 (degenerate 기준선)
      diverse : Greedy farthest-point 선택 (공간 최대 커버리지)

평가 지표:
  Task A (click prediction, click_validation_*):
      Accuracy / Precision / Recall / F1 / AUC  (IsClick=1 기준)
  Task B (ad recommendation, ad_validation_*):
      NDCG@3 / MRR / Rank distribution

실행:
    python -X utf8 experiments/batch_eval.py
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
from model.batch_interest import BatchMultiInterestModel
from model.predictor import (
    evaluate_task_a,
    evaluate_task_b_ndcg,
    score_task_a,
    score_task_b,
    train,
)

# ──────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────
DATASET_DIR = ROOT / "../datasets"
SEED = 42

# 기존 baseline_eval.py 의 최종 설정과 동일
CONFIG = ModelConfig(
    k=5,
    alpha_search=0.01,
    alpha_click=0.5,
    alpha_neg=0.0,
    temperature=1.0,
    gamma=0.7,
    gamma_search=0.5,
    threshold=0.5,
)

# 배치 모델에서 클릭 embedding을 몇 배 반복할지
# (검색 대비 클릭 신호를 강조하기 위한 배수)
CLICK_WEIGHT = 5.0


# ──────────────────────────────────────────────────────────────────────
# 출력 헬퍼
# ──────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def subsection(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def f(v: float) -> str:
    return f"{v:.4f}"


def elapsed(t0: float) -> str:
    dt = time.time() - t0
    return f"{dt:.1f}s" if dt >= 1 else f"{dt*1000:.0f}ms"


# ──────────────────────────────────────────────────────────────────────
# 추가 평가 지표
# ──────────────────────────────────────────────────────────────────────

def _mrr(scores_dict: dict, answers: dict, candidate_ids: list[int]) -> float:
    """Mean Reciprocal Rank — 정답 광고 순위의 역수 평균."""
    id_to_idx = {aid: i for i, aid in enumerate(candidate_ids)}
    recips: list[float] = []
    for sid, correct_aid in answers.items():
        if sid not in scores_dict or correct_aid not in id_to_idx:
            continue
        scores = scores_dict[sid]
        correct_score = scores[id_to_idx[correct_aid]]
        rank = int((scores > correct_score).sum()) + 1
        recips.append(1.0 / rank)
    return float(np.mean(recips)) if recips else 0.0


def _constant_scores_a(pairs, value: float) -> list[float]:
    return [value] * len(pairs)


# ──────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # ── 1. 데이터 로드 ────────────────────────────────────────────────
    section("1. 데이터 로드")
    t0 = time.time()
    ds = RecoDataset(DATASET_DIR).load()
    print(ds.summary())
    print(f"\n  로드 시간: {elapsed(t0)}")

    candidate_embs, candidate_ids = ds.all_ad_embs()
    val_clk_q   = ds.val_click_queries()
    val_clk_ans = ds.val_click_answers()
    val_ad_q    = ds.val_ad_queries()
    val_ad_ans  = ds.val_ad_answers()

    n_clicks = val_clk_ans["IsClick"].sum()
    print(f"\n  Task A 정답 분포: click={n_clicks} ({n_clicks/len(val_clk_ans):.2%}), "
          f"no-click={len(val_clk_ans)-n_clicks}")
    print(f"  Task B 후보 광고: {len(candidate_ids):,}개")

    # ── 2. 스트리밍 모델 학습 (원본) ─────────────────────────────────
    section("2. 스트리밍 모델 학습 (원본 MultiInterestModel)")
    streaming = MultiInterestModel(CONFIG)
    t0 = time.time()
    train(streaming, ds.training_stream())
    print(f"  완료: {elapsed(t0)}")

    # ── 3. 배치 모델 데이터 수집 ─────────────────────────────────────
    section("3. 배치 모델 — embedding 수집 (단일 패스)")
    collector = BatchMultiInterestModel(CONFIG, method="kmeans", click_weight=CLICK_WEIGHT)
    t0 = time.time()
    train(collector, ds.training_stream())
    n_search_users = len(collector._raw_search)
    n_click_users  = len(collector._raw_click)
    total_search   = sum(len(v) for v in collector._raw_search.values())
    total_click    = sum(len(v) for v in collector._raw_click.values())
    print(f"  수집 완료: {elapsed(t0)}")
    print(f"  검색 이벤트 보유 유저: {n_search_users:,}명  (총 {total_search:,}개 embedding)")
    print(f"  클릭 이벤트 보유 유저: {n_click_users:,}명  (총 {total_click:,}개 embedding)")
    print(f"  click_weight={CLICK_WEIGHT} → Task B 구축 시 클릭 embedding {int(CLICK_WEIGHT)}배 반복")

    # ── 4. 배치 모델 interest 구축 ────────────────────────────────────
    section("4. 배치 모델 — interest vector 구축")
    BATCH_METHODS = ["kmeans", "svd", "mean", "diverse"]
    batch_models: dict[str, BatchMultiInterestModel] = {}

    for method in BATCH_METHODS:
        m = BatchMultiInterestModel(CONFIG, method=method,
                                    click_weight=CLICK_WEIGHT, seed=SEED)
        # raw embedding 공유 (읽기 전용 — 재수집 없이 재사용)
        m._raw_search      = collector._raw_search
        m._raw_click       = collector._raw_click
        m._searched_users  = collector._searched_users
        m._clicked_users   = collector._clicked_users
        t0 = time.time()
        m.build_interests()
        print(f"  Batch-{method:<8}: {elapsed(t0)}")
        batch_models[method] = m

    # ── 5. Query-only 기준선 ──────────────────────────────────────────
    # gamma=0이므로 interest vector를 사용하지 않음 → 훈련 불필요
    query_only = MultiInterestModel(
        ModelConfig(gamma=0.0, gamma_search=0.0, threshold=CONFIG.threshold)
    )

    # ── 6. 모델 목록 정리 ─────────────────────────────────────────────
    models: dict[str, object] = {
        "Streaming (원본)"   : streaming,
        "Batch-KMeans"       : batch_models["kmeans"],
        "Batch-SVD"          : batch_models["svd"],
        "Batch-Mean"         : batch_models["mean"],
        "Batch-Diverse"      : batch_models["diverse"],
        "Query-only (γ=0)"   : query_only,
    }

    # ── 7. Task A 평가 (Click Prediction) ────────────────────────────
    section("5. Task A — Click Prediction  (20,000 pairs, threshold=0.5)")
    thr = CONFIG.threshold

    rows_a: list[tuple[str, dict, str]] = []
    for name, model in models.items():
        t0 = time.time()
        sc = score_task_a(model, val_clk_q)
        m  = evaluate_task_a(sc, val_clk_ans, thr)
        rows_a.append((name, m, elapsed(t0)))

    # 상수 기준선 추가 (점수 계산 불필요)
    rows_a.append(("Always-0 (no click)",
                   evaluate_task_a(_constant_scores_a(val_clk_q, 0.0), val_clk_ans, thr), "-"))
    rows_a.append(("Always-1 (all click)",
                   evaluate_task_a(_constant_scores_a(val_clk_q, 1.0), val_clk_ans, thr), "-"))

    col_w = 22
    hdr = (f"  {'모델':<{col_w}} {'Accuracy':>10} {'Precision':>10} "
           f"{'Recall':>8} {'F1':>8} {'AUC':>8}  {'시간':>5}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for name, m, t in rows_a:
        print(f"  {name:<{col_w}} {f(m['accuracy']):>10} {f(m['precision']):>10} "
              f"{f(m['recall']):>8} {f(m['f1']):>8} {f(m['auc']):>8}  {t:>5}")
    print(f"\n  * Precision / Recall / F1: IsClick=1 (클릭) 클래스 기준")

    # ── Task A 상세: confusion matrix (주요 모델만) ─────────────────
    subsection("Task A — Confusion Matrix (주요 모델)")
    for name, m, _ in rows_a[:6]:  # 상수 기준선 제외
        pc = m["per_class"]
        c0 = pc.get(0, {})
        c1 = pc.get(1, {"tp": 0, "fp": 0, "fn": 0, "support": 0})
        TN = c0.get("tp", 0); FP = c1["fp"]; FN = c1["fn"]; TP = c1["tp"]
        print(f"\n  [{name}]")
        print(f"    Pred→  0        1      | 지표")
        print(f"    True0  {TN:>6}  {FP:>6}   | Precision={f(c1['precision'])}  Recall={f(c1['recall'])}")
        print(f"    True1  {FN:>6}  {TP:>6}   | F1={f(c1['f1'])}  AUC={f(m['auc'])}")

    # ── 8. Task B 평가 (Ad Recommendation) ───────────────────────────
    section("6. Task B — Ad Recommendation  (214 queries, 17,518 candidates)")

    rows_b: list[tuple[str, dict, str]] = []
    for name, model in models.items():
        t0 = time.time()
        sc_dict = score_task_b(model, val_ad_q, candidate_embs)
        m = evaluate_task_b_ndcg(sc_dict, val_ad_ans, candidate_ids)
        m["mrr"] = _mrr(sc_dict, val_ad_ans, candidate_ids)
        rows_b.append((name, m, elapsed(t0)))

    # 랜덤 기준선
    sc_rand = {ev.search_id: rng.random(len(candidate_ids)).astype(np.float32)
               for ev in val_ad_q}
    m_rand = evaluate_task_b_ndcg(sc_rand, val_ad_ans, candidate_ids)
    m_rand["mrr"] = _mrr(sc_rand, val_ad_ans, candidate_ids)
    rows_b.append(("Random", m_rand, "-"))

    hdr = (f"  {'모델':<{col_w}} {'NDCG@3':>8} {'MRR':>8} "
           f"{'Rank1':>6} {'Rank2':>6} {'Rank3':>6} {'>Rank3':>7}  {'시간':>5}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for name, m, t in rows_b:
        rd = m["rank_dist"]
        print(f"  {name:<{col_w}} {f(m['ndcg@3']):>8} {f(m['mrr']):>8} "
              f"{rd[1]:>6} {rd[2]:>6} {rd[3]:>6} {rd['>3']:>7}  {t:>5}")

    # ── 9. 종합 요약 ──────────────────────────────────────────────────
    section("7. 종합 비교 요약")

    # 모델명 → (Task A F1, Task A AUC, Task B NDCG@3, Task B MRR)
    model_names = [r[0] for r in rows_a[:6]]
    a_metrics   = {r[0]: r[1] for r in rows_a[:6]}
    b_metrics   = {r[0]: r[1] for r in rows_b if r[0] in model_names}

    hdr = (f"  {'모델':<{col_w}} {'TaskA_F1':>10} {'TaskA_AUC':>10} "
           f"{'TaskB_NDCG3':>12} {'TaskB_MRR':>10}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for name in model_names:
        ma = a_metrics.get(name, {})
        mb = b_metrics.get(name, {})
        print(f"  {name:<{col_w}} "
              f"{f(ma.get('f1', 0)):>10} "
              f"{f(ma.get('auc', 0)):>10} "
              f"{f(mb.get('ndcg@3', 0)):>12} "
              f"{f(mb.get('mrr', 0)):>10}")

    print(f"\n  Config: k={CONFIG.k}, gamma={CONFIG.gamma}, gamma_search={CONFIG.gamma_search}, "
          f"threshold={CONFIG.threshold}")
    print(f"  Batch click_weight={CLICK_WEIGHT}  "
          f"(클릭 embedding을 검색 대비 {CLICK_WEIGHT:.0f}배 반복)")

    # ── 10. 방법론 분석 ───────────────────────────────────────────────
    section("8. 방법론 분석")

    # Task A F1 기준 정렬
    sorted_by_f1 = sorted(
        [(n, a_metrics[n]["f1"], a_metrics[n]["auc"],
          b_metrics.get(n, {}).get("ndcg@3", 0), b_metrics.get(n, {}).get("mrr", 0))
         for n in model_names],
        key=lambda x: x[1], reverse=True
    )

    print("\n  [Task A F1 기준 순위]")
    for rank, (name, f1, auc, ndcg, mrr_v) in enumerate(sorted_by_f1, 1):
        print(f"    {rank}위  {name:<{col_w}}  F1={f1:.4f}  AUC={auc:.4f}  "
              f"NDCG@3={ndcg:.4f}  MRR={mrr_v:.4f}")

    # Task B NDCG@3 기준 정렬
    sorted_by_ndcg = sorted(sorted_by_f1, key=lambda x: x[3], reverse=True)
    print("\n  [Task B NDCG@3 기준 순위]")
    for rank, (name, f1, auc, ndcg, mrr_v) in enumerate(sorted_by_ndcg, 1):
        print(f"    {rank}위  {name:<{col_w}}  NDCG@3={ndcg:.4f}  MRR={mrr_v:.4f}  "
              f"F1={f1:.4f}  AUC={auc:.4f}")

    # 스트리밍 대비 배치 모델 개선율
    stream_f1   = a_metrics["Streaming (원본)"]["f1"]
    stream_ndcg = b_metrics.get("Streaming (원본)", {}).get("ndcg@3", 0)

    # method → 모델 딕셔너리 키 매핑 (대소문자 포함)
    method_to_name = {
        "kmeans" : "Batch-KMeans",
        "svd"    : "Batch-SVD",
        "mean"   : "Batch-Mean",
        "diverse": "Batch-Diverse",
    }

    print("\n  [스트리밍 대비 배치 모델 개선율]")
    print(f"  {'모델':<{col_w}}  {'ΔF1':>10}  {'ΔNDCG@3':>10}")
    for method in BATCH_METHODS:
        bname = method_to_name[method]
        bf1   = a_metrics.get(bname, {}).get("f1", 0)
        bndcg = b_metrics.get(bname, {}).get("ndcg@3", 0)
        delta_f1   = bf1   - stream_f1
        delta_ndcg = bndcg - stream_ndcg
        sign_f1   = "+" if delta_f1   >= 0 else ""
        sign_ndcg = "+" if delta_ndcg >= 0 else ""
        print(f"  {bname:<{col_w}}  "
              f"{sign_f1}{delta_f1:+.4f}  {sign_ndcg}{delta_ndcg:+.4f}")

    print()


if __name__ == "__main__":
    main()
