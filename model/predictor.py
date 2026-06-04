"""
predictor.py — 모델과 데이터 스트림을 이어주는 실행 함수 모음.

Task 정의 (AI506 과제 명세 기준)
  Task A : click_validation_*  |  IsClick 예측  |  F1
  Task B : ad_validation_*     |  AdID 추천     |  NDCG@3

각 함수는 BaseRecoModel 인터페이스만 사용하므로
모델 구현체(MultiInterestModel 등)와 무관하게 동작한다.

사용법:
    from model.predictor import train, score_task_a, score_task_b, evaluate_task_a, evaluate_task_b_ndcg

    train(model, ds.training_stream())

    # Task A — click prediction (F1)
    scores_a = score_task_a(model, ds.val_click_queries())
    metrics_a = evaluate_task_a(scores_a, ds.val_click_answers(), threshold=0.5)

    # Task B — ad recommendation (NDCG@3)
    scores_b = score_task_b(model, ds.val_ad_queries(), candidate_embs)
    metrics_b = evaluate_task_b_ndcg(scores_b, ds.val_ad_answers(), candidate_ids)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterator

import numpy as np

from model.base import BaseRecoModel


# ──────────────────────────────────────────────
# 훈련
# ──────────────────────────────────────────────

def train(model: BaseRecoModel, stream: Iterator) -> None:
    """훈련 스트림 전체를 순서대로 모델에 흘려보낸다."""
    for event in stream:
        model.update_search(event.user_id, event.search_emb)
        for ad in event.ads:
            model.update_click(event.user_id, ad.ad_emb, clicked=bool(ad.is_click))


# ──────────────────────────────────────────────
# 스코어링 — 연속값 반환 (평가 지표 계산용)
# ──────────────────────────────────────────────

def score_task_a(
    model: BaseRecoModel,
    pairs: list,
) -> list[float]:
    """Task A: 각 (검색, 광고) 쌍의 클릭 가능성 연속 점수를 반환한다.

    Args:
        pairs : val_click_queries() 반환값 — list[(SearchEvent, AdRecord)]

    Returns:
        list[float]  — pairs와 같은 순서
    """
    return [
        model.score_click(ev.user_id, ev.search_emb, ad.ad_emb)
        for ev, ad in pairs
    ]


def score_task_b(
    model: BaseRecoModel,
    queries: list,
    candidate_embs: np.ndarray,
) -> dict[int, np.ndarray]:
    """Task B: 각 쿼리에 대해 모든 후보 광고의 연속 점수를 반환한다.

    Args:
        queries        : val_ad_queries() 반환값 — list[SearchEvent]
        candidate_embs : all_ad_embs() 반환값의 embedding 행렬

    Returns:
        dict[SearchID -> np.ndarray (N_candidates,)]
    """
    return {
        ev.search_id: model.score_ad_candidates(ev.user_id, ev.search_emb, candidate_embs)
        for ev in queries
    }


# ──────────────────────────────────────────────
# 예측 — 이진/순위값 반환 (제출용)
# ──────────────────────────────────────────────

def predict_task_a(
    model: BaseRecoModel,
    pairs: list,
    threshold: float,
) -> list[int]:
    """Task A: 점수가 (1 - threshold)를 넘으면 클릭(1), 아니면 비클릭(0)으로 예측한다.

    Returns:
        list[int] (0 또는 1)
    """
    return [int(s > 1 - threshold) for s in score_task_a(model, pairs)]


def predict_task_b(
    model: BaseRecoModel,
    queries: list,
    candidate_embs: np.ndarray,
    candidate_ids: list[int],
) -> dict[int, int]:
    """Task B: 각 쿼리에 대해 가장 점수가 높은 AdID를 반환한다.

    Returns:
        dict[SearchID -> 예측 AdID]
    """
    scores_dict = score_task_b(model, queries, candidate_embs)
    return {
        sid: candidate_ids[int(np.argmax(scores))]
        for sid, scores in scores_dict.items()
    }


# ──────────────────────────────────────────────
# 평가 — Task A (click prediction, F1)
# ──────────────────────────────────────────────

def evaluate_task_a(
    scores: list[float],
    answers_df,          # pandas DataFrame — "IsClick" 컬럼 포함
    threshold: float,
) -> dict:
    """Task A 평가 지표를 계산한다. 클릭(IsClick=1) 클래스 기준 지표를 주로 사용한다.

    Args:
        scores     : score_task_a() 반환값 (연속 점수)
        answers_df : val_click_answers() 반환값
        threshold  : 클릭 판정 경계값 (score > 1 - threshold 이면 클릭)

    Metrics
    -------
    Accuracy  : 전체 정답률
    Precision : 클릭으로 예측한 것 중 실제 클릭 비율  (IsClick=1 기준)
    Recall    : 실제 클릭 중 올바르게 탐지한 비율     (IsClick=1 기준)
    F1        : 클릭 클래스 F1                       (IsClick=1 기준)
    AUC       : ROC AUC (Wilcoxon-Mann-Whitney)
    per_class : {0: {...}, 1: {...}}  — confusion matrix 출력용
    """
    labels = answers_df["IsClick"].tolist()[: len(scores)]
    s      = np.array(scores[: len(labels)], dtype=float)
    preds  = [int(v > 1 - threshold) for v in s]

    pc = _multiclass_f1(preds, labels)["per_class"]
    c1 = pc.get(1, {"tp": 0, "fp": 0, "fn": 0,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})

    n       = len(labels)
    correct = sum(p == t for p, t in zip(preds, labels))

    return {
        "accuracy" : correct / n,
        "precision": c1["precision"],
        "recall"   : c1["recall"],
        "f1"       : c1["f1"],
        "auc"      : _binary_auc(s, np.array(labels)),
        "per_class": pc,
    }


# ──────────────────────────────────────────────
# 평가 — Task B (ad recommendation, NDCG@3)
# ──────────────────────────────────────────────

def evaluate_task_b_ndcg(
    scores_dict: dict[int, np.ndarray],
    answers: dict[int, int],
    candidate_ids: list[int],
) -> dict:
    """Task B NDCG@3 평가.

    score_task_b()와 동일한 스코어 딕셔너리를 받아,
    각 SearchID에서 정답 AdID의 순위 r을 구하고:
        NDCG@3 = 1 / log2(r+1)   (r <= 3)
               = 0                (r > 3)
    전체 쿼리에 대해 평균을 낸다.

    Args:
        scores_dict  : score_task_b() 반환값  {SearchID -> (N_candidates,) scores}
        answers      : val_ad_answers()        {SearchID -> 정답 AdID}
        candidate_ids: all_ad_embs() 반환값의 id 리스트

    Returns:
        {
          "ndcg@3"   : float,
          "n_queries": int,
          "rank_dist": {1:cnt, 2:cnt, 3:cnt, ">3":cnt},
        }
    """
    id_to_idx = {aid: i for i, aid in enumerate(candidate_ids)}
    common = set(scores_dict) & set(answers)

    ndcg_list: list[float] = []
    rank_dist: dict = {1: 0, 2: 0, 3: 0, ">3": 0}

    for sid in common:
        scores      = scores_dict[sid]
        correct_aid = answers[sid]
        if correct_aid not in id_to_idx:
            continue

        correct_score = scores[id_to_idx[correct_aid]]
        rank = int((scores > correct_score).sum()) + 1  # 1-indexed

        if rank <= 3:
            ndcg = 1.0 / math.log2(rank + 1)
            rank_dist[rank] += 1
        else:
            ndcg = 0.0
            rank_dist[">3"] += 1

        ndcg_list.append(ndcg)

    return {
        "ndcg@3"   : float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        "n_queries": len(ndcg_list),
        "rank_dist": rank_dist,
    }


# ──────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────

def _multiclass_f1(pred_list: list, true_list: list) -> dict:
    """Multi-class classification F1 지표를 계산한다."""
    n = len(pred_list)

    tp: dict = defaultdict(int)
    fp: dict = defaultdict(int)
    fn: dict = defaultdict(int)

    for p, t in zip(pred_list, true_list):
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    true_classes = sorted(set(true_list))

    per_class: dict = {}
    for c in true_classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {
            "tp"       : tp[c],
            "fp"       : fp[c],
            "fn"       : fn[c],
            "precision": prec,
            "recall"   : rec,
            "f1"       : f1_c,
            "support"  : tp[c] + fn[c],
        }

    f1_values   = [per_class[c]["f1"]     for c in true_classes]
    supports    = [per_class[c]["support"] for c in true_classes]
    total_sup   = sum(supports)
    macro_f1    = float(np.mean(f1_values)) if f1_values else 0.0
    weighted_f1 = float(np.dot(f1_values, supports) / total_sup) if total_sup > 0 else 0.0
    n_correct   = sum(tp[c] for c in true_classes)

    errors = [(t, p) for p, t in zip(pred_list, true_list) if p != t]

    return {
        "accuracy"    : n_correct / n if n > 0 else 0.0,
        "macro_f1"    : macro_f1,
        "weighted_f1" : weighted_f1,
        "micro_f1"    : n_correct / n if n > 0 else 0.0,
        "per_class"   : per_class,
        "confusion"   : {"correct": n_correct, "total": n, "errors": errors},
    }


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Wilcoxon-Mann-Whitney 통계량으로 binary AUC를 계산한다."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    pos_col = pos[:, None]
    neg_row = neg[None, :]
    n_pairs = len(pos) * len(neg)
    return float(
        (np.sum(pos_col > neg_row) + 0.5 * np.sum(pos_col == neg_row)) / n_pairs
    )
