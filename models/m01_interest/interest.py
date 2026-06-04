"""
MultiInterestModel — BaseRecoModel 구현체.

알고리즘 요약
─────────────────────────────────────────────────────────
두 종류의 interest vector 를 유저마다 관리한다.

  _interests       (k, dim) : 검색+클릭 공통 업데이트 → Task B (ad recommendation)
  _click_interests (k, dim) : 클릭/비클릭 전용 업데이트 → Task A (click prediction)

이벤트 발생 시 Soft Assignment 방식으로 업데이트:
  1. 각 interest vector와 새 embedding 사이의 코사인 거리 계산
  2. exp(-dist/τ) 로 가중치 산출 → 합이 1이 되도록 정규화
  3. v_i += sign * alpha * w_i * emb_normalized
       검색  : _interests 만 업데이트 (alpha_search)
       클릭  : _interests + _click_interests (alpha_click)
       비클릭: _click_interests 만 반대 방향 (alpha_neg, 0이면 무시)

예측
  Task A score_click  : (1-γ)·sim(query,ad) + γ·max_i sim(click_v_i, ad)
  Task B score_ad_candidates : (1-γ)·sim(query,ad) + γ·max_i sim(v_i, ad)
  Adaptive gamma : 클릭 이력이 없는 유저 → gamma=0 (query-only)
─────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

from shared.base import BaseRecoModel
from models.m01_interest.config import ModelConfig


class MultiInterestModel(BaseRecoModel):

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # user_id → interest matrix, shape (k, dim)  [검색+클릭, Task B용]
        self._interests: dict[int, np.ndarray] = {}
        # user_id → click-only interest matrix, shape (k, dim)  [Task A용]
        self._click_interests: dict[int, np.ndarray] = {}
        # 검색 이력이 있는 유저 집합 (gamma_search 활성화 기준)
        self._searched_users: set[int] = set()
        # 클릭 이력이 있는 유저 집합 (gamma 활성화 기준)
        self._clicked_users: set[int] = set()

    # ------------------------------------------------------------------
    # 훈련 (BaseRecoModel 구현)
    # ------------------------------------------------------------------

    def update_search(self, user_id: int, search_emb: np.ndarray) -> None:
        """검색 발생 → _interests(Task B용)만 업데이트. click_interests는 건드리지 않는다."""
        self._searched_users.add(user_id)
        self._update_store(self._interests, user_id, search_emb,
                           alpha=self.config.alpha_search, sign=+1)

    def update_click(self, user_id: int, ad_emb: np.ndarray, clicked: bool) -> None:
        """광고 노출 → 클릭이면 두 store 모두 당기고, 비클릭이면 click_interests만 밀어낸다."""
        if clicked:
            self._clicked_users.add(user_id)
            self._update_store(self._interests,       user_id, ad_emb,
                               alpha=self.config.alpha_click, sign=+1)
            self._update_store(self._click_interests, user_id, ad_emb,
                               alpha=self.config.alpha_click, sign=+1)
        elif self.config.alpha_neg > 0:
            self._update_store(self._click_interests, user_id, ad_emb,
                               alpha=self.config.alpha_neg, sign=-1)

    # ------------------------------------------------------------------
    # 예측 (BaseRecoModel 구현)
    # ------------------------------------------------------------------

    def _effective_gamma(self, user_id: int) -> float:
        """3단계 adaptive gamma를 반환한다.

        클릭 이력 있음  → cfg.gamma        (높은 가중치)
        검색 이력만 있음 → cfg.gamma_search  (낮은 가중치)
        cold-start      → 0.0              (query-only 폴백)
        """
        if user_id in self._clicked_users:
            return self.config.gamma
        if user_id in self._searched_users:
            return self.config.gamma_search
        return 0.0

    def score_ad_candidates(
        self,
        user_id: int,
        query_emb: np.ndarray,
        candidate_embs: np.ndarray,
    ) -> np.ndarray:
        """Task B: 모든 후보에 대한 연속 점수를 반환한다. shape (N_candidates,)

        Adaptive gamma (3단계):
          click 이력 → gamma / search 이력만 → gamma_search / cold → 0
        """
        q = _l2_normalize(query_emb)      # (dim,)
        C = _l2_normalize(candidate_embs) # (N, dim)
        query_ad_sim = C @ q              # (N,)

        effective_gamma = self._effective_gamma(user_id)
        if effective_gamma == 0.0:
            return query_ad_sim

        V = _l2_normalize(self._get_or_init_store(self._interests, user_id))  # (k, dim)
        interest_ad_sim = (V @ C.T).max(axis=0)                               # (N,)
        return (1 - effective_gamma) * query_ad_sim + effective_gamma * interest_ad_sim

    def score_click(
        self,
        user_id: int,
        query_emb: np.ndarray,
        ad_emb: np.ndarray,
    ) -> float:
        """Task A: 클릭 가능성 연속 점수.

        click-only interest (_click_interests) 를 사용하여 검색 오염을 피한다.
            (1-gamma)*sim(query, ad) + gamma*max_i sim(click_v_i, ad)
        Adaptive gamma: 클릭 이력이 없는 유저는 query-ad 유사도만 사용.
        """
        q = _l2_normalize(query_emb)  # (dim,)
        a = _l2_normalize(ad_emb)     # (dim,)
        query_ad_sim = float(q @ a)

        effective_gamma = self._effective_gamma(user_id)
        if effective_gamma == 0.0:
            return query_ad_sim

        V = _l2_normalize(self._get_or_init_store(self._click_interests, user_id))  # (k, dim)
        interest_ad_sim = float((V @ a).max())
        return (1 - effective_gamma) * query_ad_sim + effective_gamma * interest_ad_sim

    def predict_ad(
        self,
        user_id: int,
        query_emb: np.ndarray,
        candidate_embs: np.ndarray,
        candidate_ids: list[int],
    ) -> int:
        """Task B: score_ad_candidates 의 argmax 에 해당하는 AdID를 반환한다."""
        scores = self.score_ad_candidates(user_id, query_emb, candidate_embs)
        return candidate_ids[int(np.argmax(scores))]

    def predict_click(
        self,
        user_id: int,
        query_emb: np.ndarray,
        ad_emb: np.ndarray,
    ) -> int:
        """Task A: score_click > (1 - threshold) 이면 1, 아니면 0을 반환한다."""
        return int(self.score_click(user_id, query_emb, ad_emb) > 1 - self.config.threshold)

    # ------------------------------------------------------------------
    # 디버깅 (BaseRecoModel 선택 구현)
    # ------------------------------------------------------------------

    def get_interests(self, user_id: int) -> np.ndarray:
        """유저의 (검색+클릭) interest 행렬 복사본을 반환한다. shape: (k, dim)"""
        return self._get_or_init_store(self._interests, user_id).copy()

    def get_click_interests(self, user_id: int) -> np.ndarray:
        """유저의 click-only interest 행렬 복사본을 반환한다. shape: (k, dim)"""
        return self._get_or_init_store(self._click_interests, user_id).copy()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _get_or_init_store(
        self, store: dict[int, np.ndarray], user_id: int
    ) -> np.ndarray:
        """지정 store에서 유저의 interest 행렬을 반환. 없으면 랜덤 초기화."""
        if user_id not in store:
            k, dim = self.config.k, self.config.dim
            vecs = np.random.randn(k, dim).astype(np.float32)
            store[user_id] = _l2_normalize(vecs)
        return store[user_id]

    def _soft_weights(self, interests: np.ndarray, emb_normalized: np.ndarray) -> np.ndarray:
        """Soft assignment 가중치를 계산한다.

        Args:
            interests       : (k, dim)  raw interest 행렬
            emb_normalized  : (dim,)    이미 L2 정규화된 embedding

        Returns:
            weights: (k,)  합이 1인 soft assignment 가중치
        """
        V        = _l2_normalize(interests)           # (k, dim)
        cos_dist = 1.0 - (V @ emb_normalized)         # (k,)  코사인 거리 ∈ [0, 2]
        raw      = np.exp(-cos_dist / self.config.temperature)
        return raw / raw.sum()                        # 정규화 → 합 = 1

    def _update_store(
        self, store: dict[int, np.ndarray], user_id: int,
        emb: np.ndarray, alpha: float, sign: int
    ) -> None:
        """지정 store의 interest를 Soft assignment 기반으로 업데이트한다.

        v_i  +=  sign * alpha * w_i * emb_normalized
        """
        if alpha == 0.0:
            return

        interests      = self._get_or_init_store(store, user_id)  # (k, dim) — 원본 참조
        emb_normalized = _l2_normalize(emb)                        # (dim,)

        weights = self._soft_weights(interests, emb_normalized)    # (k,)

        interests += sign * alpha * weights[:, None] * emb_normalized


# ------------------------------------------------------------------
# 모듈 공통 유틸
# ------------------------------------------------------------------

def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """마지막 축(axis=-1) 기준으로 L2 정규화한다.

    x shape: (dim,) 또는 (N, dim)  →  같은 shape 반환
    """
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)
