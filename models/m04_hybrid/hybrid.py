"""
HybridModel — GNN (m02) + MultiInterest (m01) 결합.

────────────────────────────────────────────────────────────────────────
각 모델이 자연스럽게 강한 Task 에 집중한다.
────────────────────────────────────────────────────────────────────────

▪ Task B (ad recommendation) — m01 다중 interest 방식
  GNN 은 Option A(보강 입력)로만 기여한다.

    score = (1-γ) · sim(q, a)
          + γ     · max_j sim(v_j, ã)
                              ↑
                    ã = GNN h_ad (ad_id 있을 때) / raw text emb (폴백)
                    v_j 자체도 GNN h_search/h_ad 로 업데이트됨

  → m02 의 단일 h_user 를 섞지 않음: 다중 interest 신호를 희석시키기 때문.

▪ Task A (click prediction) — GNN link prediction 혼합
  GNN 의 score_link = sim(h_search, h_ad) 는 search-ad 의
  구조적 친밀도를 직접 포착한다 (m01 에는 없는 신호).

    score = (1 - link_alpha) · m01_click_score
          + link_alpha        · gnn.score_link(search_id, ad_id)

    m01_click_score = (1-γ) · sim(q, a) + γ · max_j sim(cv_j, ã)
      cv_j : click-only interest (_click_interests)
      ã    : GNN h_ad (Option A 보강)

  search_id / ad_id 미제공 또는 GNN 미학습 → m01_click_score 만 사용.

────────────────────────────────────────────────────────────────────────
사용 흐름
────────────────────────────────────────────────────────────────────────
    cfg   = HybridConfig(link_alpha=0.4, n_layers=2)
    model = HybridModel(cfg)

    model.fit(graph)                                  # Stage 1 (선택)

    model.update_search(user_id, q_emb, search_id=sid)
    model.update_click(user_id, a_emb, clicked=True, ad_id=aid)

    # Task B
    scores = model.score_ad_candidates(uid, q_emb, cand_embs, cand_ids)

    # Task A  ← search_id 전달이 핵심
    sc = model.score_click(uid, q_emb, a_emb, search_id=sid, ad_id=aid)
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

from shared.base import BaseRecoModel
from shared.data.graph import HeteroGraph
from models.m02_gnn.gnn import GNNModel
from models.m04_hybrid.config import HybridConfig


class HybridModel(BaseRecoModel):

    def __init__(self, config: HybridConfig):
        super().__init__(config)
        # _gnn_b: Option A 입력 보강 (Task B _interests 업데이트용)
        # _gnn_a: score_link Task A 신호용 (별도 fit 가능)
        self._gnn_b = GNNModel(config.gnn_config())
        self._gnn_a: GNNModel | None = None   # fit_task_a() 호출 전까지 None

        self._interests:       dict[int, np.ndarray] = {}  # Task B용 (검색+클릭)
        self._click_interests: dict[int, np.ndarray] = {}  # Task A용 (클릭 전용)
        self._searched_users:  set[int] = set()
        self._clicked_users:   set[int] = set()

    # ------------------------------------------------------------------
    # Stage 1 — 오프라인 GNN 전파
    # ------------------------------------------------------------------

    def fit(self, graph: HeteroGraph) -> "HybridModel":
        """Task B용 그래프 전파 (Option A 입력 보강).

        fit 없이 사용하면 pure MultiInterest (m01) 로 동작한다.
        """
        self._gnn_b.fit(graph)
        self._searched_users |= self._gnn_b._searched_users
        self._clicked_users  |= self._gnn_b._clicked_users
        return self

    def fit_task_a(self, graph: HeteroGraph) -> "HybridModel":
        """Task A score_link 전용 GNN 전파 (fit 과 별도 그래프 지정 가능).

        호출하지 않으면 score_link 는 사용되지 않는다.
        sim 엣지가 있는 그래프를 넣으면 score_link 품질이 올라간다.
        """
        self._gnn_a = GNNModel(self.config.gnn_config())
        self._gnn_a.fit(graph)
        return self

    # ------------------------------------------------------------------
    # Stage 2 — 온라인 업데이트 (BaseRecoModel 구현)
    # ------------------------------------------------------------------

    def update_search(
        self,
        user_id: int,
        search_emb: np.ndarray,
        search_id: int | None = None,
    ) -> None:
        """검색 발생 → _interests(Task B) 업데이트.

        search_id 제공 시 GNN h_search 를 Soft-Assignment 입력으로 사용 (Option A).
        """
        self._searched_users.add(user_id)
        emb = self._enrich_search(search_emb, search_id)
        self._update_store(self._interests, user_id, emb,
                           alpha=self.config.alpha_search, sign=+1)

    def update_click(
        self,
        user_id: int,
        ad_emb: np.ndarray,
        clicked: bool,
        ad_id: int | None = None,
    ) -> None:
        """광고 노출 → 클릭이면 두 스토어 모두, 비클릭이면 click_interests 만 업데이트.

        _interests (Task B): GNN h_ad 로 업데이트 (ad_id 있을 때) — Option A.
        _click_interests (Task A): 항상 raw ad_emb 로 업데이트.
          → score_click 스코어링도 raw emb 를 사용하므로 공간 일치.
        """
        emb_enriched = self._enrich_ad(ad_emb, ad_id)   # Task B용 (enriched or raw)
        if clicked:
            self._clicked_users.add(user_id)
            self._update_store(self._interests,       user_id, emb_enriched,
                               alpha=self.config.alpha_click, sign=+1)
            self._update_store(self._click_interests, user_id, ad_emb,   # raw 고정
                               alpha=self.config.alpha_click, sign=+1)
        elif self.config.alpha_neg > 0:
            self._update_store(self._click_interests, user_id, ad_emb,   # raw 고정
                               alpha=self.config.alpha_neg, sign=-1)

    # ------------------------------------------------------------------
    # 예측 (BaseRecoModel 구현)
    # ------------------------------------------------------------------

    def predict_ad(
        self,
        user_id: int,
        query_emb: np.ndarray,
        candidate_embs: np.ndarray,
        candidate_ids: list[int],
    ) -> int:
        scores = self.score_ad_candidates(user_id, query_emb, candidate_embs, candidate_ids)
        return candidate_ids[int(np.argmax(scores))]

    def predict_click(
        self,
        user_id: int,
        query_emb: np.ndarray,
        ad_emb: np.ndarray,
        search_id: int | None = None,
        ad_id:    int | None = None,
    ) -> int:
        score = self.score_click(user_id, query_emb, ad_emb, search_id, ad_id)
        return int(score > 1 - self.config.threshold)

    # ------------------------------------------------------------------
    # Task B 스코어링 — m01 다중 interest (GNN 은 입력 보강만)
    # ------------------------------------------------------------------

    def score_ad_candidates(
        self,
        user_id: int,
        query_emb: np.ndarray,
        candidate_embs: np.ndarray,
        candidate_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Task B: (1-γ)·sim(q,a) + γ·max_j sim(v_j, ã).

        m01 다중 interest 방식과 동일.
        candidate_ids 제공 시 ã = GNN h_ad, 없으면 raw emb.
        """
        q         = _l2_normalize(query_emb)
        C         = _l2_normalize(candidate_embs)
        query_sim = C @ q                            # (N,)

        γ = self._effective_gamma(user_id)
        if γ == 0.0:
            return query_sim

        C_enr = _l2_normalize(self._enrich_candidates(candidate_embs, candidate_ids))
        V     = _l2_normalize(self._get_or_init_store(self._interests, user_id))
        return (1 - γ) * query_sim + γ * (V @ C_enr.T).max(axis=0)

    # ------------------------------------------------------------------
    # Task A 스코어링 — m01 click interest + GNN link prediction 혼합
    # ------------------------------------------------------------------

    def score_click(
        self,
        user_id:   int,
        query_emb: np.ndarray,
        ad_emb:    np.ndarray,
        search_id: int | None = None,
        ad_id:     int | None = None,
    ) -> float:
        """Task A: (1-α)·m01_click_score + α·gnn.score_link(search_id, ad_id).

        m01_click_score = (1-γ)·sim(q,a) + γ·max_j sim(cv_j, ã)
          cv_j : _click_interests (클릭 전용, 검색 오염 방지)
          ã    : GNN h_ad (ad_id 있을 때) / raw emb (폴백)

        GNN link score = sim(h_search, h_ad)  (search_id, ad_id 모두 필요)
        search_id / ad_id 미제공 또는 GNN 미학습 → m01_click_score 단독 사용.
        """
        q         = _l2_normalize(query_emb)
        a         = _l2_normalize(ad_emb)
        query_sim = float(q @ a)

        γ = self._effective_gamma(user_id)

        # m01 click interest score — raw ad_emb 고정 (click_interests 와 공간 일치)
        if γ > 0.0:
            V = _l2_normalize(self._get_or_init_store(self._click_interests, user_id))
            m01_score = (1 - γ) * query_sim + γ * float((V @ a).max())
        else:
            m01_score = query_sim

        # GNN link prediction score — _gnn_a (Task A 전용 GNN, +sim 그래프) 사용
        α = self.config.link_alpha
        if α > 0.0 and search_id is not None and ad_id is not None and self._gnn_a is not None:
            gnn_link = self._gnn_a.score_link(search_id, ad_id)
            if gnn_link != 0.0:   # 0.0 = 미등록 pair 폴백 → 무시
                return (1 - α) * m01_score + α * gnn_link

        return m01_score

    # ------------------------------------------------------------------
    # 디버깅
    # ------------------------------------------------------------------

    def get_interests(self, user_id: int) -> np.ndarray:
        return self._get_or_init_store(self._interests, user_id).copy()

    def get_click_interests(self, user_id: int) -> np.ndarray:
        return self._get_or_init_store(self._click_interests, user_id).copy()

    # ------------------------------------------------------------------
    # 내부 헬퍼 — Adaptive gamma
    # ------------------------------------------------------------------

    def _effective_gamma(self, user_id: int) -> float:
        if user_id in self._clicked_users:
            return self.config.gamma
        if user_id in self._searched_users:
            return self.config.gamma_search
        return 0.0

    # ------------------------------------------------------------------
    # 내부 헬퍼 — GNN 임베딩 조회
    # ------------------------------------------------------------------

    def _enrich_search(self, raw_emb: np.ndarray, search_id: int | None) -> np.ndarray:
        if search_id is not None and self._gnn_b._fitted:
            idx = self._gnn_b._search_id_to_idx.get(search_id)
            if idx is not None:
                return self._gnn_b._search_repr[idx]
        return raw_emb

    def _enrich_ad(self, raw_emb: np.ndarray, ad_id: int | None) -> np.ndarray:
        if ad_id is not None and self._gnn_b._fitted:
            idx = self._gnn_b._ad_id_to_idx.get(ad_id)
            if idx is not None:
                return self._gnn_b._ad_feat[idx]
        return raw_emb

    def _enrich_candidates(
        self,
        raw_embs: np.ndarray,
        candidate_ids: list[int] | None,
    ) -> np.ndarray:
        if candidate_ids is None or not self._gnn_b._fitted:
            return raw_embs
        result = raw_embs.copy()
        for i, cid in enumerate(candidate_ids):
            idx = self._gnn_b._ad_id_to_idx.get(cid)
            if idx is not None:
                result[i] = self._gnn_b._ad_feat[idx]
        return result

    # ------------------------------------------------------------------
    # 내부 헬퍼 — MultiInterest (m01 과 동일)
    # ------------------------------------------------------------------

    def _get_or_init_store(
        self, store: dict[int, np.ndarray], user_id: int
    ) -> np.ndarray:
        if user_id not in store:
            k, dim = self.config.k, self.config.dim
            vecs = np.random.randn(k, dim).astype(np.float32)
            store[user_id] = _l2_normalize(vecs)
        return store[user_id]

    def _soft_weights(
        self, interests: np.ndarray, emb_normalized: np.ndarray
    ) -> np.ndarray:
        V        = _l2_normalize(interests)
        cos_dist = 1.0 - (V @ emb_normalized)
        raw      = np.exp(-cos_dist / self.config.temperature)
        return raw / raw.sum()

    def _update_store(
        self,
        store: dict[int, np.ndarray],
        user_id: int,
        emb: np.ndarray,
        alpha: float,
        sign: int,
    ) -> None:
        if alpha == 0.0:
            return
        interests      = self._get_or_init_store(store, user_id)
        emb_normalized = _l2_normalize(emb)
        weights        = self._soft_weights(interests, emb_normalized)
        interests     += sign * alpha * weights[:, None] * emb_normalized


# ------------------------------------------------------------------
# 모듈 공통 유틸
# ------------------------------------------------------------------

def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)
