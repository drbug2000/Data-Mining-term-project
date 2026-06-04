"""
gnn.py — 이종 그래프 기반 추천 모델 (Graph Neural Network).

────────────────────────────────────────────────────────────────
노드 초기화
────────────────────────────────────────────────────────────────
  user   : zero vector  (intrinsic feature 없음, 전파로 생성)
  search : query text embedding  (초기값, 전파로 보강)
  ad     : ad title text embedding  (초기값, 전파로 보강)

────────────────────────────────────────────────────────────────
1 GNN Layer = 모든 엣지를 이용한 동시 업데이트
────────────────────────────────────────────────────────────────
  Each layer updates ALL node types simultaneously:

  h_search[s] ← AGG( h_ad[a]   via search_to_ad_click (역방향)  +
                      h_user[u] via user_to_search (역방향)      )
  h_ad[a]     ← AGG( h_search[s] via search_to_ad_click          )
  h_user[u]   ← AGG( h_search[s] via user_to_search              )

  fallback: 이웃이 없는 노드는 초기 text embedding 유지

────────────────────────────────────────────────────────────────
n_layers에 따른 정보 흐름 (예시, mean agg)
────────────────────────────────────────────────────────────────
  n_layers=1 후:
    h_user   ← mean(query_embs)           [검색어 평균]
    h_search ← mean(ad_embs) + h_user_0   [클릭 ad + user(zero)]
    h_ad     ← mean(query_embs of clickers)

  n_layers=2 후:
    h_user   ← mean(h_search_1)           [click ad 정보 포함]
    h_search ← mean(h_ad_1) + h_user_1    [풍부해진 ad repr]
    h_ad     ← mean(h_search_1)           [user+click 정보 포함]

────────────────────────────────────────────────────────────────
스코어링
────────────────────────────────────────────────────────────────
  Task A (link prediction):
    score_link(search_id, ad_id) = sim(h_search, h_ad)

  Task B (ad recommendation):
    score_ad_candidates(user_id, query_emb, candidate_embs)
    = (1-γ)*sim(query, ad) + γ*sim(h_user, ad)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from shared.data.graph import HeteroGraph


# ─────────────────────────────────────────────────────────────────────────────
# GNNConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GNNConfig:
    # ── 전파 설정 ────────────────────────────────────────────────────────
    n_layers: int = 2
    """전파 레이어 수.
      2: A → B           (search←click_ad, user←search)
      3: A → B → C       (+ search←user : 새 search에 user 프로파일 반영)
      4: A → B → C → B   (한 번 더 user 집계)
    """

    agg_fn: str = "mean"
    """이웃 aggregation 함수. 'mean' | 'max' | 'sum'"""

    normalize: bool = True
    """각 레이어 후 repr을 L2 정규화할지 여부."""

    click_weight: float = 1.0
    """클릭 엣지 가중치. >1이면 클릭 신호를 일반 엣지보다 강하게 반영."""

    residual_alpha: float = 0.0
    """Residual connection 강도. 각 레이어 후:
      h_new = (1 - α) * h_propagated + α * h_initial
    α=0: residual 없음 (기존 동작), α=0.3: 초기 text emb 30% 보존."""

    user_click_init: bool = False
    """True이면 클릭 이력 있는 유저의 초기 repr을 zeros 대신
    클릭한 ad 임베딩의 평균으로 초기화한다."""

    # ── 스코어링 설정 ─────────────────────────────────────────────────────
    gamma:        float = 0.7
    """클릭 이력 있는 user의 interest mixing weight."""

    gamma_search: float = 0.5
    """검색 이력만 있는 user의 interest mixing weight."""

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"GNNConfig({items})"


# ─────────────────────────────────────────────────────────────────────────────
# GNNModel
# ─────────────────────────────────────────────────────────────────────────────

class GNNModel:
    """Multi-hop Light Propagation GNN.

    사용 흐름:
        g     = build_graph(ds, transductive=True)
        model = GNNModel(GNNConfig(n_layers=3))
        model.fit(g)

        # Task A
        score = model.score_link(search_id, ad_id)

        # Task B
        scores = score_task_b(model, queries, candidate_embs)
    """

    def __init__(self, config: GNNConfig | None = None) -> None:
        self.config = config or GNNConfig()

        self._search_repr:     np.ndarray | None = None  # (N_search, dim) — 전파 후
        self._user_repr:       np.ndarray | None = None  # (N_users,  dim) — 전파 후
        self._search_feat_raw: np.ndarray | None = None  # (N_search, dim) — 전파 전 원본
        self._ad_feat_raw:     np.ndarray | None = None  # (N_ads,    dim) — 전파 전 원본

        self._clicked_users:  set[int] = set()
        self._searched_users: set[int] = set()

        self._user_id_to_idx:   dict[int, int] = {}
        self._search_id_to_idx: dict[int, int] = {}
        self._ad_id_to_idx:     dict[int, int] = {}
        self._ad_feat:          np.ndarray | None = None

        self._fitted = False

    # ──────────────────────────────────────────────────────────────────────
    # 학습 단계
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, graph: HeteroGraph) -> "GNNModel":
        """그래프 전파로 모든 노드 repr을 사전 계산한다.

        1 GNN layer = 모든 엣지를 통한 동시 업데이트:
          h_search ← AGG( h_ad via click(역), h_user via user_to_search(역) )
          h_ad     ← AGG( h_search via click )
          h_user   ← AGG( h_search via user_to_search )

        n_layers만큼 반복한다.
        """
        cfg = self.config
        agg = _get_agg_fn(cfg.agg_fn)

        ad_feat_init     = graph.node_feat("ad")      # (N_ads,    dim) — 초기값 고정
        search_feat_init = graph.node_feat("search")  # (N_search, dim) — query text emb
        dim      = ad_feat_init.shape[1]
        N_search = graph.num_nodes("search")
        N_users  = graph.num_nodes("user")
        N_ads    = graph.num_nodes("ad")

        # ── ID 매핑 복사 ─────────────────────────────────────────────────
        self._user_id_to_idx   = graph._id_to_idx["user"]
        self._search_id_to_idx = graph._id_to_idx["search"]
        self._ad_id_to_idx     = graph._id_to_idx["ad"]

        # ── 인접 리스트 구축 ─────────────────────────────────────────────
        us_edges    = graph.edges("user_to_search")
        click_edges = graph.edges("search_to_ad_click")

        # user_to_search: (idx, weight) 튜플 저장
        us_w = graph.edge_weights("user_to_search")
        u2s: dict[int, list[tuple[int, float]]] = defaultdict(list)
        s2u: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for e in range(us_edges.shape[1]):
            ui_ = int(us_edges[0, e]); si_ = int(us_edges[1, e])
            w   = float(us_w[e]) if us_w is not None else 1.0
            u2s[ui_].append((si_, w))
            s2u[si_].append((ui_, w))

        # click 엣지: 균등 가중치 (click_weight로 전역 조절)
        click_w = graph.edge_weights("search_to_ad_click")  # None이면 1.0 균등
        s2a: dict[int, list[tuple[int, float]]] = defaultdict(list)  # (ad_idx, weight)
        a2s: dict[int, list[tuple[int, float]]] = defaultdict(list)  # (search_idx, weight)
        for e in range(click_edges.shape[1]):
            si_ = int(click_edges[0, e])
            ai_ = int(click_edges[1, e])
            w   = float(click_w[e]) if click_w is not None else 1.0
            s2a[si_].append((ai_, w))
            a2s[ai_].append((si_, w))

        # sim 엣지: (idx, weight) 튜플
        s2s: dict[int, list[tuple[int, float]]] = defaultdict(list)
        if "search_to_search_sim" in graph.edge_names():
            sim_s   = graph.edges("search_to_search_sim")
            sim_s_w = graph.edge_weights("search_to_search_sim")
            for e in range(sim_s.shape[1]):
                w = float(sim_s_w[e]) if sim_s_w is not None else 1.0
                s2s[int(sim_s[0, e])].append((int(sim_s[1, e]), w))
            print(f"[GNN] search_to_search_sim 로드: {sim_s.shape[1]:,} 엣지")

        a2a: dict[int, list[tuple[int, float]]] = defaultdict(list)
        if "ad_to_ad_sim" in graph.edge_names():
            sim_a   = graph.edges("ad_to_ad_sim")
            sim_a_w = graph.edge_weights("ad_to_ad_sim")
            for e in range(sim_a.shape[1]):
                w = float(sim_a_w[e]) if sim_a_w is not None else 1.0
                a2a[int(sim_a[0, e])].append((int(sim_a[1, e]), w))
            print(f"[GNN] ad_to_ad_sim 로드: {sim_a.shape[1]:,} 엣지")

        # ── 유저 분류 (adaptive gamma용) ─────────────────────────────────
        for ui in np.unique(us_edges[0]):
            self._searched_users.add(graph.to_id("user", int(ui)))
        for e in range(click_edges.shape[1]):
            si = int(click_edges[0, e])
            for ui, _ in s2u.get(si, []):   # 튜플 언팩
                self._clicked_users.add(graph.to_id("user", ui))

        # ── 노드 repr 초기화 ─────────────────────────────────────────────
        # user: user_click_init=True면 클릭 ad 임베딩 평균으로 초기화
        user_repr = np.zeros((N_users, dim), dtype=np.float32)
        if cfg.user_click_init:
            # 유저별 클릭 ad repr 집계
            user_ad_vecs: dict[int, list[np.ndarray]] = defaultdict(list)
            for e in range(click_edges.shape[1]):
                si_ = int(click_edges[0, e])
                ai_ = int(click_edges[1, e])
                for ui_, _ in s2u.get(si_, []):
                    user_ad_vecs[ui_].append(ad_feat_init[ai_])
            n_init = 0
            for ui_, vecs in user_ad_vecs.items():
                user_repr[ui_] = _l2_normalize(
                    np.stack(vecs).mean(axis=0)[None])[0]
                n_init += 1
            print(f"[GNN] user_click_init: {n_init:,}명 유저 repr 초기화")

        search_repr = search_feat_init.copy()
        ad_repr     = ad_feat_init.copy()

        # 전파 전 원본 저장 (MLP 피처에서 raw text 유사도 계산에 사용)
        self._search_feat_raw = search_feat_init.copy()
        self._ad_feat_raw     = ad_feat_init.copy()

        # ── n_layers만큼 반복 ────────────────────────────────────────────
        for layer_idx in range(1, cfg.n_layers + 1):
            print(f"[GNN] Layer {layer_idx} / {cfg.n_layers}")

            # ── search: click(ad) + user + sim(search) 집계 ─────────────
            new_search = search_feat_init.copy()   # fallback: query emb
            cw = cfg.click_weight
            n_s = n_s_sim = 0
            for si in range(N_search):
                vecs = []
                # click 이웃: effective_weight = click_weight × position
                for ai, pos_w in s2a.get(si, []):
                    eff_w = cw * pos_w
                    vecs.append(ad_repr[ai] * eff_w)
                for ui, u_w in s2u.get(si, []):
                    v = user_repr[ui]
                    if np.linalg.norm(v) > 1e-8:
                        vecs.append(v * u_w)
                for si2, sim_w in s2s.get(si, []):
                    vecs.append(search_repr[si2] * sim_w)
                    n_s_sim += 1
                if vecs:
                    new_search[si] = agg(np.stack(vecs))
                    n_s += 1
            if cfg.normalize:
                new_search = _l2_normalize(new_search)
            sim_tag = f" (sim contrib: {n_s_sim:,})" if n_s_sim else ""
            print(f"  search: {n_s:,} / {N_search:,} nodes updated{sim_tag}")

            # ── ad: click(search) + sim(ad) 집계 ─────────────────────────
            new_ad = ad_feat_init.copy()           # fallback: ad text emb
            n_a = n_a_sim = 0
            for ai in range(N_ads):
                vecs = []
                for si2, pos_w in a2s.get(ai, []):
                    eff_w = cw * pos_w
                    vecs.append(search_repr[si2] * eff_w)
                for ai2, sim_w in a2a.get(ai, []):
                    vecs.append(ad_repr[ai2] * sim_w)
                    n_a_sim += 1
                if vecs:
                    new_ad[ai] = agg(np.stack(vecs))
                    n_a += 1
            if cfg.normalize:
                new_ad = _l2_normalize(new_ad)
            sim_tag = f" (sim contrib: {n_a_sim:,})" if n_a_sim else ""
            print(f"  ad    : {n_a:,} / {N_ads:,} nodes updated{sim_tag}")

            # ── user: search(user_to_search 정방향) 집계 ─────────────────
            new_user = np.zeros((N_users, dim), dtype=np.float32)
            n_u = 0
            for ui in range(N_users):
                nbrs = u2s.get(ui, [])
                if nbrs:
                    vecs = np.stack([search_repr[si] * w for si, w in nbrs])
                    new_user[ui] = agg(vecs)
                    n_u += 1
            if cfg.normalize:
                norms = np.linalg.norm(new_user, axis=1, keepdims=True)
                mask  = norms[:, 0] > 1e-8
                new_user[mask] = new_user[mask] / norms[mask]
            print(f"  user  : {n_u:,} / {N_users:,} nodes updated")

            # ── Residual connection (α > 0 일 때) ────────────────────────
            α = cfg.residual_alpha
            if α > 0:
                # h_final = (1-α) * h_propagated + α * h_initial
                new_search = (1 - α) * new_search + α * search_feat_init
                new_ad     = (1 - α) * new_ad     + α * ad_feat_init
                if cfg.normalize:
                    new_search = _l2_normalize(new_search)
                    new_ad     = _l2_normalize(new_ad)

            # 동시 업데이트 (이전 repr을 읽어서 새 repr로 교체)
            search_repr = new_search
            ad_repr     = new_ad
            user_repr   = new_user

        self._search_repr = search_repr
        self._user_repr   = user_repr
        self._ad_feat     = ad_repr    # propagation 후 ad repr (scoring에 사용)
        self._fitted = True
        print(f"[GNN] fit 완료  n_layers={cfg.n_layers}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    # 추론 단계
    # ──────────────────────────────────────────────────────────────────────

    def score_link(self, search_id: int, ad_id: int) -> float:
        """Task A (link prediction): sim(h_search, h_ad).

        n_layers>=3이면 h_search에 user 클릭 프로파일이 담겨 있음.
        """
        assert self._fitted
        si = self._search_id_to_idx.get(search_id)
        ai = self._ad_id_to_idx.get(ad_id)
        if si is None or ai is None:
            return 0.0
        h_s = self._search_repr[si]
        h_a = self._ad_feat[ai]
        return float(
            (h_s / (np.linalg.norm(h_s) + 1e-8)) @
            (h_a / (np.linalg.norm(h_a) + 1e-8))
        )

    def score_ad_candidates(
        self,
        user_id:        int,
        query_emb:      np.ndarray,
        candidate_embs: np.ndarray,
    ) -> np.ndarray:
        """Task B: (1-γ)*sim(query,ad) + γ*sim(h_user,ad)."""
        assert self._fitted
        q = _l2_normalize(query_emb[None])[0]
        C = _l2_normalize(candidate_embs)
        query_sim = C @ q

        gamma_eff = self._effective_gamma(user_id)
        if gamma_eff == 0.0:
            return query_sim

        ui = self._user_id_to_idx.get(user_id)
        if ui is None:
            return query_sim
        u_repr = self._user_repr[ui]
        if np.linalg.norm(u_repr) < 1e-8:
            return query_sim

        u_norm = u_repr / (np.linalg.norm(u_repr) + 1e-8)
        return (1 - gamma_eff) * query_sim + gamma_eff * (C @ u_norm)

    def score_click(self, user_id: int, query_emb: np.ndarray,
                    ad_emb: np.ndarray) -> float:
        """Task A (user repr 기반): score_ad_candidates의 단일 광고 버전."""
        assert self._fitted
        return float(self.score_ad_candidates(user_id, query_emb, ad_emb[None])[0])

    # ──────────────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────────────────

    def _effective_gamma(self, user_id: int) -> float:
        if user_id in self._clicked_users:
            return self.config.gamma
        if user_id in self._searched_users:
            return self.config.gamma_search
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)


def _get_agg_fn(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name == "mean":
        return lambda x: x.mean(axis=0)
    elif name == "max":
        return lambda x: x.max(axis=0)
    elif name == "sum":
        return lambda x: x.sum(axis=0)
    else:
        raise ValueError(f"Unknown agg_fn: '{name}'. Choose 'mean', 'max', 'sum'.")
