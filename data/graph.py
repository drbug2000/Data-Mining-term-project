"""
graph.py — 이종 그래프(Heterogeneous Graph) 구축.

노드 타입 (3종)
─────────────────────────────────────────────────────
  user   : 유저           — 피처: 검색어 embedding 평균
  search : 검색 이벤트    — 피처: 검색어 text embedding
  ad     : 광고           — 피처: 광고 제목 text embedding

기본 엣지 타입 (빌드 시 자동 생성)
─────────────────────────────────────────────────────
  user_to_search        : user → search  (유저가 검색을 발생)
  search_to_ad_click    : search → ad    (검색에서 광고를 클릭)
  search_to_ad_show     : search → ad    (검색에서 광고가 노출, 비클릭)

확장 예시 (add_edges로 추가 가능)
─────────────────────────────────────────────────────
  ad_to_ad_sim_topk     : ad → ad        (text embedding 유사도 top-k)
  user_to_user_sim      : user → user    (검색 패턴 유사 유저)
  search_to_search_sim  : search → search (쿼리 embedding 유사 검색)

사용 예시
─────────────────────────────────────────────────────
    ds = RecoDataset("../datasets").load()
    g  = build_graph(ds)

    # 기본 조회
    g.node_feat("user")              # (N_users, 384)
    g.node_feat("ad")                # (N_ads,   384)
    g.edges("user_to_search")        # (2, E)  COO [src; dst]
    g.edge_weights("search_to_ad_click")  # (E,)

    # 새 엣지 추가
    g.add_edges(
        name="ad_to_ad_sim_topk",
        src_type="ad", dst_type="ad",
        coo=my_sim_edges,            # (2, E) int32
        weights=my_sim_scores,       # (E,) float32  optional
        description="text sim top-10",
    )

    # 노드 ID ↔ 인덱스
    g.to_idx("user", user_id)
    g.to_id("ad", ad_idx)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# EdgeSpec — 엣지 타입 메타데이터
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EdgeSpec:
    """하나의 엣지 타입을 기술하는 메타데이터."""
    name:        str    # 고유 키
    src_type:    str    # "user" | "search" | "ad"
    dst_type:    str
    description: str = ""
    is_directed: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# HeteroGraph — 이종 그래프 컨테이너
# ─────────────────────────────────────────────────────────────────────────────

class HeteroGraph:
    """3-type 이종 그래프 (user / search / ad).

    엣지는 이름으로 등록·조회하며, add_edges()로 언제든 추가할 수 있다.
    """

    # 허용 노드 타입
    NODE_TYPES = ("user", "search", "ad")

    def __init__(self) -> None:
        # 노드 피처: node_type → (N, dim) float32
        self._node_feats:   dict[str, np.ndarray] = {}

        # 노드 ID 매핑: node_type → {original_id: local_idx}
        self._id_to_idx:    dict[str, dict[int, int]] = {}
        self._idx_to_id:    dict[str, dict[int, int]] = {}

        # 엣지 저장소
        self._edge_specs:   dict[str, EdgeSpec]     = {}   # name → spec
        self._edge_coo:     dict[str, np.ndarray]   = {}   # name → (2, E) int32
        self._edge_weights: dict[str, Optional[np.ndarray]] = {}  # name → (E,) float32 | None

    # ── 노드 등록 ──────────────────────────────────────────────────────────

    def set_nodes(
        self,
        node_type: str,
        feat: np.ndarray,          # (N, dim) float32
        id_to_idx: dict[int, int], # original_id → local idx
    ) -> None:
        """노드 타입의 피처와 ID 매핑을 등록한다."""
        assert node_type in self.NODE_TYPES, f"Unknown node type: {node_type}"
        self._node_feats[node_type]  = feat.astype(np.float32)
        self._id_to_idx[node_type]   = id_to_idx
        self._idx_to_id[node_type]   = {v: k for k, v in id_to_idx.items()}

    # ── 엣지 등록 / 조회 ──────────────────────────────────────────────────

    def add_edges(
        self,
        name:        str,
        src_type:    str,
        dst_type:    str,
        coo:         np.ndarray,                    # (2, E) int32 — [src_idx; dst_idx]
        weights:     Optional[np.ndarray] = None,   # (E,)  float32
        description: str = "",
        is_directed: bool = True,
        overwrite:   bool = False,
    ) -> None:
        """새 엣지 타입을 등록한다.

        이미 존재하는 이름이면 overwrite=True일 때만 덮어쓴다.

        Args:
            name       : 고유 식별자 (예: "ad_to_ad_sim_topk")
            src_type   : 소스 노드 타입 ("user" | "search" | "ad")
            dst_type   : 목적지 노드 타입
            coo        : (2, E) int32 배열 — coo[0]=src_idx, coo[1]=dst_idx
            weights    : (E,) float32 배열 (없으면 None)
            description: 설명 문자열
            is_directed: 방향성 여부
            overwrite  : 동일 이름 엣지 덮어쓰기 허용
        """
        if name in self._edge_specs and not overwrite:
            raise ValueError(
                f"Edge '{name}' already exists. Use overwrite=True to replace."
            )
        assert src_type in self.NODE_TYPES, f"Unknown src_type: {src_type}"
        assert dst_type in self.NODE_TYPES, f"Unknown dst_type: {dst_type}"
        assert coo.shape[0] == 2, "coo must have shape (2, E)"

        self._edge_specs[name]   = EdgeSpec(name, src_type, dst_type, description, is_directed)
        self._edge_coo[name]     = coo.astype(np.int32)
        self._edge_weights[name] = weights.astype(np.float32) if weights is not None else None

    def remove_edges(self, name: str) -> None:
        """등록된 엣지 타입을 제거한다."""
        for d in (self._edge_specs, self._edge_coo, self._edge_weights):
            d.pop(name, None)

    # ── 조회 API ──────────────────────────────────────────────────────────

    def node_feat(self, node_type: str) -> np.ndarray:
        """(N, dim) float32 피처 행렬을 반환한다."""
        return self._node_feats[node_type]

    def num_nodes(self, node_type: str) -> int:
        return len(self._id_to_idx.get(node_type, {}))

    def edges(self, name: str) -> np.ndarray:
        """(2, E) int32 COO 배열을 반환한다."""
        return self._edge_coo[name]

    def edge_weights(self, name: str) -> Optional[np.ndarray]:
        """(E,) float32 가중치 배열을 반환한다. 없으면 None."""
        return self._edge_weights[name]

    def edge_spec(self, name: str) -> EdgeSpec:
        return self._edge_specs[name]

    def edge_names(self) -> list[str]:
        """등록된 모든 엣지 이름 목록을 반환한다."""
        return list(self._edge_specs.keys())

    def to_idx(self, node_type: str, original_id: int) -> int:
        """원본 ID → 로컬 인덱스 변환."""
        return self._id_to_idx[node_type][original_id]

    def to_id(self, node_type: str, idx: int) -> int:
        """로컬 인덱스 → 원본 ID 변환."""
        return self._idx_to_id[node_type][idx]

    def neighbor_indices(
        self,
        node_type: str,
        node_idx:  int,
        edge_name: str,
        side:      str = "src",   # "src": node_idx가 src, 이웃은 dst
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """한 노드의 이웃 인덱스와 가중치를 반환한다.

        Args:
            node_type : 조회 노드의 타입
            node_idx  : 로컬 인덱스
            edge_name : 엣지 이름
            side      : "src" — node_idx가 src인 엣지, 이웃은 dst
                        "dst" — node_idx가 dst인 엣지, 이웃은 src

        Returns:
            (neighbor_indices, weights)
        """
        spec = self._edge_specs[edge_name]
        coo  = self._edge_coo[edge_name]
        wt   = self._edge_weights[edge_name]

        if side == "src":
            assert node_type == spec.src_type
            mask = coo[0] == node_idx
            nb   = coo[1][mask]
        else:
            assert node_type == spec.dst_type
            mask = coo[1] == node_idx
            nb   = coo[0][mask]

        return nb, (wt[mask] if wt is not None else None)

    # ── 요약 출력 ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = ["=== HeteroGraph Summary ==="]

        lines.append("  [ Nodes ]")
        for nt in self.NODE_TYPES:
            if nt not in self._node_feats:
                continue
            n   = self.num_nodes(nt)
            dim = self._node_feats[nt].shape[1]
            lines.append(f"    {nt:<8} : {n:>8,} nodes  (feat dim={dim})")

        lines.append("  [ Edges ]")
        for name, spec in self._edge_specs.items():
            e  = self._edge_coo[name].shape[1]
            wt = "weighted" if self._edge_weights[name] is not None else "unweighted"
            lines.append(
                f"    {name:<30} : {e:>9,} edges  "
                f"({spec.src_type} → {spec.dst_type}, {wt})"
            )
            if spec.description:
                lines.append(f"      └ {spec.description}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# build_graph — 기본 그래프 구축
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    ds,
    verbose:      bool = True,
    transductive: bool = False,
    include_test: bool = False,
    top_k_sim:    int  = 0,
) -> HeteroGraph:
    """RecoDataset으로부터 이종 그래프를 구축한다.

    Args:
        ds            : RecoDataset (.load() 완료 상태)
        verbose       : 진행 로그 출력 여부
        transductive  : val click search 노드를 그래프에 추가
        include_test  : test search 노드도 추가 (레이블 없음, 구조만)
        top_k_sim     : >0이면 각 search/ad 노드에서 텍스트 유사도 top-K
                        이웃까지 무방향 엣지를 추가한다.
                        0이면 sim 엣지 없음.
                        권장값: 10  (10%는 너무 많아 over-smoothing 유발)

    생성되는 엣지:
        user_to_search           (기본)
        search_to_ad_click       (기본)
        search_to_ad_show        (기본)
        search_to_search_sim     (top_k_sim > 0 일 때)
        ad_to_ad_sim             (top_k_sim > 0 일 때)
    """

    def log(msg):
        if verbose:
            print(msg)

    g   = HeteroGraph()
    dim = 384  # embedding 차원

    # ────────────────────────────────────────────────────────────────
    # 1. Ad 노드
    # ────────────────────────────────────────────────────────────────
    log("[graph] ad 노드 구축 중...")
    ad_feat_matrix, ad_ids_ordered = ds.all_ad_embs()
    ad_id_to_idx = {aid: i for i, aid in enumerate(ad_ids_ordered)}
    g.set_nodes("ad", ad_feat_matrix, ad_id_to_idx)
    log(f"  ad : {g.num_nodes('ad'):,}개")

    # ────────────────────────────────────────────────────────────────
    # 2. Search 노드
    # ────────────────────────────────────────────────────────────────
    log("[graph] search 노드 구축 중...")

    # 훈련 스트림에 등장하는 SearchID만 수집
    search_ids_seen:  list[int] = []
    search_emb_dict:  dict[int, np.ndarray] = {}  # search_id → emb

    for ev in ds.training_stream():
        sid = ev.search_id
        if sid not in search_emb_dict:
            search_ids_seen.append(sid)
            search_emb_dict[sid] = ev.search_emb

    # val_ad_queries search 포함 (Task B inference용)
    for ev in ds.val_ad_queries():
        sid = ev.search_id
        if sid not in search_emb_dict:
            search_ids_seen.append(sid)
            search_emb_dict[sid] = ev.search_emb

    # transductive: click_validation search 노드 추가 (Task A link prediction용)
    if transductive:
        for (ev, _ad) in ds.val_click_queries():
            sid = ev.search_id
            if sid not in search_emb_dict:
                search_ids_seen.append(sid)
                search_emb_dict[sid] = ev.search_emb

    search_id_to_idx = {sid: i for i, sid in enumerate(search_ids_seen)}
    N_search = len(search_id_to_idx)

    search_feat = np.zeros((N_search, dim), dtype=np.float32)
    for sid, idx in search_id_to_idx.items():
        search_feat[idx] = search_emb_dict[sid]

    g.set_nodes("search", search_feat, search_id_to_idx)
    log(f"  search : {g.num_nodes('search'):,}개")

    # ────────────────────────────────────────────────────────────────
    # 3. User 노드
    # ────────────────────────────────────────────────────────────────
    log("[graph] user 노드 구축 중...")

    user_feat_sum:   dict[int, np.ndarray] = defaultdict(lambda: np.zeros(dim, dtype=np.float64))
    user_search_cnt: dict[int, int]        = defaultdict(int)
    all_user_ids:    set[int]              = set()

    for ev in ds.training_stream():
        uid = ev.user_id
        all_user_ids.add(uid)
        user_feat_sum[uid]   += ev.search_emb.astype(np.float64)
        user_search_cnt[uid] += 1

    for ev in ds.val_ad_queries():
        all_user_ids.add(ev.user_id)

    # transductive: click_val 유저도 노드에 포함
    if transductive:
        for (ev, _ad) in ds.val_click_queries():
            all_user_ids.add(ev.user_id)

    user_id_to_idx = {uid: i for i, uid in enumerate(sorted(all_user_ids))}
    N_users        = len(user_id_to_idx)

    # user node는 zero 초기화 — intrinsic feature 없음.
    # GNN message passing (user ← search)에서 search embedding들을 집계해 채움.
    user_feat = np.zeros((N_users, dim), dtype=np.float32)

    g.set_nodes("user", user_feat, user_id_to_idx)
    n_no_search = sum(1 for uid in user_id_to_idx if user_search_cnt[uid] == 0)
    log(f"  user : {N_users:,}개  (no search history={n_no_search:,})")

    # ────────────────────────────────────────────────────────────────
    # 4. 엣지 수집
    # ────────────────────────────────────────────────────────────────
    log("[graph] 엣지 수집 중...")

    user_search_pairs: set[tuple[int, int]] = set()
    click_pos:  dict[tuple[int, int], int] = {}   # (si,ai) → position
    show_pairs: set[tuple[int, int]]       = set()
    user_click_cnt: dict[int, int]         = defaultdict(int)  # ui → 클릭 수

    for ev in ds.training_stream():
        uid = ev.user_id
        sid = ev.search_id
        if uid not in user_id_to_idx or sid not in search_id_to_idx:
            continue
        ui = user_id_to_idx[uid]
        si = search_id_to_idx[sid]
        user_search_pairs.add((ui, si))
        for ad in ev.ads:
            if ad.ad_id not in ad_id_to_idx:
                continue
            ai  = ad_id_to_idx[ad.ad_id]
            key = (si, ai)
            if ad.is_click:
                click_pos[key] = ad.position
                user_click_cnt[ui] += 1
            else:
                show_pairs.add(key)

    def pairs_to_coo_unweighted(pairs):
        if not pairs:
            return np.zeros((2, 0), dtype=np.int32)
        return np.array(list(pairs), dtype=np.int32).T

    # ── search_to_ad_click (균등 가중치, position 기반 제거) ────────────
    # click_weight는 GNNConfig에서 전역으로 제어
    clicked_search_idxs: set[int] = {si for (si, _ai) in click_pos}

    if click_pos:
        click_items = list(click_pos.items())
        click_coo   = np.array([[s, a] for (s, a), _ in click_items],
                               dtype=np.int32).T
    else:
        click_coo   = np.zeros((2, 0), dtype=np.int32)

    g.add_edges(
        name        = "search_to_ad_click",
        src_type    = "search",
        dst_type    = "ad",
        coo         = click_coo,
        weights     = None,
        description = "검색→클릭 광고 (균등, GNN click_weight로 조절)",
    )

    # transductive / include_test: show 엣지 + user_to_search 추가
    if transductive:
        for (ev, ad) in ds.val_click_queries():
            sid = ev.search_id
            if sid not in search_id_to_idx or ad.ad_id not in ad_id_to_idx:
                continue
            show_pairs.add((search_id_to_idx[sid], ad_id_to_idx[ad.ad_id]))
            ui = user_id_to_idx.get(ev.user_id)
            if ui is not None:
                user_search_pairs.add((ui, search_id_to_idx[sid]))

    if include_test:
        for ev in ds.test_ad_queries():
            sid = ev.search_id
            if sid not in search_id_to_idx:
                continue
            ui = user_id_to_idx.get(ev.user_id)
            if ui is not None:
                user_search_pairs.add((ui, search_id_to_idx[sid]))
        for (ev, ad) in ds.test_click_queries():
            sid = ev.search_id
            if sid not in search_id_to_idx or ad.ad_id not in ad_id_to_idx:
                continue
            si = search_id_to_idx[sid]
            show_pairs.add((si, ad_id_to_idx[ad.ad_id]))
            ui = user_id_to_idx.get(ev.user_id)
            if ui is not None:
                user_search_pairs.add((ui, si))
        log("  [include_test] test search/ad 쌍 추가 완료")

    show_coo = pairs_to_coo_unweighted(show_pairs)
    g.add_edges(
        name        = "search_to_ad_show",
        src_type    = "search",
        dst_type    = "ad",
        coo         = show_coo,
        weights     = None,
        description = "검색→노출 광고 (비클릭)"
                      + (" + val" if transductive else "")
                      + (" + test" if include_test else ""),
    )

    # ── user_to_search (클릭 이력 있는 유저 → weight=3.0, 없으면 1.0) ─
    us_list = list(user_search_pairs)
    us_coo  = np.array(us_list, dtype=np.int32).T if us_list else np.zeros((2,0), dtype=np.int32)
    # user_to_search 가중치:
    #   해당 search에서 실제 클릭이 발생한 경우 → 3.0 (강한 신호 search)
    #   클릭 없는 search → 1.0
    us_weights = np.array(
        [3.0 if si in clicked_search_idxs else 1.0 for _, si in us_list],
        dtype=np.float32,
    )
    g.add_edges(
        name        = "user_to_search",
        src_type    = "user",
        dst_type    = "search",
        coo         = us_coo,
        weights     = us_weights,
        description = "유저→검색 (가중치: 클릭발생search=3.0, 없음=1.0)"
                      + (" + val" if transductive else "")
                      + (" + test" if include_test else ""),
    )

    # ── Similarity 엣지 (top_k_sim > 0, weight=raw cosine) ──────────────
    if top_k_sim > 0:
        log(f"\n[graph] similarity 엣지 구축 (top_k={top_k_sim}, weight=cosine)...")

        # Ad-Ad sim: 클릭된 ad → 유사 ad (클릭 신호를 유사 ad로 전파)
        clicked_ad_idxs = np.array(
            sorted({ai for (_si, ai) in click_pos}), dtype=np.int32
        )
        ad_feat = g.node_feat("ad")
        log(f"  ad sim 계산 대상: 클릭발생 {len(clicked_ad_idxs):,}개 → 전체 {g.num_nodes('ad'):,}개")
        a_coo, a_sim_w = _build_topk_sim_coo_from_subset(
            ad_feat, clicked_ad_idxs, top_k_sim,
            chunk_size=500, label="ad-ad", verbose=verbose,
        )
        g.add_edges(
            name        = "ad_to_ad_sim",
            src_type    = "ad",
            dst_type    = "ad",
            coo         = a_coo,
            weights     = a_sim_w,
            description = f"ad 유사도 top-{top_k_sim}, 클릭발생ad→all (가중치=cosine)",
        )
        log(f"  ad_to_ad_sim: {a_coo.shape[1]:,} 방향 엣지")

        # Search-search sim: 클릭이 발생한 search에서만 유사 search로 엣지
        # → 클릭 신호를 유사한 다른 search(val/test 포함)에 전파
        search_feat  = g.node_feat("search")
        clicked_s_idx = np.array(sorted(clicked_search_idxs), dtype=np.int32)
        log(f"  search sim 계산 대상: 클릭발생 {len(clicked_s_idx):,}개 → 전체 {g.num_nodes('search'):,}개")
        s_coo, s_sim_w = _build_topk_sim_coo_from_subset(
            search_feat, clicked_s_idx, top_k_sim,
            chunk_size=500, label="search-search", verbose=verbose,
        )
        g.add_edges(
            name        = "search_to_search_sim",
            src_type    = "search",
            dst_type    = "search",
            coo         = s_coo,
            weights     = s_sim_w,
            description = f"search 유사도 top-{top_k_sim}, 클릭발생search→all (가중치=cosine)",
        )
        log(f"  search_to_search_sim: {s_coo.shape[1]:,} 방향 엣지")

    log("")
    log(g.summary())
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Similarity 엣지 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _build_topk_sim_coo(
    feat: np.ndarray,
    top_k: int,
    chunk_size: int = 2000,
    label: str = "",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """전체 노드 간 top-K 코사인 유사도 엣지 구축 (무방향, cosine² 가중치).

    Returns: (coo (2,E) int32, weights (E,) float32)
    """
    N = feat.shape[0]
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    fn = (feat / (norms + 1e-8)).astype(np.float32)

    seen: dict[tuple[int, int], float] = {}   # (min,max) → cosine²
    n_chunks = (N + chunk_size - 1) // chunk_size

    for c, i in enumerate(range(0, N, chunk_size)):
        end  = min(i + chunk_size, N)
        sims = fn[i:end] @ fn.T                   # (B, N)
        np.fill_diagonal(sims[:, i:end], -2.0)
        k    = min(top_k, N - 1)
        topk = np.argpartition(sims, -k, axis=1)[:, -k:]
        rows = np.arange(end - i)[:, None]
        sim_vals = sims[rows, topk]
        for j in range(end - i):
            src = i + j
            for rank in range(k):
                sv = float(sim_vals[j, rank])
                if sv <= 0:
                    continue
                dst = int(topk[j, rank])
                key = (min(src, dst), max(src, dst))
                seen[key] = max(seen.get(key, 0.0), sv ** 2)  # cosine²
        if verbose and (c + 1) % 5 == 0:
            print(f"    {label}: {c+1}/{n_chunks} chunks, {len(seen):,} 엣지 쌍")

    if not seen:
        return np.zeros((2, 0), dtype=np.int32), np.zeros(0, dtype=np.float32)

    src_list, dst_list, wt_list = [], [], []
    for (s, d), w in seen.items():
        src_list += [s, d]; dst_list += [d, s]; wt_list += [w, w]

    coo = np.array([src_list, dst_list], dtype=np.int32)
    wts = np.array(wt_list, dtype=np.float32)
    return coo, wts


def _build_topk_sim_coo_from_subset(
    feat: np.ndarray,
    subset_idx: np.ndarray,
    top_k: int,
    chunk_size: int = 500,
    label: str = "",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """val/test 노드 → 전체 노드 top-K 유사도 엣지 구축 (무방향, cosine² 가중치).

    Returns: (coo (2,E) int32, weights (E,) float32)
    """
    N_all = feat.shape[0]
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    fn    = (feat / (norms + 1e-8)).astype(np.float32)

    M = len(subset_idx)
    seen: dict[tuple[int, int], float] = {}
    n_chunks = (M + chunk_size - 1) // chunk_size

    for c, i in enumerate(range(0, M, chunk_size)):
        end   = min(i + chunk_size, M)
        s_idx = subset_idx[i:end]
        sims  = fn[s_idx] @ fn.T                   # (B, N_all)
        for j, src in enumerate(s_idx):
            sims[j, int(src)] = -2.0
        k    = min(top_k, N_all - 1)
        topk = np.argpartition(sims, -k, axis=1)[:, -k:]
        rows = np.arange(end - i)[:, None]
        sim_vals = sims[rows, topk]
        for j, src in enumerate(s_idx):
            for rank in range(k):
                sv = float(sim_vals[j, rank])
                if sv <= 0:
                    continue
                dst = int(topk[j, rank])
                if dst == int(src):
                    continue
                key = (min(int(src), dst), max(int(src), dst))
                seen[key] = max(seen.get(key, 0.0), sv)  # raw cosine (크지 않게)
        if verbose and (c + 1) % 10 == 0:
            print(f"    {label}: {c+1}/{n_chunks} chunks, {len(seen):,} 엣지 쌍")

    if not seen:
        return np.zeros((2, 0), dtype=np.int32), np.zeros(0, dtype=np.float32)

    src_list, dst_list, wt_list = [], [], []
    for (s, d), w in seen.items():
        src_list += [s, d]; dst_list += [d, s]; wt_list += [w, w]

    coo = np.array([src_list, dst_list], dtype=np.int32)
    wts = np.array(wt_list, dtype=np.float32)
    return coo, wts
