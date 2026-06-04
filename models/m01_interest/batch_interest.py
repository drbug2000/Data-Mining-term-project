"""
BatchMultiInterestModel — 배치 방식 interest vector 구축.

스트리밍 soft-assignment 대신, 훈련 스트림 전체를 한 번 읽으면서
각 유저의 embedding을 모두 수집한 뒤, build_interests() 호출 시
일괄 클러스터링으로 interest vector를 생성한다.

지원 방법 (method=):
  "kmeans"   Spherical K-Means (k-means++ 초기화, 코사인 유사도)
             L2 정규화된 벡터에서 최대 도트곱 = 최소 유클리드 거리이므로
             구면 K-Means가 적합하다.
  "svd"      top-k 우측 특이벡터 — 유저 embedding 행렬의 최대 분산 방향.
             서로 직교(orthogonal)한 k개의 관심사 축을 제공한다.
  "mean"     가중 평균 벡터를 k개 복제 — 사실상 k=1 degenerate 기준선.
             interest 분리가 전혀 없을 때 얼마나 성능이 떨어지는지 확인.
  "diverse"  Greedy farthest-point 선택 — embedding 공간 최대 커버리지.
             중심에서 가장 가까운 것을 시작점으로, 이후 선택된 집합과
             가장 다른 embedding을 순차 추가한다.

click_weight (float):
  _interests (Task B) 구축 시 클릭 embedding을 반복 추가하는 배수.
  검색 대비 클릭 신호를 얼마나 강조할지 제어한다.
  _click_interests (Task A) 는 클릭 embedding만 사용하므로 무관.

사용 예:
    from model.batch_interest import BatchMultiInterestModel
    from model.predictor import train

    model = BatchMultiInterestModel(config, method="kmeans", click_weight=5.0)
    train(model, ds.training_stream())   # embedding 수집
    model.build_interests()              # 클러스터링 실행
    # 이후 MultiInterestModel과 동일하게 예측 가능
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from models.m01_interest.config import ModelConfig
from models.m01_interest.interest import MultiInterestModel, _l2_normalize


class BatchMultiInterestModel(MultiInterestModel):
    """배치 방식으로 interest vector를 구축하는 모델.

    MultiInterestModel을 상속해 예측 로직을 그대로 사용하고,
    update_search / update_click 만 오버라이드해 raw embedding을 수집한다.
    build_interests() 호출 후에야 _interests / _click_interests 가 채워진다.
    """

    METHODS = frozenset({"kmeans", "svd", "mean", "diverse"})

    def __init__(
        self,
        config: ModelConfig,
        method: str = "kmeans",
        click_weight: float = 5.0,
        seed: int = 42,
    ):
        super().__init__(config)
        if method not in self.METHODS:
            raise ValueError(
                f"method는 {sorted(self.METHODS)} 중 하나여야 합니다. 입력값: {method!r}"
            )
        self.method = method
        self.click_weight = click_weight
        self.seed = seed

        # 훈련 중 수집하는 raw embedding 저장소
        self._raw_search: dict[int, list[np.ndarray]] = defaultdict(list)
        self._raw_click: dict[int, list[np.ndarray]] = defaultdict(list)

    # ------------------------------------------------------------------
    # 훈련 — embedding만 수집, interest vector 업데이트 없음
    # ------------------------------------------------------------------

    def update_search(self, user_id: int, search_emb: np.ndarray) -> None:
        """검색 발생 → L2 정규화 후 raw_search에 저장."""
        self._searched_users.add(user_id)
        self._raw_search[user_id].append(_l2_normalize(search_emb))

    def update_click(self, user_id: int, ad_emb: np.ndarray, clicked: bool) -> None:
        """클릭 발생 시 raw_click에 저장. 비클릭은 무시."""
        if clicked:
            self._clicked_users.add(user_id)
            self._raw_click[user_id].append(_l2_normalize(ad_emb))

    # ------------------------------------------------------------------
    # 배치 interest 구축 — 훈련 완료 후 반드시 호출
    # ------------------------------------------------------------------

    def build_interests(self) -> None:
        """수집된 embedding 전체를 사용해 모든 유저의 interest vector를 일괄 구축.

        - _interests (Task B): 검색 + 클릭 (클릭은 click_weight 배 반복)
        - _click_interests (Task A): 클릭 embedding만 사용
          클릭 이력 없는 유저는 예측 시 _get_or_init_store가 랜덤 초기화
          (effective_gamma=0 이므로 실제로는 참조되지 않음)
        """
        k = self.config.k
        reps = max(1, round(self.click_weight))

        for uid in set(self._raw_search) | set(self._raw_click):
            s_embs = self._raw_search.get(uid, [])
            c_embs = self._raw_click.get(uid, [])

            # Task B: 검색 + 클릭 (가중)
            combined = s_embs + c_embs * reps
            self._interests[uid] = self._build(combined, k)

            # Task A: 클릭 only
            if c_embs:
                self._click_interests[uid] = self._build(c_embs, k)

    def _build(self, embs: list[np.ndarray], k: int) -> np.ndarray:
        """선택된 방법으로 k개의 interest vector를 반환한다."""
        if not embs:
            return _random_vecs(k, self.config.dim, self.seed)

        E = np.stack(embs).astype(np.float32)  # (n, dim)

        if self.method == "kmeans":
            return _kmeans(E, k, seed=self.seed)
        if self.method == "svd":
            return _svd(E, k, seed=self.seed)
        if self.method == "mean":
            return _mean(E, k)
        if self.method == "diverse":
            return _diverse(E, k, seed=self.seed)
        raise AssertionError(f"도달 불가: {self.method}")  # METHODS 검증에서 막힘


# ──────────────────────────────────────────────────────────────────────
# 클러스터링 헬퍼 함수
# ──────────────────────────────────────────────────────────────────────

def _random_vecs(k: int, dim: int, seed: int) -> np.ndarray:
    """랜덤 정규분포 → L2 정규화 벡터 k개."""
    v = np.random.default_rng(seed).standard_normal((k, dim)).astype(np.float32)
    return _l2_normalize(v)


def _pad_to_k(embs: list[np.ndarray], k: int, seed: int) -> np.ndarray:
    """embedding 수가 k보다 적을 때, 나머지를 랜덤 벡터로 채운다."""
    dim = embs[0].shape[0]
    result = list(embs)
    rng = np.random.default_rng(seed + len(embs))  # 길이에 따라 seed 변화
    while len(result) < k:
        v = rng.standard_normal(dim).astype(np.float32)
        result.append(_l2_normalize(v))
    return _l2_normalize(np.stack(result))


def _kmeans(E: np.ndarray, k: int, n_iter: int = 100, seed: int = 42) -> np.ndarray:
    """Spherical K-Means — k-means++ 초기화, 코사인 유사도 기반 할당.

    L2 정규화된 embedding에서 코사인 유사도 최대 = 유클리드 거리 최소이므로
    argmax(dot-product) 으로 할당하고, 중심 업데이트 후 재정규화(구면 K-Means).

    Args:
        E      : (n, dim) L2 정규화된 float32 행렬
        k      : 클러스터 수
        n_iter : 최대 반복 횟수
        seed   : 난수 시드

    Returns:
        centers: (k, dim) L2 정규화된 클러스터 중심
    """
    n, dim = E.shape
    rng = np.random.default_rng(seed)

    if n <= k:
        return _pad_to_k(list(E), k, seed)

    # k-means++ 초기화 (코사인 거리 기반 확률적 선택)
    first_idx = int(rng.integers(n))
    centers = [E[first_idx]]

    for _ in range(k - 1):
        sims = E @ np.stack(centers).T          # (n, s) 코사인 유사도
        max_sim = sims.max(axis=1)              # (n,) 가장 가까운 중심과의 유사도
        cos_dist = np.maximum(1.0 - max_sim, 0.0)   # 코사인 거리 ∈ [0, 2]
        total = cos_dist.sum()
        prob = cos_dist / total if total > 1e-12 else np.ones(n) / n
        centers.append(E[int(rng.choice(n, p=prob))])

    centers = _l2_normalize(np.stack(centers))  # (k, dim)

    # Spherical K-Means 반복
    for _ in range(n_iter):
        # 할당: 각 점에 대해 가장 유사한 중심 선택
        labels = (E @ centers.T).argmax(axis=1)   # (n,)

        # 중심 업데이트 + 재정규화
        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = labels == i
            new_centers[i] = E[mask].mean(axis=0) if mask.any() else centers[i]

        new_centers = _l2_normalize(new_centers)

        if np.allclose(centers, new_centers, atol=1e-5):
            break
        centers = new_centers

    return centers  # (k, dim)


def _svd(E: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Top-k 우측 특이벡터 — 유저 embedding 행렬의 최대 분산 방향.

    E = U Σ V^T 에서 V^T의 첫 k행(=V의 첫 k열)이 반환값.
    각 행이 하나의 interest 방향이며, 서로 직교한다.

    중심화(centering) 없이 원점 기준 분산을 포착한다.
    L2 정규화된 embedding은 단위구 위에 있으므로 원점 기준 방향이 더 의미 있다.

    Args:
        E    : (n, dim) float32, L2 정규화 권장
        k    : 반환할 특이벡터 수
        seed : 부족 시 랜덤 패딩에 사용

    Returns:
        (k, dim) L2 정규화된 특이벡터 행렬
    """
    n, dim = E.shape

    if n == 1:
        return _pad_to_k([E[0]], k, seed)

    try:
        # full_matrices=False → thin SVD, 빠름
        _, _, Vt = np.linalg.svd(E.astype(np.float64), full_matrices=False)
    except np.linalg.LinAlgError:
        return _random_vecs(k, dim, seed)

    Vt = Vt.astype(np.float32)          # (min(n, dim), dim)
    available = Vt.shape[0]

    if available >= k:
        return _l2_normalize(Vt[:k])

    # 특이벡터가 k개보다 적으면 랜덤으로 패딩
    return _pad_to_k(list(Vt), k, seed)


def _mean(E: np.ndarray, k: int) -> np.ndarray:
    """모든 embedding의 가중 평균 1개를 k개 복제.

    k개 interest vector가 모두 동일하므로 다중 관심사 표현 능력이 없다.
    배치 방식의 최하위 기준선(degenerate baseline) 역할.

    Args:
        E : (n, dim)
        k : 반환할 행 수 (동일 벡터를 k번 복제)

    Returns:
        (k, dim) 동일 벡터 k개
    """
    mean_v = _l2_normalize(E.mean(axis=0))              # (dim,)
    return np.tile(mean_v, (k, 1)).astype(np.float32)   # (k, dim)


def _diverse(E: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Greedy farthest-point 선택 — embedding 공간 최대 커버리지.

    알고리즘:
      1. 중심(centroid)과 코사인 유사도가 가장 높은 embedding을 시작점으로.
      2. 이미 선택된 집합과 최대 코사인 유사도가 최소인 embedding을 순차 추가.
         즉, 선택된 어떤 벡터와도 가장 다른 벡터를 고른다.

    실제 embedding을 직접 사용하므로 interpretability 가 높다.
    데이터 수가 충분할 때 가장 효과적이다.

    Args:
        E    : (n, dim) L2 정규화된 float32
        k    : 선택할 벡터 수
        seed : 부족 시 랜덤 패딩에 사용

    Returns:
        (k, dim) 선택된 embedding, L2 정규화
    """
    n, dim = E.shape

    if n <= k:
        return _pad_to_k(list(E), k, seed)

    # 시작: centroid와 가장 가까운 embedding (가장 대표적인 관심사)
    centroid = _l2_normalize(E.mean(axis=0))     # (dim,)
    selected = [int((E @ centroid).argmax())]

    for _ in range(k - 1):
        sel_E = E[selected]                       # (s, dim)
        # 각 후보와 이미 선택된 집합 간 최대 코사인 유사도
        max_sim_to_sel = (E @ sel_E.T).max(axis=1)  # (n,)
        # 이미 선택된 index를 제외 (2.0은 cosine similarity 최대값 1.0 초과)
        max_sim_to_sel[selected] = 2.0
        # 가장 다른(min) 벡터 선택
        selected.append(int(max_sim_to_sel.argmin()))

    return _l2_normalize(E[selected])             # (k, dim)
