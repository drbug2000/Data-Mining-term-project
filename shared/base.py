"""
BaseRecoModel — 모든 추천 모델이 구현해야 하는 인터페이스.

shared/eval/ 과 각 models/mXX/experiments/ 는
이 파일만 바라보면 된다. 구체적인 모델 구현(models/mXX/*.py)은
이 인터페이스를 상속해서 작성한다.

┌─────────────────────────────────────────────┐
│              사용 흐름                       │
│                                             │
│  1. 훈련: update_search / update_click 반복  │
│  2. 예측: predict_ad / predict_click 호출    │
│  3. 디버깅: get_interests 로 벡터 확인       │
└─────────────────────────────────────────────┘
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

class BaseRecoModel(ABC):
    """추천 모델 인터페이스.

    구현 클래스는 __init__(self, config) 를 정의하고
    아래 4개의 abstractmethod를 모두 구현해야 한다.
    config 타입은 각 모델 디렉토리의 XxxConfig dataclass를 사용한다.
    """

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------------
    # 훈련 (스트리밍 업데이트)
    # ------------------------------------------------------------------

    @abstractmethod
    def update_search(self, user_id: int, search_emb: np.ndarray) -> None:
        """검색 이벤트 발생 시 user interest를 업데이트한다.

        Args:
            user_id    : 유저 식별자
            search_emb : 검색어 임베딩, shape (dim,)
        """

    @abstractmethod
    def update_click(self, user_id: int, ad_emb: np.ndarray, clicked: bool) -> None:
        """광고 노출 이벤트 발생 시 user interest를 업데이트한다.

        clicked=True  → ad_emb 방향으로 interest를 당김  (+alpha_click)
        clicked=False → ad_emb 반대 방향으로 interest를 밂 (-alpha_neg)
                        (config.alpha_neg == 0 이면 아무 것도 하지 않음)

        Args:
            user_id  : 유저 식별자
            ad_emb   : 광고 제목 임베딩, shape (dim,)
            clicked  : 클릭 여부
        """

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------

    @abstractmethod
    def predict_ad(
        self,
        user_id: int,
        query_emb: np.ndarray,
        candidate_embs: np.ndarray,
        candidate_ids: list[int],
    ) -> int:
        """Task B — 후보 광고 중 클릭 가능성이 가장 높은 AdID를 반환한다.

        Args:
            user_id        : 유저 식별자
            query_emb      : 검색어 임베딩, shape (dim,)
            candidate_embs : 후보 광고 임베딩 행렬, shape (N_candidates, dim)
            candidate_ids  : 후보 광고 ID 리스트, len == N_candidates

        Returns:
            int: 선택된 AdID
        """

    @abstractmethod
    def predict_click(
        self,
        user_id: int,
        query_emb: np.ndarray,
        ad_emb: np.ndarray,
    ) -> int:
        """Task A — 해당 광고를 클릭할지 예측한다.

        Args:
            user_id   : 유저 식별자
            query_emb : 검색어 임베딩, shape (dim,)
            ad_emb    : 광고 제목 임베딩, shape (dim,)

        Returns:
            int: 1 (클릭 예측) or 0 (비클릭 예측)
        """

    # ------------------------------------------------------------------
    # 스코어링 (선택 구현 — 평가 지표 계산에 필요)
    # ------------------------------------------------------------------

    def score_ad_candidates(
        self,
        user_id: int,
        query_emb: np.ndarray,
        candidate_embs: np.ndarray,
    ) -> np.ndarray:
        """Task B — 모든 후보 광고에 대한 연속 점수를 반환한다.

        Returns:
            np.ndarray: shape (N_candidates,). 높을수록 클릭 가능성이 높다.
            predict_ad 는 이 배열의 argmax 가 된다.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement score_ad_candidates()"
        )

    def score_click(
        self,
        user_id: int,
        query_emb: np.ndarray,
        ad_emb: np.ndarray,
    ) -> float:
        """Task A — 클릭 가능성에 대한 연속 점수를 반환한다.

        Returns:
            float: 높을수록 클릭 가능성이 높다.
            predict_click 은 이 값이 threshold 를 넘는지로 결정된다.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement score_click()"
        )

    # ------------------------------------------------------------------
    # 디버깅 / 분석용 (선택 구현)
    # ------------------------------------------------------------------

    def get_interests(self, user_id: int) -> np.ndarray:
        """유저의 interest vector 행렬을 반환한다.

        Returns:
            np.ndarray: shape (k, dim). 유저가 없으면 zeros 반환.

        구현하지 않아도 되지만, 디버깅·시각화 시 유용하다.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_interests()"
        )
