"""
HybridConfig — GNN (m02) + MultiInterest (m01) 결합 모델의 하이퍼파라미터.

모든 필드를 flat 하게 정의하고 gnn_config() 로 GNNConfig 인스턴스를 생성한다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from models.m02_gnn.gnn import GNNConfig


@dataclass
class HybridConfig:

    # ── Architecture (m01 계승) ────────────────────────────────────────
    k: int = 5
    """유저 1인당 유지하는 interest vector 수."""

    dim: int = 384
    """임베딩 차원. 데이터셋 고정값(384)과 일치해야 한다."""

    # ── 온라인 업데이트 강도 (m01 계승) ──────────────────────────────
    alpha_search: float = 0.1
    alpha_click:  float = 0.5
    alpha_neg:    float = 0.0
    temperature:  float = 1.0

    # ── Adaptive gamma (m01 계승) ─────────────────────────────────────
    gamma:        float = 0.7
    """클릭 이력 있는 유저의 총 personalization 가중치."""

    gamma_search: float = 0.3
    """검색 이력만 있는 유저의 총 personalization 가중치."""

    threshold:    float = 0.8
    """Task A 클릭 예측 임계값."""

    # ── GNN 전파 설정 (m02 계승) ──────────────────────────────────────
    n_layers:       int   = 2
    agg_fn:         str   = "mean"
    normalize:      bool  = True
    click_weight:   float = 1.0
    residual_alpha: float = 0.0
    user_click_init: bool = False

    # ── 하이브리드 전용 ───────────────────────────────────────────────
    link_alpha: float = 0.5
    """Task A score_click 에서 GNN link prediction 비율.
      0.0 → 순수 m01 click interest score
      1.0 → 순수 GNN score_link(h_search, h_ad)
      0.5 → 두 신호 동등 결합 (기본값)
    search_id / ad_id 가 제공되지 않거나 GNN 미학습이면 m01 score 로 폴백.
    Task B (score_ad_candidates) 에는 영향 없음 — 항상 m01 다중 interest 방식.
    """

    # ── 직렬화 ────────────────────────────────────────────────────────

    def gnn_config(self) -> GNNConfig:
        """GNNModel 생성용 GNNConfig 를 반환한다."""
        return GNNConfig(
            n_layers=self.n_layers,
            agg_fn=self.agg_fn,
            normalize=self.normalize,
            click_weight=self.click_weight,
            residual_alpha=self.residual_alpha,
            user_click_init=self.user_click_init,
            gamma=self.gamma,
            gamma_search=self.gamma_search,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> HybridConfig:
        valid = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> HybridConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def __str__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"HybridConfig({items})"
