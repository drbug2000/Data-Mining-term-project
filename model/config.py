"""
ModelConfig — 모든 하이퍼파라미터를 한 곳에서 정의한다.

사용법:
    from model.config import ModelConfig

    cfg = ModelConfig()               # 기본값으로 생성
    cfg = ModelConfig(k=3, alpha_click=0.8)  # 일부만 변경
    cfg.save("cfg.json")              # JSON 저장
    cfg = ModelConfig.load("cfg.json")       # JSON 불러오기
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    k: int = 5
    """유저 1인당 유지하는 interest vector 개수.
    값이 클수록 다양한 관심사를 표현할 수 있지만 데이터가 부족하면 오히려 노이즈."""

    dim: int = 384
    """임베딩 차원. 데이터셋 고정값(384)과 반드시 일치해야 한다."""

    # ------------------------------------------------------------------
    # Update rates  (0 = 비활성, 1 = 한 번에 완전 덮어쓰기)
    # ------------------------------------------------------------------

    alpha_search: float = 0.1
    """검색 이벤트 발생 시 interest 업데이트 강도.
    검색은 광고 클릭보다 약한 신호이므로 작은 값을 사용한다."""

    alpha_click: float = 0.5
    """광고 클릭 시 interest 업데이트 강도.
    클릭은 명확한 선호 신호이므로 alpha_search보다 크게 설정한다."""

    alpha_neg: float = 0.0
    """비클릭 광고에 대한 negative 업데이트 강도.
    0이면 비활성(권장 기본값). position bias로 인해 비클릭 ≠ 비관심일 수 있으므로
    사용 시 매우 작은 값(예: 0.01)으로 설정한다."""

    # ------------------------------------------------------------------
    # Soft assignment
    # ------------------------------------------------------------------

    temperature: float = 1.0
    """Soft assignment의 부드러움을 조절하는 온도 파라미터 τ.
    - 낮을수록(예: 0.1): 가장 가까운 interest vector 하나가 거의 모든 가중치를 가져감 (hard)
    - 높을수록(예: 5.0): 모든 interest vector에 균등하게 분산됨 (soft)"""

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    gamma: float = 0.5
    """클릭 이력이 있는 유저의 user interest 기여 비율 (높은 가중치).
    score = (1-gamma) * sim(query, ad) + gamma * sim(best_interest, ad)"""

    gamma_search: float = 0.0
    """검색 이력만 있는 유저(클릭 없음)의 user interest 기여 비율 (낮은 가중치).
    0이면 기존 adaptive gamma 동작(click 유저만 활성화)과 동일.
    cold-start 유저(검색 기록조차 없음)는 항상 0으로 폴백.
    권장: gamma_search < gamma."""

    threshold: float = 0.8
    """Task A 클릭 예측 임계값.
    user interest와 ad 사이의 최소 거리가 threshold 미만이면 클릭으로 예측한다."""

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> ModelConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def __str__(self) -> str:
        lines = ["ModelConfig("]
        for k, v in self.to_dict().items():
            lines.append(f"    {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)
