"""
config.py — GatedCTRModel 하이퍼파라미터 dataclass (gyuchan 버전).

model/config.py 의 ModelConfig 와 동일한 스타일(필드별 docstring + save/load)을 따른다.
BaseRecoModel 은 self.config 를 저장만 하므로 ModelConfig 대신 이 GateConfig 를 그대로
넘겨도 동작한다.

사용법:
    from model_gyuchan.config import GateConfig

    cfg = GateConfig()                     # 기본값
    cfg = GateConfig(proj_dim=32, gate_s=0.5)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class GateConfig:
    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    dim: int = 384
    """임베딩 차원. 데이터셋 고정값(384)과 일치해야 한다."""

    proj_dim: int = 64
    """content head 의 저차원 상호작용 부분공간 크기 (low-rank bilinear rank).
    클릭으로 학습되는 query x ad 매칭 부분공간의 차원."""

    # ------------------------------------------------------------------
    # Content head 학습 (low-rank bilinear, numpy + Adam)
    # ------------------------------------------------------------------

    head_epochs: int = 60
    """content head 최대 epoch 수 (early stopping 이 보통 먼저 멈춘다). 원본 bi-encoder 와 동일."""

    head_batch: int = 8192
    """content head 미니배치 크기."""

    head_lr: float = 1e-3
    """content head Adam learning rate."""

    head_wd: float = 1e-3
    """content head weight decay (U, V 행렬에만 적용)."""

    head_patience: int = 8
    """content head early stopping patience (내부 val AUC 기준). 원본 bi-encoder 와 동일."""

    # ------------------------------------------------------------------
    # Entity (CTR) path — logistic regression over trusted log-CTRs
    # ------------------------------------------------------------------

    k_smooth: int = 20
    """엔티티 CTR Laplace smoothing pseudo-count. 소수 관측치를 global CTR 로 당긴다."""

    ent_epochs: int = 200
    """entity logistic regression 최대 epoch."""

    ent_lr: float = 5e-2
    """entity logistic regression learning rate."""

    login_boost: float = 1.7
    """로그인 유저 클릭 보정 배수. log(login_boost) 가 not-logged-on 행에 음의 offset 으로 들어간다."""

    # ------------------------------------------------------------------
    # Interest (personalization) — per-user clicked-ad prototypes
    # ------------------------------------------------------------------

    interest_cap: int = 200
    """유저당 보관하는 클릭 광고 임베딩 최대 개수 (memory cap, 초과 시 최근 것 유지)."""

    interest_beta: float = 0.5
    """content path 에 더해지는 interest(클릭 이력) 유사도 가중치. 클릭 이력 없는 유저는 0."""

    task_b_interest_beta: float = 0.5
    """Task B 광고 추천에서 raw query-ad cosine 에 더하는 클릭 광고 interest 유사도 가중치."""

    # ------------------------------------------------------------------
    # Support gate — warm(entity) vs cold(content) 자동 전환
    # ------------------------------------------------------------------

    use_gate: bool = False
    """True 면 support 시그모이드 게이트(warm=entity/cold=content)로 결합.
    False(기본)면 [5 CTR + z(content)] F1-objective log-linear (원본 0.1106 재현 경로)."""

    gate_s: float = 1.0
    """게이트 logistic ramp 의 부드러움. 작을수록 hard 스위치, 클수록 완만."""

    gate_t: float | None = None
    """게이트 임계 support 값. None 이면 학습 데이터 support 의 중앙값으로 자동 설정."""

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    seed: int = 42
    """재현용 random seed."""

    # ------------------------------------------------------------------
    # Serialization helpers (ModelConfig 와 동일 인터페이스)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GateConfig:
        valid = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> GateConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def __str__(self) -> str:
        lines = ["GateConfig("]
        for k, v in self.to_dict().items():
            lines.append(f"    {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)
