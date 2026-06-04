# 코딩 컨벤션 — AI506 Term Project

> 이 문서는 프로젝트의 실제 코드에서 귀납적으로 추출한 규칙을 정리한다.  
> 새 파일을 추가하거나 기존 코드를 수정할 때 이 규칙을 따른다.

---

## 1. 프로젝트 구조

```
project-root/
├── model/          # 모델 구현 (BaseRecoModel 상속 클래스들)
│   ├── base.py         추상 인터페이스 (수정 자제)
│   ├── config.py       하이퍼파라미터 dataclass
│   ├── interest.py     MultiInterestModel (스트리밍 방식)
│   ├── batch_interest.py BatchMultiInterestModel (배치 방식)
│   ├── gnn.py          GNNModel
│   ├── ctr_mlp.py      CTRPredictor
│   ├── predictor.py    공통 학습·평가 함수 (train / score / evaluate)
│   └── __init__.py     공개 API 선언
├── data/           # 데이터 로딩 (git 제외)
│   ├── dataset.py      RecoDataset
│   └── graph.py        그래프 구축
├── experiments/    # 실험 스크립트 (모델 코드 포함 금지)
├── clustering_results/ # 시각화 결과 PNG
├── PLAN.md         설계 문서
├── REPORT.md       실험 보고서
├── EXPERIMENT_LOG.md 실험 이력
└── CONVENTIONS.md  이 파일
```

### 원칙
- **`model/`** 은 데이터 로딩·실험 로직 없이 순수 모델 코드만 포함한다.
- **`experiments/`** 스크립트는 모델을 import해서 사용할 뿐, 직접 구현하지 않는다.
- **`data/`** 는 `.gitignore`에 포함 — 대용량 `.npy` / `.csv` 파일 포함.

---

## 2. Python 코딩 스타일

### 2-1. 파일 헤더 (필수)

모든 `.py` 파일은 **모듈 docstring**으로 시작한다.

```python
"""
module_name.py — 한 줄 한국어 설명.

긴 설명이 필요하면 이곳에 작성한다.
알고리즘 요약, 사용 예시 등을 포함한다.

실행 (실험 스크립트):
    python -X utf8 experiments/module_name.py
"""
```

- 첫 줄 형식: `파일명.py — 설명` (em-dash `—` 사용)
- 실험 스크립트는 실행 명령어를 docstring 마지막에 명시한다.

### 2-2. `from __future__ import annotations` (필수)

모든 `.py` 파일 **첫 번째 import**로 선언한다.

```python
from __future__ import annotations
```

Python 3.10 미만에서도 `X | Y` 타입 힌트 등 최신 문법을 사용할 수 있게 한다.

### 2-3. import 순서

```python
# 1. __future__
from __future__ import annotations

# 2. 표준 라이브러리 (알파벳 순)
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# 3. 외부 라이브러리 (numpy 우선)
import numpy as np
import pandas as pd
# sklearn 등

# 4. 내부 모듈 (data → model 순)
from data.dataset import RecoDataset
from model.base import BaseRecoModel
from model.config import ModelConfig
```

각 그룹 사이에 **빈 줄 하나**를 넣는다. 그룹 내부는 알파벳 순 정렬을 권장한다.

### 2-4. 들여쓰기 및 줄 길이

- 들여쓰기: **4 spaces** (탭 사용 금지)
- 줄 길이: 엄격한 제한 없음, 가독성 우선. 긴 수식은 적절히 줄바꿈.

### 2-5. 타입 힌트 (권장)

`model/` 코드는 타입 힌트를 **적극 사용**한다.  
실험 스크립트는 생략 가능.

```python
# Good
def update_search(self, user_id: int, search_emb: np.ndarray) -> None:

def score_ad_candidates(
    self,
    user_id: int,
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
) -> np.ndarray:

# 컬렉션
self._interests: dict[int, np.ndarray] = {}
self._clicked_users: set[int] = set()
```

---

## 3. 명명 규칙

### 3-1. 파일 이름

| 위치 | 규칙 | 예시 |
|------|------|------|
| `model/` | `snake_case.py` | `batch_interest.py`, `ctr_mlp.py` |
| `experiments/` | `snake_case.py`, 목적 명확하게 | `baseline_eval.py`, `gamma_sweep_by_group.py` |
| 문서 | `UPPER_CASE.md` | `PLAN.md`, `REPORT.md` |

### 3-2. 클래스

**PascalCase** + 역할을 나타내는 suffix.

```python
class BaseRecoModel       # 추상 인터페이스
class MultiInterestModel  # 구체 모델
class BatchMultiInterestModel
class ModelConfig         # 하이퍼파라미터
class CTRPredictor        # 예측기
class GNNModel
```

### 3-3. 함수 / 메서드

**snake_case**.

```python
# 공개 메서드
def update_search(...)
def predict_ad(...)
def build_interests(...)

# 비공개 헬퍼 — 앞에 _ 하나
def _soft_weights(...)
def _get_or_init_store(...)
def _effective_gamma(...)

# 모듈 레벨 유틸 — 앞에 _ 하나
def _l2_normalize(x):
def _kmeans(E, k):
def _binary_auc(scores, labels):
```

### 3-4. 변수 / 상수

```python
# 상수 (스크립트 최상단) — UPPER_CASE
DATASET_DIR = ROOT / "../datasets"
SEED        = 42
CONFIG      = ModelConfig(...)

# 일반 변수 — snake_case
user_id, search_emb, candidate_ids

# numpy 행렬 — 대문자 단일 문자 관용
E   # embedding matrix (n, dim)
V   # interest matrix (k, dim)
C   # candidate embedding matrix (N, dim)
```

---

## 4. 코드 섹션 구분

### 4-1. 클래스 내부 섹션 — `# ──` (얇은 선)

```python
class MultiInterestModel(BaseRecoModel):

    # ------------------------------------------------------------------
    # 훈련 (BaseRecoModel 구현)
    # ------------------------------------------------------------------

    def update_search(self, ...): ...

    # ------------------------------------------------------------------
    # 예측 (BaseRecoModel 구현)
    # ------------------------------------------------------------------

    def predict_ad(self, ...): ...

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _soft_weights(self, ...): ...
```

### 4-2. 실험 스크립트 섹션 — `# ──` (얇은 선, 번호 포함)

```python
# ── 1. 데이터 로드 ────────────────────────
section("1. Dataset")
...

# ── 2. 모델 학습 ──────────────────────────
section("2. Training")
...
```

`section()` 헬퍼 함수를 각 실험 스크립트 상단에 정의한다:

```python
def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
```

### 4-3. 모듈 레벨 함수 그룹 — `# ──` (얇은 선)

```python
# ──────────────────────────────────────────────
# 클러스터링 헬퍼
# ──────────────────────────────────────────────

def _kmeans(E, k): ...
def _svd(E, k): ...
```

---

## 5. Docstring 규칙

### 5-1. 함수 docstring

```python
def score_ad_candidates(
    self,
    user_id: int,
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
) -> np.ndarray:
    """Task B: 모든 후보에 대한 연속 점수를 반환한다. shape (N_candidates,)

    Adaptive gamma (3단계):
      click 이력 → gamma / search 이력만 → gamma_search / cold → 0

    Args:
        user_id        : 유저 식별자
        query_emb      : 검색어 임베딩, shape (dim,)
        candidate_embs : 후보 광고 임베딩 행렬, shape (N, dim)

    Returns:
        np.ndarray shape (N,). 높을수록 클릭 가능성이 높다.
    """
```

- 첫 줄: 한국어 한 문장 요약 (마침표 포함)
- 이후: 알고리즘 설명, Args, Returns 순서
- `Args:` / `Returns:` 는 필요할 때만 작성 (self-explanatory한 경우 생략 가능)

### 5-2. dataclass 필드 docstring

```python
@dataclass
class ModelConfig:
    k: int = 5
    """유저 1인당 유지하는 interest vector 개수."""

    alpha_click: float = 0.5
    """광고 클릭 시 interest 업데이트 강도."""
```

필드 바로 아래에 한 줄 또는 여러 줄 docstring을 붙인다.

---

## 6. 모델 추가 방법

새 추천 모델은 `model/base.py`의 `BaseRecoModel`을 상속한다.

```python
# model/my_model.py
from __future__ import annotations

import numpy as np
from model.base import BaseRecoModel
from model.config import ModelConfig


class MyModel(BaseRecoModel):

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # 내부 상태 초기화

    # ------------------------------------------------------------------
    # 필수 구현 (abstractmethod)
    # ------------------------------------------------------------------

    def update_search(self, user_id: int, search_emb: np.ndarray) -> None: ...
    def update_click(self, user_id: int, ad_emb: np.ndarray, clicked: bool) -> None: ...
    def predict_ad(self, user_id, query_emb, candidate_embs, candidate_ids) -> int: ...
    def predict_click(self, user_id, query_emb, ad_emb) -> int: ...

    # ------------------------------------------------------------------
    # 권장 구현 (평가에 필요)
    # ------------------------------------------------------------------

    def score_ad_candidates(self, user_id, query_emb, candidate_embs) -> np.ndarray: ...
    def score_click(self, user_id, query_emb, ad_emb) -> float: ...
```

`model/__init__.py`에 추가:

```python
from model.my_model import MyModel
```

---

## 7. 실험 스크립트 작성 규칙

```python
"""
experiment_name.py — 실험 목적 한 줄 설명.

실행:
    python -X utf8 experiments/experiment_name.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent      # 프로젝트 루트 (항상 이 방식)
sys.path.insert(0, str(ROOT))

from data.dataset import RecoDataset
from model import ModelConfig, MultiInterestModel
from model.predictor import train, score_task_a, score_task_b, evaluate_task_a, evaluate_task_b_ndcg

DATASET_DIR = ROOT / "../datasets"       # 데이터셋 경로 (항상 이 위치)
SEED = 42                                # 재현성을 위한 고정 시드


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.2f}s"


def main() -> None:
    np.random.seed(SEED)
    # ...


if __name__ == "__main__":
    main()
```

### 핵심 원칙
- `ROOT`, `DATASET_DIR`, `SEED`는 항상 동일한 방식으로 정의한다.
- `main()` 함수에 로직을 집약하고 `if __name__ == "__main__": main()` 으로 호출한다.
- 실행 시 반드시 `-X utf8` 플래그를 사용한다 (한글 출력 안전).
- 결과는 `section()` / `print()`로 콘솔 출력. 외부 파일 저장은 선택 사항.

---

## 8. 평가 함수 사용 규칙

공통 평가 함수는 `model/predictor.py`에만 정의한다.  
실험 스크립트에서 중복 구현 **금지**.

```python
from model.predictor import (
    train,                  # 훈련 스트림 실행
    score_task_a,           # Task A 연속 점수
    score_task_b,           # Task B 연속 점수
    evaluate_task_a,        # Task A: Accuracy / Precision / Recall / F1 / AUC
    evaluate_task_b_ndcg,   # Task B: NDCG@3 / rank_dist
)
```

**threshold sweep** (스크립트 내 정의 허용):

```python
def sweep_threshold(scores, answers_df):
    best_f1, best_thr, best_m = 0.0, 0.5, None
    for thr in np.arange(0.05, 0.96, 0.05):
        m = evaluate_task_a(scores, answers_df, float(thr))
        if m["f1"] > best_f1:
            best_f1, best_thr, best_m = m["f1"], float(thr), m
    return best_f1, best_thr, best_m
```

---

## 9. Git 커밋 규칙

### 커밋 메시지 형식

```
<타입>: <한국어 요약> (~50자)

[선택] 본문 — 변경 이유, 주요 내용 설명
```

### 타입

| 타입 | 용도 |
|------|------|
| `feat` | 새 기능 / 새 모델 추가 |
| `exp` | 실험 스크립트 추가 또는 수정 |
| `fix` | 버그 수정 |
| `refactor` | 동작 변경 없는 코드 정리 |
| `docs` | 문서 변경 (PLAN, REPORT, CONVENTIONS 등) |
| `chore` | .gitignore, 설정 등 기타 |

### 예시

```
feat: BatchMultiInterestModel 추가 (KMeans/SVD/Mean/Diverse)

스트리밍 soft-assignment 대신 배치 클러스터링으로 interest vector 구축.
4가지 방법 구현 및 baseline_eval 대비 성능 비교 실험 포함.

exp: batch_eval.py — 스트리밍 vs 배치 비교 실험 추가

docs: REPORT.md — 가설 검증 분석 (섹션 7) 추가
```

---

## 10. `.gitignore` 관리 원칙

| 제외 대상 | 이유 |
|---------|------|
| `data/` | 대용량 `.npy` / `.csv` 파일 |
| `S26_AI506_Project.pdf` | 배포 제한 과제 원본 |
| `__pycache__/`, `*.pyc` | Python 캐시 |
| `.claude/` | 로컬 AI 도구 설정 |
| `*.json`, `*.log` | 실험 임시 결과물 |

커밋해야 할 결과물(시각화 PNG 등)은 `clustering_results/` 처럼 별도 디렉토리에 모아서 명시적으로 추가한다.
