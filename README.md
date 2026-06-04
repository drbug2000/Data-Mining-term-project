# AI506 Term Project — Ad Recommendation System

> **Course**: AI506 Data Mining and Search  
> **Task**: Ad recommendation (Task B: NDCG@3) + Click prediction (Task A: F1)

---

## 프로젝트 구조

```
project-root/
│
├── shared/                     # ★ 공유 파이프라인 (고정 — 직접 수정 자제)
│   ├── base.py                 #   BaseRecoModel 추상 인터페이스
│   ├── data/                   #   앞단: 데이터 로딩
│   │   ├── dataset.py          #     RecoDataset — CSV·NPY 파싱, 스트림 제공
│   │   └── graph.py            #     HeteroGraph, build_graph — 이종 그래프
│   └── eval/                   #   뒷단: 성능 측정
│       └── predictor.py        #     train / score_* / evaluate_* / predict_*
│
├── models/                     # ★ 독립 모델 디렉토리 (각자 개발)
│   ├── m01_interest/           #   Text Embedding Interest Model
│   │   ├── config.py           #     ModelConfig (하이퍼파라미터)
│   │   ├── interest.py         #     MultiInterestModel (스트리밍 soft-assignment)
│   │   ├── batch_interest.py   #     BatchMultiInterestModel (KMeans/SVD/Mean/Diverse)
│   │   └── experiments/        #     모델 전용 실험 스크립트
│   │       ├── baseline_eval.py
│   │       ├── batch_eval.py
│   │       └── ...
│   │
│   ├── m02_gnn/                #   Graph Neural Network Model
│   │   ├── gnn.py              #     GNNConfig, GNNModel
│   │   └── experiments/
│   │       └── gnn_eval.py
│   │
│   └── m03_ctr_mlp/            #   CTR MLP (GNN repr + 통계 피처)
│       ├── ctr_mlp.py          #     CTRConfig, CTRPredictor
│       └── experiments/
│           ├── ctr_eval.py
│           └── ...
│
├── analysis/                   # 데이터·임베딩 범용 분석 (모델 독립)
│   ├── embedding_clustering.py
│   └── cluster_category_match.py
│
├── clustering_results/         # 시각화 결과 PNG
├── CONVENTIONS.md              # 코딩·설계 규칙
├── PLAN.md                     # 설계 문서
├── REPORT.md                   # 실험 보고서
└── EXPERIMENT_LOG.md           # 실험 이력
```

---

## 설계 철학

```
[shared/data/]  →  [models/mXX/]  →  [shared/eval/]
  데이터 로딩        모델 로직          성능 측정
  (공유, 고정)      (독립 개발)        (공유, 고정)
```

- **앞단(shared/data)**과 **뒷단(shared/eval)**은 고정 — 모든 모델이 동일한 데이터·평가 파이프라인을 사용한다.  
- **각 모델(models/mXX)**은 완전히 독립적으로 개발·비교할 수 있다.  
- 새 모델을 추가해도 기존 모델 코드와 평가 코드는 변경할 필요 없다.

---

## 빠른 시작

```bash
# 저장소 클론
git clone https://github.com/drbug2000/Data-Mining-term-project.git
cd Data-Mining-term-project

# 데이터셋을 ../datasets/ 에 위치 (프로젝트 루트 한 단계 위)
# datasets/
#   searchinfo.csv, adinfo.csv, userinfo.csv
#   searchinfo_text_embs.npy, adinfo_title_embs.npy
#   search_stream_training.csv
#   ad_validation_query.csv, ad_validation_answer.csv
#   click_validation_query.csv, click_validation_answer.csv

# 기본 평가 실행 (m01_interest)
python -X utf8 models/m01_interest/experiments/baseline_eval.py

# 배치 방식 비교
python -X utf8 models/m01_interest/experiments/batch_eval.py

# GNN 평가
python -X utf8 models/m02_gnn/experiments/gnn_eval.py

# CTR MLP 평가
python -X utf8 models/m03_ctr_mlp/experiments/ctr_eval.py
```

---

## 새 모델 추가하기

### 1. 디렉토리 생성

```
models/m04_your_model/
├── __init__.py
├── config.py          # YourConfig dataclass
├── your_model.py      # YourModel(BaseRecoModel)
└── experiments/
    └── your_eval.py
```

### 2. `BaseRecoModel` 상속

```python
# models/m04_your_model/your_model.py
from __future__ import annotations
import numpy as np
from shared.base import BaseRecoModel
from models.m04_your_model.config import YourConfig


class YourModel(BaseRecoModel):

    def __init__(self, config: YourConfig):
        super().__init__(config)

    # 필수 구현 (4개)
    def update_search(self, user_id: int, search_emb: np.ndarray) -> None: ...
    def update_click(self, user_id: int, ad_emb: np.ndarray, clicked: bool) -> None: ...
    def predict_ad(self, user_id, query_emb, candidate_embs, candidate_ids) -> int: ...
    def predict_click(self, user_id, query_emb, ad_emb) -> int: ...

    # 평가용 (권장)
    def score_ad_candidates(self, user_id, query_emb, candidate_embs) -> np.ndarray: ...
    def score_click(self, user_id, query_emb, ad_emb) -> float: ...
```

### 3. `__init__.py` 등록

```python
# models/m04_your_model/__init__.py
from models.m04_your_model.config import YourConfig
from models.m04_your_model.your_model import YourModel
```

### 4. 실험 스크립트 작성

```python
# models/m04_your_model/experiments/your_eval.py
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent   # 프로젝트 루트
sys.path.insert(0, str(ROOT))

from shared.data.dataset import RecoDataset
from shared.eval.predictor import train, score_task_a, score_task_b
from shared.eval.predictor import evaluate_task_a, evaluate_task_b_ndcg
from models.m04_your_model import YourConfig, YourModel

DATASET_DIR = ROOT / "../datasets"
SEED = 42

def main():
    np.random.seed(SEED)
    ds = RecoDataset(DATASET_DIR).load()
    candidate_embs, candidate_ids = ds.all_ad_embs()

    model = YourModel(YourConfig())
    train(model, ds.training_stream())

    # Task A
    sc_a = score_task_a(model, ds.val_click_queries())
    m_a  = evaluate_task_a(sc_a, ds.val_click_answers(), threshold=0.5)
    print(f"Task A — F1={m_a['f1']:.4f}  AUC={m_a['auc']:.4f}")

    # Task B
    sc_b = score_task_b(model, ds.val_ad_queries(), candidate_embs)
    m_b  = evaluate_task_b_ndcg(sc_b, ds.val_ad_answers(), candidate_ids)
    print(f"Task B — NDCG@3={m_b['ndcg@3']:.4f}")

if __name__ == "__main__":
    main()
```

---

## import 경로 참조표

| 구 경로 | 새 경로 |
|---------|---------|
| `from data.dataset import RecoDataset` | `from shared.data.dataset import RecoDataset` |
| `from data.graph import build_graph` | `from shared.data.graph import build_graph` |
| `from model.base import BaseRecoModel` | `from shared.base import BaseRecoModel` |
| `from model.predictor import train, ...` | `from shared.eval.predictor import train, ...` |
| `from model import ModelConfig, MultiInterestModel` | `from models.m01_interest import ModelConfig, MultiInterestModel` |
| `from model.batch_interest import BatchMultiInterestModel` | `from models.m01_interest import BatchMultiInterestModel` |
| `from model.gnn import GNNConfig, GNNModel` | `from models.m02_gnn import GNNConfig, GNNModel` |
| `from model.ctr_mlp import CTRConfig, CTRPredictor` | `from models.m03_ctr_mlp import CTRConfig, CTRPredictor` |

`ROOT` 경로 (실험 스크립트):

| 위치 | ROOT 정의 |
|------|-----------|
| `models/mXX/experiments/*.py` | `Path(__file__).parent.parent.parent.parent` |
| `analysis/*.py` | `Path(__file__).parent.parent` |

---

## 현재 모델 성능 (validation 기준)

| 모델 | Task A F1 | Task A AUC | Task B NDCG@3 | Task B MRR |
|------|:---------:|:----------:|:-------------:|:----------:|
| m01 MultiInterest (streaming) | 0.0335 | 0.5576 | 0.1445 | 0.1582 |
| m01 Batch-KMeans | 0.0322 | 0.5695 | 0.1246 | 0.1344 |
| m01 Batch-SVD | 0.0331 | 0.5706 | 0.0936 | 0.1086 |
| m02 GNN | — | — | — | — |
| m03 CTR MLP | — | — | — | — |
| Query-only baseline | 0.0250 | 0.5375 | 0.0989 | 0.1197 |
| HistCTR baseline | 0.0436 | — | 0.0211 | — |
