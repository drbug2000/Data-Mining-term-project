# m04_gated — F1-objective CTR log-linear + 클릭튜닝 content head

기존 `shared/`·`models/` 코드를 건드리지 않고 추가한 모델(gyuchan). 공통 평가
(`shared.eval.predictor`)와 공통 데이터(`shared.data.dataset`)를 그대로 사용한다.

## 결과 (실데이터, click_validation 20k)

| 모델 | Task A F1 | AUC |
|---|---|---|
| **m04_gated (5 CTR + content, F1-objective)** | **0.1014** | **0.6995** |

원본 방법론(AI506 Task1 `5+content`, honest F1 0.1106 / AUC 0.7041)을 이 프레임워크에서
재현. AUC 가 원본과 일치하고 F1 도 ~0.10 으로 재현된다(numpy 재구현 + threshold sweep 차이로 ±0.01).

## 모델

```
s = Σ wᵢ·log(CTRᵢ) + w_c·z(content) + LO·[not logged_on]
    CTRᵢ ∈ {HistCTR, ad_ctr, ip_ctr, dev_ctr, cat_ctr}   # 원본 5개 신뢰 CTR
    content = scale·cos(L2(Uᵀq̂), L2(Vᵀâ)) + b  (+ interest_beta·max cos(user 클릭광고, â))
    지수 wᵢ = ranking-F1 을 직접 최대화하는 좌표상승 (log-loss 아님 — 1번 레버: MLE 0.046 vs F1-fit 0.103)
```

선택: `GateConfig(use_gate=True)` → support 시그모이드 게이트(warm=entity/cold=content). 기본은
위 log-linear(원본 0.1106 재현 경로).

## train/val 분리 (원본 honest 재현의 핵심)

`fit(ds)` 는 `ds.training_stream()` 전체를 받아 **내부에서 SearchID 80/20 split**:
- content head: 내부-train 학습 / 내부-val AUC early-stop,
- F1 지수: 내부-val 에서 적합하되 **내부-val 의 CTR feature 는 내부-train 카운트로만 계산**
  (자기 클릭 leak 방지 — 이게 빠지면 user_ctr 가 leak 로 과적합돼 외부 F1 이 0.06 대로 무너진다).

최종 외부 점수(`score_pairs`)는 full-train 카운트로 feature 생성. `click_validation` 라벨은
최종 F1/AUC 보고에만 쓰고 어떤 선택에도 쓰지 않는다.

> IPID/device 키는 `SearchEvent` 에 없어 `searchinfo.csv`·`userinfo.csv` 를 `ds.dir` 에서
> 직접 읽는다(shared 미수정).

## 구조 / 실행

| 파일 | 역할 |
|---|---|
| `config.py` | `GateConfig` (ModelConfig 스타일 dataclass) |
| `gated_ctr.py` | `GatedCTRModel(BaseRecoModel)` — `fit(ds)` + `score_pairs` + `score_click`/`score_ad_candidates` |
| `experiments/eval_gated.py` | 공통 평가 (Task A F1 / Task B NDCG@3) |
| `__init__.py` | `GateConfig`, `GatedCTRModel` 공개 |

```bash
# 프로젝트 루트에서. datasets 는 ../datasets (기존 컨벤션)
python -X utf8 models/m04_gated/experiments/eval_gated.py
```

```python
from shared.data.dataset import RecoDataset
from shared.eval.predictor import evaluate_task_a
from models.m04_gated import GateConfig, GatedCTRModel

ds = RecoDataset("../datasets").load()
model = GatedCTRModel(GateConfig()).fit(ds)          # 내부 SearchID 80/20 로만 선택
scores = model.score_pairs(ds.val_click_queries())   # rank-정규화 [0,1)
metric = evaluate_task_a(scores, ds.val_click_answers(), threshold=0.5)
```

의존성: `numpy`, `pandas` (torch 불필요 — content head 는 numpy 정규화-cosine bi-encoder + Adam).
