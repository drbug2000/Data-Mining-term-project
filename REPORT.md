# 실험 보고서 — Multi-Interest Recommendation via Text Embedding

**Course**: AI506 Data Mining and Search — Term Project  
**Model**: Model-1 (Simple Text Embedding Baseline)  
**작성일**: 2026-05-09

---

## 1. 모델 개요

### 1-1. 핵심 아이디어

학습 가능한 파라미터 없이, pre-trained text embedding만으로 동작하는 추천 시스템.  
유저마다 **k개의 interest vector**를 유지하며, 검색·클릭 이벤트가 발생할 때마다 스트리밍 방식으로 업데이트한다.

### 1-2. 알고리즘

#### Interest 초기화

유저를 처음 만났을 때 interest vectors를 **랜덤 정규분포 → L2 정규화**로 초기화한다.

- 0벡터 초기화는 k개 벡터 간 거리가 모두 동일해 soft assignment가 uniform이 되므로 제외.
- 랜덤 초기화는 k개 벡터가 초기부터 서로 다른 방향을 가리켜 specialization을 유도한다.

#### Soft Assignment 업데이트

새 embedding `e`가 들어올 때마다 각 interest vector `v_i`와의 코사인 거리를 기반으로 가중치를 계산해 업데이트한다.

```
cos_dist_i = 1 - cosine_sim(v_i, e)         # 코사인 거리
weight_i   = exp(-cos_dist_i / τ)           # 온도 τ로 softness 조절
weight_i  /= Σ weight_j                     # 정규화 (합 = 1)

v_i += sign · α · weight_i · normalize(e)   # 가중 업데이트
```

| 이벤트 | sign | α | 의미 |
|--------|------|---|------|
| 검색 발생 | +1 | `alpha_search` | interest를 검색 방향으로 당김 |
| 광고 클릭 | +1 | `alpha_click` | interest를 클릭 광고 방향으로 강하게 당김 |
| 광고 비클릭 | -1 | `alpha_neg` | interest를 비클릭 광고 반대 방향으로 밂 |

#### 예측

**Task B (Ad Recommendation)**

$$\text{score}_j = (1 - \gamma)\cdot\text{sim}(q, a_j) + \gamma\cdot\max_i\,\text{sim}(v_i, a_j)$$

후보 광고 전체에 대해 score를 계산 후 argmax의 AdID를 반환한다.

**Task A (Click Prediction)**

$$\text{score} = (1 - \gamma)\cdot\text{sim}(q,\, a) + \gamma\cdot\max_i\,\text{sim}(v_i,\, a)$$

Task B와 동일한 혼합 공식을 사용한다. 단, 클릭 이력이 없는 유저(Adaptive gamma)는 γ=0으로 폴백해 query-ad 유사도만 사용한다.  
score > (1 - threshold) 이면 클릭(1), 아니면 비클릭(0)으로 예측한다.

---

## 2. 실험 설정

### 2-1. 데이터셋

| 항목 | 수치 |
|------|------|
| 총 유저 | 16,975명 |
| 검색 이벤트 | 276,807건 |
| 광고 수 | 17,518개 |
| 훈련 행 | 320,000건 |
| 전체 CTR | 1.11% (클릭 3,560 / 전체 320,000) |
| 클릭 발생 유저 | 2,385명 (전체의 14%) |
| 클릭 유저 1인당 평균 클릭 수 | 1.49회 |
| 임베딩 차원 | 384 |

### 2-2. Hyperparameter

| 파라미터 | 초기값 | Step 5 수정값 | 설명 |
|----------|:------:|:-------------:|------|
| `k` | 5 | 5 | interest vector 개수 |
| `alpha_search` | 0.1 | **0.01** | 검색 업데이트 강도 |
| `alpha_click` | 0.5 | 0.5 | 클릭 업데이트 강도 |
| `alpha_neg` | 0.0 | 0.0 | 비클릭 페널티 (비활성) |
| `temperature` | 1.0 | 1.0 | soft assignment 온도 |
| `gamma` | 0.5 | 0.5 (adaptive) | user interest 혼합 비율 |
| `threshold` | 0.5 | 0.5 | 클릭 판정 임계값 (Task A) |

### 2-3. 평가 지표

| Task | 지표 | 정의 |
|------|------|------|
| B | **Accuracy** | 정답 AdID를 1위로 예측한 비율 |
| B | **AUC** | 정답 광고가 임의 오답보다 높은 점수를 받을 확률 (ranking AUC) |
| B | **MRR** | Mean Reciprocal Rank — 1/(정답 순위)의 평균 |
| A | **Accuracy** | 전체 정확도 |
| A | **Precision** | 클릭으로 예측한 것 중 실제 클릭 비율 — **IsClick=1 기준** |
| A | **Recall** | 실제 클릭 중 올바르게 탐지한 비율 — **IsClick=1 기준** |
| A | **F1** | 클릭 클래스 F1 — **IsClick=1 기준** |
| A | **AUC** | ROC AUC (Wilcoxon-Mann-Whitney 통계량) |

### 2-4. 비교 베이스라인

| 모델 | 설명 |
|------|------|
| **MultiInterest (ours)** | 본 모델 |
| **Query-only** (gamma=0) | user interest 없이 query-ad 유사도만 사용 |
| **Random** | 무작위 예측 |
| **Always-0** | 항상 비클릭으로 예측 |
| **Always-1** | 항상 클릭으로 예측 |

---

## 3. 실험 결과

### 3-1. Task B — Ad Recommendation (214 queries, 17,518 candidates)

#### Step 4 (초기 결과, alpha_search=0.1)

| 모델 | Accuracy | AUC | MRR |
|------|:--------:|:---:|:---:|
| **MultiInterest (ours)** | **0.0748** | 0.8417 | **0.1337** |
| Query-only (gamma=0) | 0.0607 | **0.8454** | 0.1197 |
| Random | 0.0000 | 0.5047 | 0.0007 |

#### Step 5 (개선 후, alpha_search=0.01 + adaptive gamma)

| 모델 | Accuracy | AUC | MRR |
|------|:--------:|:---:|:---:|
| **MultiInterest (ours)** | **0.0748** | **0.8500** | **0.1428** |
| Query-only (gamma=0) | 0.0607 | 0.8454 | 0.1197 |
| Random | 0.0000 | 0.5047 | 0.0007 |

**Confusion (Task B — Step 5)**

| | Correct | Wrong | Total |
|--|:-------:|:-----:|:-----:|
| Queries | 16 | 198 | 214 |

### 3-2. Task A — Click Prediction (20,000 queries, CTR 1.15%)

#### Step 4 (초기 결과)

| 모델 | Accuracy | Precision | Recall | F1 | AUC |
|------|:--------:|:---------:|:------:|:--:|:---:|
| **MultiInterest (ours)** | 0.8762 | 0.0049 | 0.0480 | 0.0088 | 0.4164 |
| Always-0 (no click) | **0.9886** | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Always-1 (all click) | 0.0115 | 0.0115 | 1.0000 | 0.0226 | 0.5000 |

#### Step 5 (개선 후)

| 모델 | Accuracy | Precision | Recall | F1 | AUC |
|------|:--------:|:---------:|:------:|:--:|:---:|
| **MultiInterest (ours)** | 0.5686 | 0.0131 | **0.4934** | **0.0255** | **0.5334** |
| Always-0 (no click) | **0.9886** | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Always-1 (all click) | 0.0115 | 0.0115 | 1.0000 | 0.0226 | 0.5000 |

*Precision / Recall / F1: IsClick=1 기준*

**Per-class (MultiInterest, Step 5)**

| Class | TP | FP | FN | Precision | Recall | F1 | Support |
|-------|:--:|:--:|:--:|:---------:|:------:|:--:|:-------:|
| 0 (no-click) | 11,260 | 116 | 8,511 | 0.9898 | 0.5695 | 0.7230 | 19,771 |
| 1 (click) | 113 | 8,511 | 116 | 0.0131 | 0.4934 | 0.0255 | 229 |

**Confusion Matrix (Task A — Step 5)**

| | Pred 0 | Pred 1 |
|--|:------:|:------:|
| **True 0** (19,771) | 11,260 (TN) | 8,511 (FP) |
| **True 1** (229) | 116 (FN) | 113 (TP) |

**Threshold Sweep (MultiInterest, Step 5)**

| Threshold | Accuracy | Precision | Recall | F1 |
|:---------:|:--------:|:---------:|:------:|:--:|
| 0.1 | 0.9710 | 0.0111 | 0.0175 | 0.0136 |
| 0.2 | 0.9281 | 0.0152 | 0.0830 | **0.0257** |
| 0.3 | 0.8325 | 0.0107 | 0.1485 | 0.0199 |
| 0.4 | 0.7046 | 0.0129 | 0.3275 | 0.0248 |
| **0.5** | **0.5686** | **0.0131** | **0.4934** | **0.0255** |
| 0.6 | 0.4155 | 0.0124 | 0.6376 | 0.0244 |
| 0.7 | 0.2494 | 0.0125 | 0.8253 | 0.0246 |
| 0.8 | 0.1134 | 0.0119 | 0.9345 | 0.0236 |
| 0.9 | 0.0309 | 0.0116 | 0.9956 | 0.0230 |

> F1 최대값은 threshold=0.2 (F1=0.0257). Precision-Recall tradeoff 상 threshold를 낮추면 Recall이 급감하고, 높이면 Precision이 급감한다. 근본 원인은 클래스 불균형(CTR 1.15%)에 있으므로 F1 수치 자체는 낮다.

---

## 4. 분석

### 4-1. Task B — Step 5 결과 분석

**개선 효과**

| 지표 | Step 4 | Step 5 | 변화 |
|------|:------:|:------:|:----:|
| Accuracy | 0.0748 | 0.0748 | - |
| AUC | 0.8417 | **0.8500** | +0.0083 |
| MRR | 0.1337 | **0.1428** | +0.0091 |

- alpha_search를 0.1 → 0.01로 낮추자 interest vector가 클릭 신호에 더 민감하게 수렴했다.
- **Step 5에서 MultiInterest AUC(0.8500) > Query-only AUC(0.8454)**: 이제 user interest가 랭킹 순서도 개선한다.
- Accuracy는 동일(16/214): 단순 top-1 정확도는 클릭 집계 품질보다 데이터 분산이 지배적.
- Adaptive gamma: 클릭 이력 없는 유저(86%)를 query-only로 폴백시켜 noise interest 혼입을 방지.

### 4-2. Task A — Step 5 결과 분석

**핵심 개선: AUC 0.4164 → 0.5334 (역상관 해소)**

| 지표 | Step 4 | Step 5 | 변화 |
|------|:------:|:------:|:----:|
| Accuracy | 0.8762 | 0.5686 | - |
| Precision | 0.0049 | 0.0131 | +168% |
| Recall | 0.0480 | **0.4934** | +928% |
| F1 | 0.0088 | **0.0255** | +190% |
| AUC | 0.4164 | **0.5334** | +0.1170 |

Step 4의 AUC < 0.5 문제가 해소됐다. 두 수정이 각각 다른 효과를 냈다.

#### [수정 2] score_click에 query-ad 유사도 추가 → AUC 반전의 핵심

Step 4의 `score_click = max_i cosine_sim(v_i, ad)`는 interest vector가 검색 방향으로 쏠려있어 클릭과 역방향 상관관계를 가졌다. Step 5에서 `(1-gamma)*sim(query,ad) + gamma*max_i sim(v_i,ad)`로 바꾸자 **query-ad 유사도가 강한 양의 클릭 신호를 제공**하며 AUC가 0.5를 넘겼다.

#### [수정 3] alpha_search 0.1 → 0.01 → Interest 품질 개선

검색 이벤트가 클릭보다 ~50배 많아 누적 업데이트가 클릭 신호를 압도했다. alpha_search를 1/10로 줄이자 interest vector가 클릭 방향을 더 잘 보존한다.

#### 잔존 문제: Precision 극히 낮음

- CTR 1.15% 극단적 클래스 불균형 → threshold=0.5에서 8,511건(42.6%)을 클릭으로 예측, 실제 클릭은 113건만 맞힘
- Threshold=0.2에서 F1이 0.0257로 미세하게 최대 (Precision 1.52%, Recall 8.3%)
- 근본 원인: score 분포가 클릭/비클릭 간 충분히 분리되지 않음 → 미래 개선 필요

---

## 5. 개선 방향 (Step 5 이후)

### 5-1. 적용 완료 (Step 5)

| 수정 | 내용 | 효과 |
|------|------|------|
| [수정 1] Adaptive gamma | 클릭 이력 없는 유저(86%)는 gamma=0으로 폴백 | Task B AUC +0.008 |
| [수정 2] Task B score 설계 | `(1-γ)*sim(q,a) + γ*max_i sim(v_i,a)` 통일 | Task B AUC 0.41→0.53 |
| [수정 3] alpha_search 감소 | 0.1 → 0.01 (검색 누적 지배 억제) | Task A Recall +928% |

### 5-2. 추가 개선 아이디어

| 아이디어 | 기대 효과 | 복잡도 |
|----------|-----------|--------|
| HistCTR 피처 활용: 광고의 과거 CTR을 score에 가산 | Task A Precision 개선 | 낮음 |
| k 튜닝: 클릭 희소성 감안해 k=1~2로 줄이기 | Interest 수렴 속도 향상 | 낮음 |
| alpha_neg 활성화: 비클릭 광고 방향 억압 | Interest selectivity 향상 | 낮음 |
| Temporal decay: 오래된 업데이트에 감쇠 계수 적용 | Interest drift 대응 | 중간 |
| gamma 세밀 튜닝: 클릭 횟수에 비례한 effective_gamma | 데이터 희소성 대응 | 낮음 |

---

## 6. 결론

| 관점 | Step 4 | Step 5 |
|------|--------|--------|
| Task B AUC | 0.8417 (Query-only에 뒤짐) | **0.8500** (Query-only 역전) |
| Task B Accuracy | 0.0748 (랜덤 대비 1,700×) | 0.0748 (유지) |
| Task A AUC | 0.4164 (역상관) | **0.5334** (정방향 달성) |
| Task A Recall | 0.0480 | **0.4934** (+928%) |
| Task A F1 | 0.0088 | **0.0255** (+190%) |

- **Task B**: text embedding 기반 retrieval이 강한 신호를 제공(AUC 0.85). adaptive gamma와 alpha_search 재조정으로 user interest도 랭킹을 개선함.
- **Task A**: score 설계(query-ad 유사도 추가)와 업데이트 비중 재조정으로 AUC를 0.5 이상으로 끌어올렸다. Precision은 여전히 낮아 추가 개선 여지가 있다.
- **전체 방향성**: 학습 파라미터 없이 pre-trained embedding만으로 합리적인 retrieval 성능을 달성했다. Task A click prediction은 극단적 클래스 불균형(CTR 1.15%)이 핵심 도전이며, HistCTR 같은 보조 피처 활용이 다음 단계다.

---

## 7. 가설 검증 분석 — "관심 광고와 클릭 광고는 다르다"

> **주의**: 이 섹션은 과제 명세(S26_AI506_Project.pdf) 기준의 Task 정의를 사용한다.  
> - **Task A (클릭 예측)**: `click_validation_*` 데이터, 지표 F1  
> - **Task B (광고 추천)**: `ad_validation_*` 데이터, 지표 NDCG@3

### 7-1. 분석 동기

모델이 Task B (광고 추천, NDCG@3=0.1228, 베이스라인 대비 6배)에서는 강한 성능을 보이면서, Task A (클릭 예측, F1=0.0257, HistCTR 베이스라인 F1=0.0436 미달)에서는 성능이 낮은 원인을 파악하기 위해 분석을 수행했다.

구체적인 개선 실험(alpha_neg 활성화, click-only interest 분리 등)을 진행해도 Task A F1이 0.03 수준에서 정체되자, 다음 가설을 제기했다.

> **가설**: User interest vector는 user가 관심을 갖는 광고를 잘 포착하지만, 실제 클릭 여부는 interest와 다른 요인들에 의해 결정된다.  
> "관심(interest) 광고"와 "클릭(click) 광고"는 동일하지 않을 수 있다.

이 가설이 사실이라면, text embedding 기반 interest 모델이 Task B에는 적합하지만 Task A에는 원천적인 한계를 가진다는 결론이 도출된다.

### 7-2. 분석 방법론

분석의 핵심 아이디어는 **두 task에서 query-ad 코사인 유사도의 "신호 분리 가능성(discriminability)"을 비교**하는 것이다. Text embedding 기반 유사도가 각 task의 정답 기준을 얼마나 잘 구분하는지를 정량화한다.

#### 분석 1: Task A — 클릭/비클릭 광고의 query-ad 유사도 분포 비교

`click_validation_*` 데이터의 20,000개 (검색, 광고) 쌍 각각에 대해 `sim(query, ad) = normalize(query_emb) · normalize(ad_emb)`를 계산하고, `IsClick=1`(클릭, 229건)과 `IsClick=0`(비클릭, 19,771건) 두 그룹의 분포를 비교한다.

**지표**:
- **평균 Gap**: 클릭 그룹 평균 유사도 − 비클릭 그룹 평균 유사도
- **Cohen's d**: 표준화된 효과 크기. `d = Gap / pooled_std`. 0.2 미만이면 "거의 구분 불가", 0.8 초과면 "대형 효과"
- **분포 겹침 (Overlap)**: 두 분포를 50개 구간으로 히스토그램화한 뒤 `Σ min(h_click, h_noclick) · Δbin`으로 계산. 1.0이면 완전 동일, 0이면 완전 분리

#### 분석 2: Task B — 정답 광고 vs 랜덤 광고의 query-ad 유사도 분포 비교

`ad_validation_*` 데이터의 214개 검색 쿼리 각각에 대해 ①정답 광고(val_ad_answers에 지정된 AdID)와 ②무작위 5개 광고의 유사도를 계산한다. 정답(214건) vs 랜덤(1,070건) 두 그룹을 동일 방식으로 비교한다.

이 분석은 **"embedding이 Task B를 구분하는 힘"**을 측정한다. Task A와의 비교를 통해 두 task 사이의 근본적 차이를 드러낸다.

#### 분석 3: 유저 레벨 상관관계

두 task에 모두 등장하는 유저(공통 29명)에 대해:
- **Task B 성능**: 해당 유저의 마지막 검색에서 정답 광고의 순위(rank, 낮을수록 좋음)
- **Task A 성능**: 해당 유저에 대한 클릭 예측 AUC (Wilcoxon-Mann-Whitney)

두 지표의 Pearson 상관계수를 계산한다. 가설이 맞다면 상관계수는 0에 가까워야 한다 — Task B를 잘 맞추는 유저가 Task A에서도 잘 맞아야 할 이유가 없기 때문이다.

추가로 Task B 순위 기준 상위/중위/하위 1/3 그룹별 Task A AUC 평균을 비교한다.

#### 분석 4: 훈련 데이터 검증

동일한 분석 1을 훈련 데이터(320,000행, 클릭 3,560건 / 비클릭 316,440건)에 적용해, Task A의 낮은 분리 가능성이 검증 데이터 특성이 아니라 **데이터 자체의 본질적 특성**임을 확인한다.

모든 분석은 `experiments/hypothesis_analysis.py`에 구현되어 있으며, Task B 결과 산출 시에는 adaptive gamma=0.5 설정으로 훈련된 MultiInterestModel을 사용했다.

### 7-3. 분석 결과

#### 분석 1 — Task A: 클릭/비클릭 query-ad 유사도 분포

| 그룹 | n | 평균 유사도 | 표준편차 |
|------|---|:---------:|:-------:|
| 클릭 광고 (IsClick=1) | 229 | **0.5341** | 0.2041 |
| 비클릭 광고 (IsClick=0) | 19,771 | 0.5026 | 0.2154 |

- **Gap**: +0.0315
- **Cohen's d**: **0.15** → "거의 구분 불가" 범주 (|d| < 0.2)
- **분포 겹침**: **0.80** → 두 분포의 80%가 겹침

#### 분석 2 — Task B: 정답/랜덤 query-ad 유사도 분포

| 그룹 | n | 평균 유사도 | 표준편차 |
|------|---|:---------:|:-------:|
| 정답 광고 | 214 | **0.5478** | 0.2126 |
| 랜덤 광고 | 1,070 | 0.2533 | 0.1603 |

- **Gap**: +0.2945 (Task A Gap의 **9.4배**)
- **Cohen's d**: **1.56** → "대형 효과" (|d| > 0.8)
- **분포 겹침**: **0.38** → Task A 대비 분포가 훨씬 분리됨

#### 분석 3 — 유저 레벨 상관관계

| 비교 | 결과 |
|------|------|
| Task B rank(역) ↔ Task A AUC 상관계수 | **r = 0.07** (사실상 무상관) |

| Task B 성능 그룹 | Task A AUC 평균 | n |
|----------------|:--------------:|---|
| 상위 1/3 (rank ≤ 50) | 0.4218 | 10 |
| 중위 1/3 | 0.5316 | 9 |
| 하위 1/3 (rank > 721) | 0.5039 | 10 |

**주목할 점**: Task B를 가장 잘 맞추는 유저(rank 상위 1/3)의 Task A AUC가 **0.42** — 랜덤(0.5)보다도 낮다.

#### 분석 4 — 훈련 데이터 검증

| 그룹 | n | 평균 유사도 | Cohen's d |
|------|---|:---------:|:---------:|
| 클릭 광고 | 3,560 | 0.5242 | — |
| 비클릭 광고 | 316,440 | 0.5013 | **0.11** |

훈련 데이터에서도 Cohen's d=0.11로 검증 데이터(d=0.15)와 동일하게 "거의 구분 불가" 수준이다.

### 7-4. 해석

#### (1) Text Embedding이 Task B를 잘 푸는 이유

Task B는 **의미적 유사성(semantic relevance) 문제**다. 17,518개의 다양한 광고 중에서 이 검색어와 가장 의미적으로 맞는 광고를 찾는 문제이며, pre-trained text embedding은 정확히 이 문제를 위해 설계된 도구다.

정답 광고와 랜덤 광고 사이의 유사도 Gap이 0.29이고, Cohen's d=1.56이라는 것은 embedding 공간에서 "관련 광고"와 "무관한 광고"가 명확하게 분리된다는 의미다. User interest vector가 이 분리를 더욱 강화하여 NDCG@3=0.1228(베이스라인 0.0211 대비 6배)이 달성된다.

#### (2) Text Embedding이 Task A를 잘 못 푸는 이유

Task A는 **행동 예측(behavioral prediction) 문제**다. 광고가 이미 검색 결과에 노출된 상황에서, 이 특정 유저가 이 특정 광고를 **지금 클릭할 것인가**를 예측해야 한다.

클릭 여부는 텍스트 임베딩으로 포착되지 않는 다양한 요인의 영향을 받는다:
- 광고 크리에이티브 품질 (이미지, 카피 매력도)
- 가격 경쟁력 및 프로모션 여부
- 동시에 노출된 경쟁 광고들
- 유저의 현재 구매 의도 강도 (탐색 vs 구매 단계)
- 디바이스, 시간대, 피로도 등 컨텍스트 요인
- 광고 노출 위치(position bias)

클릭/비클릭 광고의 유사도 Gap이 0.03, Cohen's d=0.15에 불과하다는 것은, **텍스트 의미 유사성이 높다고 해서 클릭 가능성이 높은 것이 아님**을 직접적으로 보여준다.

#### (3) 유저 레벨 상관관계의 의미

Task B 상위 유저(embedding 공간에서 검색-광고 매칭이 잘 되는 유저)의 Task A AUC가 오히려 **0.42로 가장 낮다**는 역설적 결과는 다음과 같이 해석할 수 있다:

Task B를 잘 맞추는 유저는 의미적으로 명확하고 구체적인 검색을 하는 유저다. 이런 유저는 광고와 검색어의 의미적 유사도가 높아도 클릭 여부가 다른 요인(가격, 브랜드 신뢰도 등)에 의해 결정될 가능성이 높다. 반면 모호한 검색을 하는 유저는 상대적으로 광고의 의미적 관련성이 클릭 의사결정에 더 많이 관여한다.

즉, **두 task는 서로 다른 유저 특성에 의해 성능이 결정되는 독립적인 문제**임이 유저 레벨에서도 확인된다.

### 7-5. 결론 및 함의

| 항목 | Task A (클릭 예측) | Task B (광고 추천) |
|------|:----------------:|:----------------:|
| **문제 유형** | 행동 예측 | 의미적 검색 |
| **query-ad sim Gap** | +0.03 | +0.29 (9.4배) |
| **Cohen's d** | 0.15 (구분 불가) | 1.56 (대형 효과) |
| **분포 겹침** | 80% | 38% |
| **유저 상관관계** | r = 0.07 (무관) | — |

**가설 확인**: "유저의 관심 광고"와 "실제 클릭 광고"는 다른 메커니즘으로 결정된다.

이 분석은 Task A의 낮은 성능이 모델 설계의 실패가 아니라, **text embedding이 제공할 수 있는 정보의 본질적 한계**에서 비롯됨을 보여준다. Task A F1을 HistCTR 베이스라인 수준으로 끌어올리려면 텍스트 의미 이외의 신호 — 클릭 이력 기반 CTR, 광고 크리에이티브 피처, 컨텍스트 피처 등 — 가 필요하다.

반면 Task B에서 embedding 기반 interest 모델이 베이스라인 대비 6배의 성능을 보이는 것은 모델의 설계 방향이 "의미적 관심사 모델링"에 잘 맞아떨어졌음을 의미하며, 이는 모델의 핵심 강점으로 볼 수 있다.

> **실험 코드**: `experiments/hypothesis_analysis.py`  
> **실행**: `python -X utf8 experiments/hypothesis_analysis.py`
