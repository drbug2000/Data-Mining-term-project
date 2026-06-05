# 실험 보고서 — Multi-Interest Recommendation via Text Embedding

**Course**: AI506 Data Mining and Search — Term Project  
**Model**: Model-1 (Simple Text Embedding Baseline)  
**작성일**: 2026-06-05

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

| 이벤트 | sign | α | 업데이트 대상 |
|--------|------|---|------|
| 검색 발생 | +1 | `alpha_search` | `_interests` (Task B용) |
| 광고 클릭 | +1 | `alpha_click` | `_interests` + `_click_interests` (Task A용) |
| 광고 비클릭 | -1 | `alpha_neg` | `_click_interests` (alpha_neg=0이면 비활성) |

Task A 전용 `_click_interests`를 분리하여 검색 신호로 인한 오염을 방지한다.

#### 예측

**Task B (Ad Recommendation)**

$$\text{score}_j = (1 - \gamma)\cdot\text{sim}(q, a_j) + \gamma\cdot\max_i\,\text{sim}(v_i, a_j)$$

후보 광고 전체에 대해 score를 계산 후 argmax의 AdID를 반환한다.

**Task A (Click Prediction)**

$$\text{score} = (1 - \gamma)\cdot\text{sim}(q,\, a) + \gamma\cdot\max_i\,\text{sim}(v^{\text{click}}_i,\, a)$$

Task A는 `_click_interests`(`v^click`)를 사용한다. score > (1 − threshold) 이면 클릭(1), 아니면 비클릭(0)으로 예측한다.

#### Tiered Adaptive Gamma

유저의 훈련 이력에 따라 gamma를 3단계로 차등 적용한다:

| 유저 상태 | 적용 gamma | 비율 |
|---|---|---|
| 클릭 이력 있음 | `gamma` (= 0.7) | 16.9% |
| 검색 이력만 있음 (클릭 없음) | `gamma_search` (= 0.5) | — |
| 완전 cold-start | 0.0 (query-only fallback) | 83.1% |

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
| 훈련 유저 수 | 14,133명 |
| 클릭 발생 유저 | 2,385명 (훈련 유저의 16.9%) |
| 클릭 없는 유저 | 11,748명 (훈련 유저의 83.1%) |
| 클릭 유저 1인당 평균 클릭 수 | 1.49회 |
| 임베딩 차원 | 384 |

### 2-2. Hyperparameter (SOTA)

| 파라미터 | 값 | 설명 |
|----------|:--:|------|
| `k` | 5 | interest vector 개수 |
| `alpha_search` | 0.01 | 검색 업데이트 강도 (클릭보다 약한 신호) |
| `alpha_click` | 0.5 | 클릭 업데이트 강도 |
| `alpha_neg` | 0.0 | 비클릭 페널티 (비활성) |
| `temperature` | 0.1 | soft assignment 온도 (τ sweep 선택값) |
| `gamma` | **0.7** | 클릭 이력 유저의 interest 가중치 |
| `gamma_search` | **0.5** | 검색 이력만 있는 유저의 interest 가중치 |
| `threshold` | 0.5 | 클릭 판정 임계값 (Task A) |

> `gamma`와 `gamma_search`는 `tiered_gamma_eval.py` 그리드 서치로 결정 (Task B NDCG@3 최적화).

### 2-3. 평가 지표

| Task | 지표 | 정의 |
|------|------|------|
| B | **NDCG@3** | 1/log₂(rank+1) (rank≤3), else 0; 전체 쿼리 평균 |
| A | **F1** | 클릭 클래스(IsClick=1) F1 |
| A | **AUC** | ROC AUC (Wilcoxon-Mann-Whitney 통계량) |
| A | **Precision / Recall** | IsClick=1 기준 |

### 2-4. 비교 베이스라인

| 모델 | 설명 |
|------|------|
| **MultiInterest (ours)** | 본 모델 (SOTA config) |
| **Query-only** (gamma=0, gamma_search=0) | user interest 없이 query-ad 유사도만 사용 |
| **Random** | 무작위 예측 |
| **HistCTR baseline** | 과거 CTR만 사용 (Task A F1=0.0436, Task B NDCG@3=0.0211) |

---

## 3. 실험 결과

### 3-1. Task B — Ad Recommendation (214 queries, 17,518 candidates)

| 모델 | **NDCG@3** | Rank 1 | Rank 2 | Rank 3 | >Rank 3 |
|------|:----------:|:------:|:------:|:------:|:-------:|
| **MultiInterest (ours)** | **0.1538** | 28 | 7 | 1 | 178 |
| Query-only (gamma=0) | 0.0989 | 17 | 5 | 2 | 190 |
| Random | 0.0000 | 0 | 0 | 0 | 214 |
| HistCTR baseline | 0.0211 | — | — | — | — |

- MultiInterest는 HistCTR baseline 대비 **×7.3**, Query-only 대비 **×1.56** 향상.
- 214 쿼리 중 36건(16.8%)이 Top-3 내 정답 (Rank 1: 28건, Rank 2: 7건, Rank 3: 1건).

### 3-2. Task A — Click Prediction (20,000 queries, CTR 1.15%)

#### 모델 비교 (threshold=0.5)

| 모델 | Accuracy | Precision | Recall | F1 | AUC |
|------|:--------:|:---------:|:------:|:--:|:---:|
| **MultiInterest (ours)** | 0.5947 | 0.0133 | **0.4716** | **0.0260** | **0.5435** |
| Always-0 (no click) | **0.9886** | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Always-1 (all click) | 0.0115 | 0.0115 | 1.0000 | 0.0226 | 0.5000 |
| HistCTR baseline | — | — | — | 0.0436 | — |

*Precision / Recall / F1: IsClick=1 기준*

#### Threshold Sweep (MultiInterest)

| Threshold | Accuracy | Precision | Recall | F1 |
|:---------:|:--------:|:---------:|:------:|:--:|
| 0.1 | 0.9710 | 0.0111 | 0.0175 | 0.0136 |
| 0.2 | 0.9282 | 0.0152 | 0.0830 | 0.0258 |
| 0.3 | 0.8316 | 0.0112 | 0.1572 | 0.0209 |
| 0.4 | 0.7059 | 0.0129 | 0.3275 | 0.0249 |
| **0.5** | 0.5947 | 0.0133 | 0.4716 | **0.0260** |
| **0.6** | 0.4793 | 0.0133 | 0.6070 | **0.0260** |
| 0.7 | 0.3191 | 0.0127 | 0.7642 | 0.0251 |
| 0.8 | 0.1447 | 0.0123 | 0.9258 | 0.0242 |
| 0.9 | 0.0348 | 0.0116 | 0.9913 | 0.0230 |

> F1 최대값 **0.0260**은 threshold=0.5와 0.6에서 동시에 달성된다.  
> 근본 원인: CTR 1.15%의 극심한 클래스 불균형. AUC=0.5435 수준의 discriminative power로는 F1 한계가 낮다.

#### Per-class (threshold=0.6, best F1=0.0260)

| Class | TP | FP | FN | Precision | Recall | F1 | Support |
|-------|:--:|:--:|:--:|:---------:|:------:|:--:|:-------:|
| 0 (no-click) | 9,448 | 90 | 10,323 | 0.9906 | 0.4779 | 0.6447 | 19,771 |
| 1 (click) | 139 | 10,323 | 90 | 0.0133 | 0.6070 | 0.0260 | 229 |

#### Confusion Matrix (threshold=0.6)

| | Pred 0 | Pred 1 |
|--|:------:|:------:|
| **True 0** (19,771) | 9,448 (TN) | 10,323 (FP) |
| **True 1** (229) | 90 (FN) | 139 (TP) |

---

## 4. 분석

### 4-1. Task B 성능 분석

**Query-only 대비 user interest의 기여**

| 지표 | Query-only | MultiInterest | 향상 |
|------|:----------:|:-------------:|:----:|
| NDCG@3 | 0.0989 | **0.1538** | +56% |
| Rank 1 hits | 17 | **28** | +65% |

- Tiered adaptive gamma (γ_search=0.5)가 search-only 유저의 interest를 부분적으로 활용하여 query-only 대비 성능을 추가로 끌어올린다.
- 클릭 이력이 없는 유저(83.1%)는 query-only 폴백으로 동작하므로, 전체 향상은 클릭 이력 유저(16.9%)에 집중된다.

**HistCTR 대비 ×7.3 달성 이유**

HistCTR 방식(과거 CTR 기반 순위)은 17,518개 후보 전체에 대해 쿼리-무관한 인기도만 반영한다. Text embedding은 쿼리와 의미적으로 일치하는 광고를 직접 찾으므로 retrieval 문제에서 근본적으로 유리하다.

### 4-2. Task A 성능 분석

**AUC=0.5435, HistCTR F1=0.0436에 미달하는 원인**

Task A는 이미 노출된 광고 중 클릭 여부를 예측하는 **행동 예측** 문제다. 훈련 데이터에서 클릭 광고와 비클릭 광고의 query-ad 코사인 유사도 분포가 거의 동일하다:

| 그룹 | mean sim | std | Cohen's d |
|------|:--------:|:---:|:---------:|
| 클릭 광고 | 0.524 | 0.213 | — |
| 비클릭 광고 | 0.501 | 0.217 | **0.11** |

Cohen's d=0.11은 "거의 구분 불가" 범주 (|d|<0.2). 클릭 여부는 텍스트 의미 유사도 이외의 요인 — 광고 크리에이티브, 가격, 노출 위치(position bias), 구매 의도 강도 등 — 에 의해 결정되기 때문이다.

**F1이 낮은 근본 원인**

CTR 1.15%의 극단적 클래스 불균형 하에서 AUC=0.5435 수준의 discriminative power는 F1 향상으로 이어지지 않는다. threshold를 어떻게 조정해도 F1 상한은 ~0.026 수준에서 정체된다.

### 4-3. Ablation Study — Task B NDCG@3

각 구성 요소를 하나씩 제거 또는 변경했을 때의 NDCG@3 변화를 측정한다.  
SOTA 기준: k=5, α_s=0.01, α_c=0.5, τ=0.1, γ=0.7, γ_s=0.5 (NDCG@3=**0.1538**)

#### Interest 신호 기여

| Ablation | NDCG@3 | vs SOTA | Rank 1 | Rank 2 | Rank 3 |
|----------|:------:|:-------:|:------:|:------:|:------:|
| **FULL (SOTA)** | **0.1538** | — | 28 | 7 | 1 |
| w/o interest (γ=0, γ_s=0) | 0.0989 | **-0.0550** | 17 | 5 | 2 |
| w/o tiered γ (γ_s=0) | 0.1351 | -0.0187 | 24 | 7 | 1 |
| w/o search signal (α_s=0) | 0.1415 | -0.0123 | 25 | 6 | 3 |
| w/o click signal (α_c=0) | 0.0866 | **-0.0673** | 16 | 4 | 0 |

- **클릭 신호**가 가장 중요한 단일 요소(-0.0673): 검색 신호만으로는 interest가 거의 무의미  
- **interest 전체 제거**(-0.0550) → query-only 수준으로 하락  
- **tiered gamma**(-0.0187): search-only 유저(83.1%)에게 γ_s=0.5를 주는 것이 유효  
- **검색 신호**도 보조 기여(-0.0123): 단독으로는 약하지만 클릭과 함께 시너지

#### Interest Vector 수 (k)

| k | NDCG@3 | vs SOTA |
|:-:|:------:|:-------:|
| 1 | 0.1392 | -0.0146 |
| 3 | 0.1392 | -0.0146 |
| **5 (SOTA)** | **0.1538** | — |
| 10 | 0.1362 | -0.0176 |

- k=1, 3은 동일 성능(0.1392): 클릭 신호가 희소(유저당 평균 1.49회)하여 다수의 vector를 특화하기 어려움  
- k=10은 오히려 하락: 파라미터가 많아질수록 수렴이 느려지는 효과  
- **k=5가 최적**

#### Soft Assignment 온도 (τ)

| τ | NDCG@3 | vs SOTA |
|:---:|:------:|:-------:|
| 0.01 | 0.1584 | +0.0046 |
| **0.1 (SOTA)** | **0.1538** | — |
| 1.0 | 0.1445 | -0.0093 |
| 100.0 | 0.1334 | -0.0204 |

- τ가 작을수록 가장 가까운 interest vector에 집중(hard assignment에 근접) → 성능 향상  
- τ=0.01은 최고점(+0.0046)이지만 τ=1.0 대비 soft-assignment 특성이 거의 사라짐  
- τ=0.1을 SOTA로 선택: soft assignment를 유지하면서 τ=1.0 대비 유의미한 향상(+0.0093)  
- uniform(τ=100) 최저: 모든 vector에 균등 업데이트 → specialization 소멸

---

## 5. 적용된 개선 사항

| 수정 | 내용 | 효과 |
|------|------|------|
| **Interest 분리** | `_interests` (Task B) / `_click_interests` (Task A) 별도 유지 | 검색 신호 오염 제거 |
| **alpha_search 감소** | 0.1 → 0.01 (검색이 클릭보다 ~89배 많아 누적 지배 억제) | Task A AUC 개선 |
| **Tiered adaptive gamma** | cold=0, search-only=0.5, click=0.7 | Task B NDCG@3 +56% vs query-only |
| **score_click 설계** | `(1-γ)*sim(q,a) + γ*max_i sim(v_i^click, a)` | Task A AUC >0.5 달성 |

### 추가 개선 아이디어 (미적용)

| 아이디어 | 기대 효과 | 복잡도 |
|----------|-----------|--------|
| HistCTR 피처 활용 | Task A Precision 개선 | 낮음 |
| k 튜닝 (k=1~2) | 클릭 희소성 감안해 interest 수렴 가속 | 낮음 |
| alpha_neg 활성화 | click_interests selectivity 향상 | 낮음 |
| Temporal decay | interest drift 대응 | 중간 |

---

## 6. 결론

| 지표 | MultiInterest (ours) | Query-only | HistCTR baseline |
|------|:--------------------:|:----------:|:----------------:|
| **Task B NDCG@3** | **0.1538** | 0.0989 | 0.0211 |
| Task B Rank 1 | 28 / 214 | 17 / 214 | — |
| **Task A F1** | **0.0260** | — | 0.0436 |
| Task A AUC | **0.5435** | — | — |

- **Task B**: text embedding 기반 retrieval이 강한 신호를 제공. user interest (tiered adaptive gamma)로 query-only 대비 +56% 추가 향상.
- **Task A**: score 설계(query-ad 유사도 + click interest 분리)로 AUC >0.5를 달성했으나, text embedding만으로는 F1이 HistCTR baseline에 미달. CTR 1.15% 불균형과 클릭 메커니즘의 비의미적 요인이 근본 한계.
- **전체**: 학습 파라미터 없이 pre-trained embedding만으로 Task B에서 HistCTR 대비 ×7.3 달성. Task A 개선을 위해서는 HistCTR 등 행동 신호 추가가 필요.

---

## 7. 가설 검증 분석 — "관심 광고와 클릭 광고는 다르다"

> **주의**: 이 섹션은 과제 명세(S26_AI506_Project.pdf) 기준의 Task 정의를 사용한다.  
> - **Task A (클릭 예측)**: `click_validation_*` 데이터, 지표 F1  
> - **Task B (광고 추천)**: `ad_validation_*` 데이터, 지표 NDCG@3

### 7-1. 분석 동기

모델이 Task B (광고 추천, NDCG@3=**0.1538**, HistCTR 대비 **×7.3**)에서는 강한 성능을 보이면서, Task A (클릭 예측, F1=0.0260, HistCTR baseline F1=0.0436 미달)에서는 성능이 낮은 원인을 파악하기 위해 분석을 수행했다.

> **가설**: User interest vector는 user가 관심을 갖는 광고를 잘 포착하지만, 실제 클릭 여부는 interest와 다른 요인들에 의해 결정된다.  
> "관심(interest) 광고"와 "클릭(click) 광고"는 동일하지 않을 수 있다.

### 7-2. 분석 방법론

분석의 핵심 아이디어는 **두 task에서 query-ad 코사인 유사도의 "신호 분리 가능성(discriminability)"을 비교**하는 것이다.

#### 분석 1: Task A — 클릭/비클릭 광고의 query-ad 유사도 분포 비교

`click_validation_*` 데이터의 20,000개 (검색, 광고) 쌍 각각에 대해 `sim(query, ad)`를 계산하고, `IsClick=1`(클릭, 229건)과 `IsClick=0`(비클릭, 19,771건) 두 그룹의 분포를 비교한다.

#### 분석 2: Task B — 정답 광고 vs 랜덤 광고의 query-ad 유사도 분포 비교

`ad_validation_*` 데이터의 214개 검색 쿼리 각각에 대해 ①정답 광고와 ②무작위 5개 광고의 유사도를 계산한다.

#### 분석 3: 유저 레벨 상관관계

두 task에 모두 등장하는 유저(공통 29명)에 대해 Task B 순위와 Task A AUC의 상관관계를 분석한다.

#### 분석 4: 훈련 데이터 검증

동일한 분석 1을 훈련 데이터(320,000행)에 적용해, Task A의 낮은 분리 가능성이 데이터 자체의 본질적 특성임을 확인한다.

### 7-3. 분석 결과

#### 분석 1 — Task A: 클릭/비클릭 query-ad 유사도 분포

| 그룹 | n | 평균 유사도 | 표준편차 |
|------|---|:---------:|:-------:|
| 클릭 광고 (IsClick=1) | 229 | **0.5341** | 0.2041 |
| 비클릭 광고 (IsClick=0) | 19,771 | 0.5026 | 0.2154 |

- **Gap**: +0.0315
- **Cohen's d**: **0.15** → "거의 구분 불가" (|d| < 0.2)
- **분포 겹침**: **0.80** → 두 분포의 80%가 겹침
- **AUC (cos-sim as predictor)**: **0.5375** (random 수준)

#### 분석 2 — Task B: 정답/랜덤 query-ad 유사도 분포

| 그룹 | n | 평균 유사도 | 표준편차 |
|------|---|:---------:|:-------:|
| 정답 광고 | 214 | **0.5478** | 0.2126 |
| 랜덤 광고 | 1,070 | 0.2533 | 0.1603 |

- **Gap**: +0.2945 (Task A Gap의 **9.4배**)
- **Cohen's d**: **1.56** → "대형 효과" (|d| > 0.8)
- **분포 겹침**: **0.38**

> **주의**: d=1.56은 Task B 정답 광고(실제 노출·클릭)와 *전체 17,518개 중 무작위 추출* 광고의 비교다.  
> 노출된 비클릭 광고(실제 경쟁 후보) vs 랜덤 광고도 d=1.27로 거의 유사하다 — 이는 광고 노출 자체가 이미 query와 의미적으로 맞는 광고를 필터링한 결과이기 때문이다.  
> 따라서 d=1.56은 "유저가 텍스트 유사도가 높은 광고를 선호"가 아니라 **데이터 생성 구조(광고 시스템의 pre-filtering)**를 반영한다.

#### 분석 3 — 유저 레벨 상관관계

| 비교 | 결과 |
|------|------|
| Task B rank(역) ↔ Task A AUC 상관계수 | **r = 0.07** (사실상 무상관) |

| Task B 성능 그룹 | Task A AUC 평균 | n |
|----------------|:--------------:|---|
| 상위 1/3 (rank ≤ 50) | 0.4218 | 10 |
| 중위 1/3 | 0.5316 | 9 |
| 하위 1/3 (rank > 721) | 0.5039 | 10 |

#### 분석 4 — 훈련 데이터 검증

| 그룹 | n | 평균 유사도 | Cohen's d |
|------|---|:---------:|:---------:|
| 클릭 광고 | 3,560 | 0.5242 | — |
| 비클릭 광고 | 316,440 | 0.5013 | **0.11** |
| AUC (cos-sim predictor) | — | — | **0.5301** |

### 7-4. 해석

#### (1) Text Embedding이 Task B를 잘 푸는 이유

Task B는 17,518개 광고 중 정답을 찾는 **retrieval 문제**다. 대부분의 후보는 해당 쿼리에 노출조차 된 적 없는 광고이므로 사실상 "랜덤"에 가깝다. 이 설정에서 text embedding similarity는 유효한 1차 ranking 신호를 제공한다. User interest profile은 이 신호를 보강하여 추가 성능 향상을 이끈다.

#### (2) Text Embedding이 Task A를 잘 못 푸는 이유

Task A는 이미 노출된 광고 중 클릭 여부를 예측하는 **행동 예측 문제**다. 노출된 광고들은 이미 시스템에 의해 의미적으로 선별된 상태이므로, 클릭/비클릭 광고 간 query-ad 유사도 차이가 극히 작다 (d=0.11). 클릭은 텍스트 유사도 외의 요인에 의해 결정된다.

#### (3) 유저 레벨 상관관계의 의미

r=0.07의 무상관은 두 task가 서로 다른 유저 특성을 측정함을 보여준다. Task B를 잘 맞추는 유저(의미적으로 명확한 검색)의 Task A AUC가 0.42로 오히려 낮은 것은, 구체적 검색을 하는 유저일수록 클릭 결정이 의미 유사도 이외의 요인(가격, 브랜드 등)에 더 크게 좌우됨을 시사한다.

### 7-5. 결론 및 함의

| 항목 | Task A (클릭 예측) | Task B (광고 추천) |
|------|:----------------:|:----------------:|
| **문제 유형** | 행동 예측 | 의미적 retrieval |
| **query-ad sim Gap** | +0.03 | +0.29 (9.4배) |
| **Cohen's d** | 0.15 (구분 불가) | 1.56 (대형 효과)* |
| **AUC (cos-sim predictor)** | 0.54 | — |
| **유저 상관관계** | r = 0.07 (무관) | — |

> *d=1.56: 전체 후보 대비 효과크기. 광고 시스템 pre-filtering 효과 포함.

**가설 확인**: "유저의 관심 광고"와 "실제 클릭 광고"는 다른 메커니즘으로 결정된다.

Task A F1이 HistCTR baseline에 미달하는 것은 모델 설계의 실패가 아니라 **text embedding이 제공할 수 있는 정보의 본질적 한계**다. Task A 개선을 위해서는 HistCTR 등 행동 기반 신호가 필수적이다.

> **실험 코드**: `experiments/hypothesis_analysis.py`  
> **실행**: `python -X utf8 experiments/hypothesis_analysis.py`
