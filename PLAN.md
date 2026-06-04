# Multi-Interest Recommendation via Text Embedding
**Course**: AI506 Data Mining and Search — Term Project  
**Model**: Model-1 (Simple Text Embedding baseline)  
**Last updated**: 2026-05-09

---

## 1. 목표

검색 기록과 광고 클릭 기록만으로, 학습 가능한 파라미터 없이 동작하는 경량 추천 시스템을 구축한다.  
핵심 아이디어: **pre-trained text embedding을 user interest vector로 축적해 다중 관심사를 표현**한다.

---

## 2. 데이터셋 구조

### 파일 목록 (`../datasets/`)

| 파일 | 행 수 | 컬럼 | 설명 |
|------|-------|------|------|
| `userinfo.csv` | 16,975 | UserID, UserAgentID, UserAgentOSID, UserDeviceID, UserAgentFamilyID | 유저 메타데이터 |
| `searchinfo.csv` | 276,807 | UserID, **SearchID**, IPID, IsUserLoggedOn, **CategoryID** | 검색 이벤트 (행 순서 = embedding 순서) |
| `searchinfo_text_embs.npy` | 276,807 × 384 | — | 검색어 텍스트 임베딩 (searchinfo와 행 1:1 대응) |
| `adinfo.csv` | 17,518 | **AdID**, **CategoryID**, Price | 광고 메타데이터 (행 순서 = embedding 순서) |
| `adinfo_title_embs.npy` | 17,518 × 384 | — | 광고 제목 임베딩 (adinfo와 행 1:1 대응) |
| `search_stream_training.csv` | 320,000 | SearchID, AdID, Position, HistCTR, **IsClick** | 학습용 클릭 스트림 |
| `ad_validation_query.csv` | 214 | SearchID | Task A 검증 쿼리 |
| `ad_validation_answer.csv` | 214 | SearchID, AdID | Task A 정답 |
| `click_validation_query.csv` | 20,000 | SearchID, AdID, Position, HistCTR | Task B 검증 쿼리 |
| `click_validation_answer.csv` | 20,000 | SearchID, AdID, Position, HistCTR, IsClick | Task B 정답 |
| `ad_test_query.csv` | 214 | SearchID | Task A 테스트 쿼리 |
| `click_test_query.csv` | 20,000 | SearchID, AdID, Position, HistCTR | Task B 테스트 쿼리 |

### 핵심 통계

- 총 유저: 16,975명 / 학습 내 클릭 발생 유저: 2,385명
- Click ratio: 3,560 / 320,000 ≈ **1.1%** (극도의 class imbalance)
- 클릭 유저 1인당 평균 클릭 수: **1.49회** (매우 sparse)
- 임베딩 차원: **384** (sentence-transformers 계열 추정)

### 평가 Task

| Task | 입력 | 출력 | 평가 지표 (추정) |
|------|------|------|-----------------|
| **Task A** (Ad Recommendation) | SearchID | 추천 AdID 1개 | Accuracy / MRR |
| **Task B** (Click Prediction) | SearchID + AdID | IsClick (0/1) | AUC / Accuracy |

---

## 3. 알고리즘 설계

### 3-1. 핵심 구조

```
User Profile
└── interest_vectors: [k × 384]  # k개의 다중 관심사 벡터
```

- 초기값: **랜덤 초기화** (mean=0, std=1의 정규분포에서 샘플링 후 L2 정규화)
  - 0벡터 초기화는 모든 vector가 동일한 distance를 가져 soft assignment가 무의미해지는 문제가 있음
  - 랜덤 초기화는 k개의 vector가 초기부터 다양한 방향을 가리켜 specialization이 유도됨
- `k`는 hyperparameter (기본값: 5)

### 3-2. Soft Assignment (Distance-based Weighted Update)

새 임베딩 `e`가 주어졌을 때, 각 interest vector `v_i`와의 유사도를 계산해 가중 업데이트:

```
dist_i    = ||v_i - e||_2          # L2 거리 (또는 cosine distance)
weight_i  = exp(-dist_i / τ)       # temperature τ로 softness 조절
weight_i  = weight_i / Σ weight_j  # normalize → soft assignment

v_i ← v_i + α · weight_i · e      # 가중 업데이트 (α: learning rate)
```

### 3-3. 이벤트별 업데이트 규칙

| 이벤트 | 업데이트 방향 | Update rate | 비고 |
|--------|-------------|------------|------|
| 검색 발생 | `+search_emb` 방향 | `α_search` (소) | interest를 검색 의도 쪽으로 당김 |
| 광고 클릭 | `+ad_emb` 방향 | `α_click` (대) | interest를 클릭한 광고 쪽으로 당김 |
| 광고 비클릭 | **`-ad_emb` 방향** | `α_neg` (극소, 선택적) | interest를 비클릭 광고 **반대**로 밂 |

```
# 검색 시
v_i ← v_i + α_search · weight_i · search_emb

# 클릭 시
v_i ← v_i + α_click · weight_i · ad_emb

# 비클릭 시: 클릭된 ad/검색 방향과 반대인 non-clicked ad 방향에서 멀어짐
v_i ← v_i - α_neg · weight_i · ad_emb
```

> **설계 근거**: 비클릭은 `+ad_emb` update의 반대 방향인 `-ad_emb`를 사용한다. 결과적으로
> interest는 클릭된 광고·검색 방향으로 수렴하고 비클릭 광고 방향에서 멀어진다.  
> 단, position bias(상단 광고는 무조건 많이 노출됨)로 인해 비클릭이 "관심 없음"과 동일하지
> 않을 수 있으므로 `alpha_neg=0`(비활성) 옵션을 기본값으로 유지한다.

### 3-4. 예측 (Task A — Ad Recommendation)

주어진 검색의 search embedding `q`와 후보 광고들의 ad embedding `a_j`에 대해:

```
# User interest 중 q와 가장 가까운 것의 거리
user_interest_score = min_i(dist(v_i, q))  # user profile과의 관련성

# 각 광고 j에 대한 점수
score_j = sim(q, a_j)                      # 검색-광고 직접 유사도
        + γ · sim(user_interest, a_j)      # user interest와의 유사도

# 가장 높은 score의 AdID 선택
```

### 3-5. 예측 (Task B — Click Prediction)

```
# User의 interest 중 ad_emb에 가장 가까운 거리
d = min_i(dist(v_i, ad_emb))

# threshold 기반 이진 분류
IsClick = 1 if d < threshold else 0
```

---

## 4. 모델 API 명세 (Step 2 기준)

### 입력

```python
# 초기화
model = MultiInterestModel(
    k=5,                    # interest vector 개수
    dim=384,                # embedding 차원
    alpha_search=0.1,       # 검색 업데이트율
    alpha_click=0.5,        # 클릭 업데이트율
    alpha_neg=0.01,         # 비클릭 업데이트율 (0이면 비활성)
    temperature=1.0,        # soft assignment 온도
    threshold=0.8,          # Task B 분류 임계값
    gamma=0.5,              # Task A user interest 혼합 비율
)

# 학습 (스트림 방식)
model.update_search(user_id: int, search_emb: np.ndarray)
model.update_click(user_id: int, ad_emb: np.ndarray, clicked: bool)

# 예측
ad_id = model.predict_ad(user_id: int, query_emb: np.ndarray,
                          candidate_ad_embs: np.ndarray,
                          candidate_ad_ids: list) -> int

is_click = model.predict_click(user_id: int, query_emb: np.ndarray,
                                ad_emb: np.ndarray) -> int  # 0 or 1
```

### 출력

| 메서드 | 반환값 | 설명 |
|--------|--------|------|
| `predict_ad` | `int` | 추천 AdID |
| `predict_click` | `int` (0/1) | 클릭 여부 예측 |
| `get_interests(user_id)` | `np.ndarray [k×384]` | 디버깅용 interest 벡터 |

### Hyperparameter 요약

| 이름 | 기본값 | 범위 | 설명 |
|------|--------|------|------|
| `k` | 5 | 1~20 | interest vector 개수 |
| `dim` | 384 | 고정 | embedding 차원 |
| `alpha_search` | 0.1 | 0~1 | 검색 업데이트 강도 |
| `alpha_click` | 0.5 | 0~1 | 클릭 업데이트 강도 |
| `alpha_neg` | 0.01 | 0~0.1 | 비클릭 페널티 강도 |
| `temperature` | 1.0 | 0.1~5.0 | soft assignment softness |
| `threshold` | 0.8 | 0~2 | Task B 클릭 예측 임계값 |
| `gamma` | 0.5 | 0~1 | Task A user interest 혼합 비율 |

---

## 5. 파일 구조

```
[model-1]multi-interest by simple text embedding model/
├── PLAN.md                        # 이 문서
├── data/
│   └── dataset.py                 # Step 1: 데이터 파싱
├── model/
│   ├── __init__.py
│   ├── config.py                  # Step 2: Hyperparameter 정의
│   ├── interest.py                # Step 3: 핵심 모델 (MultiInterestModel)
│   └── predictor.py               # Step 3: 예측 로직
└── experiments/
    └── baseline_eval.py           # Step 4: 성능 평가
```

---

## 6. 개발 스케줄

| Step | 작업 | 산출물 | 상태 |
|------|------|--------|------|
| 1 | 데이터 파싱 코드 작성 | `data/dataset.py` | 예정 |
| 2 | API 명세 확정 (입출력, hyperparameter) | `model/config.py` | 예정 |
| 3 | 모델 내부 구현 | `model/interest.py`, `model/predictor.py` | 예정 |
| 4 | 성능 테스트 (validation 기준) | `experiments/baseline_eval.py` | 예정 |
| 5 | 분석 기반 수정 보완 | — | 예정 |

---

## 7. 예상 도전 과제 및 대응

| 문제 | 원인 | 대응 방안 |
|------|------|----------|
| Cold-start (신규 유저) | 랜덤 초기화된 interest vector | 이벤트 누적 전까지 검색-광고 직접 유사도만 사용 |
| Sparse click data | 클릭율 1.1%, 1인당 1.5회 | `alpha_neg=0` 설정해 비클릭 무시 옵션 |
| Interest drift (시간에 따른 변화) | soft assignment의 연속 업데이트로 자연 반영 | temperature 조절로 drift 속도 제어 |
| Task A 후보 광고 수 미정 | 전체 17,518개 광고 중 선택 | category 필터로 후보 축소 고려 |
| Class imbalance (Task B) | IsClick=1 : 1.1% | threshold 조정으로 sensitivity 확보 |
