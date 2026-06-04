# 실험 로그 — GNN + CTR MLP 파이프라인

**작성일**: 2026-06-03  
**목표**: Task A (클릭 예측, F1) 성능 최대화

---

## 파이프라인 구조

```
RecoDataset
    │
    ▼
build_graph()  →  HeteroGraph
    │               (user / search / ad 노드)
    │               (click / show / sim 엣지)
    ▼
GNNModel.fit()  →  node repr (h_search, h_ad, h_user)
    │
    ▼
CTRPredictor.fit()  →  MLP (10d 피처 → 클릭 확률)
    │
    ▼
Task A 평가 (F1, AUC)  /  Task B 평가 (NDCG@3)
```

---

## 데이터셋 통계

| 항목 | 수치 |
|------|------|
| 총 유저 | 16,975 |
| 검색 이벤트 | 276,807 |
| 광고 수 | 17,518 |
| 훈련 행 | 320,000 (클릭 3,560, CTR 1.11%) |
| Val Task A | 20,000 쌍 (클릭 229, CTR 1.15%) |
| Val Task B | 214 쿼리 |
| 임베딩 차원 | 384 |

---

## Cold-start 분석 결과

Val Task A (20,000 쌍) 기준:

| 그룹 | 비율 | CTR |
|------|:----:|:---:|
| 신규 유저 (훈련 미등장) | 29.0% | 1.36% |
| 기존 유저, 클릭 이력 없음 | 47.1% | 1.00% |
| 기존 유저, 클릭 이력 있음 | 23.9% | 1.17% |
| 신규 광고 (훈련 미노출) | 5.0% | 1.01% |
| 기존 광고, 클릭 이력 없음 | 57.4% | 0.81% |
| 기존 광고, 클릭 이력 있음 | 37.6% | **1.67%** |

→ **val의 77.4%가 cold-start 조건** (user 신규/비클릭 OR ad 신규)

---

## 피처 분석 결과 (gap_analysis.py)

훈련 통계 피처의 val coverage:

| 피처 | val에서 prior로 대체되는 비율 |
|------|:----:|
| `log_ad_click` | **62.4%** |
| `user_cat_ctr` | **44.0%** |
| `log_user_srch` | **29.0%** |

핵심 발견:
- `sim_su` (query-user 유사도): train 상관 **-0.460** vs val 상관 -0.015 → spurious leakage
- `user_ctr`, `user_cat_ctr`: 훈련 레이블을 직접 집계 → train에서만 강한 신호
- `is_logged_on`: val에서 std=0 (상수) → 정보 없음
- `hist_ctr`: train 0.087 vs val 0.076 → **가장 안정적인 신호**

---

## 주요 실험 결과 — Task A (F1 / AUC)

### 모델 구성별 비교

| 모델 | F1 | AUC | 비고 |
|------|:--:|:---:|------|
| Always-0 baseline | 0.0000 | 0.5000 | |
| HistCTR baseline | 0.0436 | — | 강한 단순 베이스라인 |
| Interest 모델 (γ=0.7) | 0.0335 | 0.5576 | |
| GNN link pred only | 0.0242 | 0.5115 | |
| GNN + MLP (초기, 393d bilinear) | 0.0356 | 0.5882 | |
| GNN + MLP (16d, 통계 포함) | 0.0467 | 0.5750 | |
| GNN + MLP (8d, 통계 제거) | 0.0480 | 0.6208 | |
| GNN + MLP (16d + early stop) | 0.0524 | 0.6398 | smoothing 추가 |
| GNN + MLP (8d, user emb 제거) | 0.0480 | 0.6208 | |
| GNN + MLP (6d, 통계+user 제거) | 0.0749 | 0.6435 | **user emb 제거 핵심** |
| GNN + MLP (10d, pos+cat 추가) | **0.0760** | **0.6691** | ✅ **현재 최고** |

### 과적합 개선 이력

| 조치 | 효과 |
|------|------|
| Early stopping (patience=5) | val F1 0.0416 → 0.0524 |
| AdamW weight decay (λ=1e-3) | 과적합 속도 감소 |
| Dropout 0.2 → 0.4 | 정규화 강화 |
| hidden 128→64, 64→32 | 파라미터 감소 |
| Bayesian smoothing (m=10) | 통계 피처 분산 안정화 |
| Z-score 정규화 | 피처 스케일 통일 |

---

## 현재 최적 구성 (Task A 기준)

### GNN 설정

```python
GNNConfig(n_layers=2, agg_fn="mean", gamma=0.7, gamma_search=0.5)
```

```
build_graph(transductive=True, include_test=True, top_k_sim=0)
```

| 항목 | 값 |
|------|-----|
| 노드 | user 15,144 / search 240,851 / ad 17,518 |
| 엣지 | click 3,560 / show 336,471 / user_to_search 240,698 |
| sim 엣지 | **없음** (Task A에 해로움) |

### MLP 설정

```python
CTRConfig(
    hidden_dim=64, n_epochs=50, batch_size=512,
    lr=1e-3, dropout=0.4, weight_decay=1e-3,
    focal_gamma=2.0, smooth_prior=10, patience=5
)
```

### 피처 (10d)

| # | 피처 | 출처 | 비고 |
|---|------|------|------|
| 1 | `sim_sa` | `cosine(h_s_gnn, h_a_gnn)` | GNN 전파 후 query-ad 유사도 |
| 2 | `log_pos` | `log(1+position)` | CSV |
| 3 | `inv_pos` | `1/(1+position)` | 지수 감소 bias |
| 4 | `is_top1` | `position==1` | 1위 광고 플래그 |
| 5 | `hist_ctr` | HistCTR 컬럼 | CSV, 가장 강한 신호 |
| 6 | `cat_match` | `search_cat==ad_cat` | 카테고리 일치 |
| 7 | `search_cat_id` | SearchCategoryID | z-score 정규화 |
| 8 | `ad_cat_id` | AdCategoryID | z-score 정규화 |
| 9 | `log_price` | `log(1+price)` | adinfo.csv |
| 10 | `is_logged_on` | IsUserLoggedOn | searchinfo.csv |

**제거된 피처**:
- `sim_ua`, `sim_su` (user GNN repr): val 76%가 cold-start → spurious correlation
- 훈련 통계 피처 8종: val coverage 29~62% → leakage 후 val 성능 저하
- 384d bilinear interaction: 파라미터 과다 (데이터 대비)

### 최종 성능

| Task | 지표 | 값 |
|------|------|:--:|
| **Task A** | F1 | **0.0760** |
| **Task A** | AUC | **0.6691** |
| **Task A** | Precision | 0.0536 |
| **Task A** | Recall | 0.1310 |
| Task B | NDCG@3 | 0.0906 |

---

## Sim 엣지 실험 결과 (Task A 기준)

| top_k_sim | 피처 | Task A F1 | Task A AUC | Task B NDCG@3 |
|:---------:|------|:---------:|:----------:|:-------------:|
| **0 (최적)** | 단일 sim_sa | **0.0760** | **0.6691** | 0.0906 |
| 2 | raw+gnn 분리 | 0.0581 | 0.6512 | 0.0906 |
| 3 | raw+gnn 분리 | 0.0629 | 0.6123 | 0.0906 |
| 10 | 단일 sim_sa | 0.0553 | 0.5950 | **0.0930** |
| 10 | raw+gnn 분리 | 0.0549 | 0.6460 | 0.0930 |

**결론**: sim 엣지는 Task B(광고 추천)에 유리하나 Task A(클릭 예측)에 일관되게 해로움.  
원인: GNN 전파가 `sim_sa` 피처를 오염시켜 MLP 과적합 유발.

---

## 핵심 인사이트 요약

1. **User embedding 제거가 가장 큰 개선 요인** (F1 +56%): val의 76%가 cold-start user라 `h_u`가 noise. `sim_su`는 train 상관 -0.46이지만 val 상관 -0.015인 spurious feature.

2. **hist_ctr이 가장 안정적인 신호**: train/val 상관이 유사 (0.087/0.076). HistCTR baseline이 단순하지만 강한 이유.

3. **훈련 통계 피처는 leakage**: `user_ctr`, `user_cat_ctr` 등은 훈련 레이블 집계값 → train F1 0.55 달성하지만 val F1 0.048에 그침. Val coverage 44~62% zero.

4. **Early stopping이 핵심 정규화**: Val F1 피크가 epoch 2~13에 분포. 30 epoch까지 돌리면 일관되게 성능 하락.

5. **Task A와 Task B는 독립적 문제**: Cohen's d 분석에서 query-ad 유사도의 click 구분력 d=0.15 (Task A) vs d=1.56 (Task B). 같은 모델이 두 task를 동시에 최적화하기 어려움.

---

## 실행 방법

```bash
# 현재 최적 구성으로 평가
python -X utf8 experiments/ctr_eval.py

# Cold-start 분석
python -X utf8 experiments/coldstart_analysis.py

# 피처 분포/leakage 분석
python -X utf8 experiments/gap_analysis.py
```
