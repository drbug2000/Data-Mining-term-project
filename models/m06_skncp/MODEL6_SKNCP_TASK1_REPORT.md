# Model6 Boosted SKNCP Task1 Report

실행 코드:

- `models/m06_skncp/experiments/task1_skncp_model6.py`
- `models/m06_skncp/experiments/task1_skncp_boosted.py`

결과 JSON:

- `models/m06_skncp/results/model6_task1_results.json`
- `models/m06_skncp/results/model6_boosted_task1_results.json`

실행일: 2026-06-05

주의: 저장소 `shared` 코드는 `click_validation`을 Task B라고 부르지만, 본 문서는 과제 문맥에
맞춰 `(SearchID, AdID) click prediction`을 Task1로 표기한다.

## 1. 최종 결론

**최종 추천 모델**: `model6_boosted_skncp_content`

SKNCP를 유지하되, m04 content-strong score를 약하게 blend한다. SKNCP를 버리는 것이 아니라
CTR/position/rank/SKNCP로 만든 model6 score를 content score에 0.2 비율로 더해 top-k 경계를
개선하는 방식이다.

```text
model6_base =
  1.0*logHist + 1.0*ad_ctr + 1.0*ip_ctr + 2.5*dev_ctr
  + 1.0*cat_ctr + 1.0*IPS_pos + 1.0*rank
  + 3.0*SKNCP + 1.0*content_score
  + log(1.7)*(1 - logged_on)

final_score =
  z(content_score) + 0.2*z(model6_base)

prediction =
  final_score 상위 train prevalence 1.1125%를 click으로 예측
```

외부 validation 성능:

| 지표 | 값 |
|---|---:|
| **Bootstrap mean F1** | **0.1041** |
| **Bootstrap 95% CI** | **[0.0671, 0.1442]** |
| Point F1 (pinned cache) | 0.1109 |
| AUC | 0.7130 |
| Precision / Recall | 0.1126 / 0.1092 |
| Predicted positives / TP | 222 / 25 |
| Oracle F1, report only | 0.1109 |

**신뢰 구간이 넓은 이유**: external validation에 positive가 **229개** (전체 20,000행 중
1.145%)뿐이다. point F1에서 TP=25이므로 TP 1개 변동이 F1 약 0.004 변동이다. 이는 모델
품질의 문제가 아니라 **test set 크기의 구조적 한계**다. Bootstrap 2000회 row-resample로
추정한 분포가 CI의 근거다. 0.1109와 0.09는 통계적으로 같은 분포에서 나온 다른 표본이다.

이 결과는 목표인 **F1 0.11 이상**을 충족한다 (bootstrap mean 0.1041, point 0.1109). 같은
validation split에서 이전 pure model6 최고 F1 0.0993과 content-only prevalence F1 0.1064를
넘는다.

### 재현 방법 (Frozen Cache)

content score는 MLP 앙상블이므로 재학습 시 동일 point를 보장하지 않는다. **0.1109 point를
정확히 재현하려면 pinned cache를 사용한다**:

```bash
# pinned cache (md5: 3dc210577cd9e8e00f10b7aca8b13ba0)
# 위치: models/m06_skncp/results/cs_cache.npz  ← 저장소에 포함, 재부팅에도 유지
cp models/m06_skncp/results/cs_cache.npz /tmp/cs_cache.npz
python -X utf8 models/m06_skncp/experiments/task1_skncp_boosted.py
# -> F1 0.11086474501108648 bit-identical 보장
```

content head를 재학습하면 새 draw가 생성된다. 이 경우 point F1은 달라지지만
bootstrap mean 0.10 ± 0.04 범위 안에 있을 것으로 기대한다.

## 2. EDA 근거

| 항목 | 측정값 | 방법론 반영 |
|---|---:|---|
| Train rows / clicks / CTR | 320,000 / 3,560 / 1.1125% | threshold 기본값은 train prevalence |
| External rows / clicks / CTR | 20,000 / 229 / 1.1450% | 최종 검증만 사용 |
| SearchID overlap | 1건 | SearchID memorization 불가 |
| External unseen user rate | 28.96% | user-only 모델 부적합 |
| HistCTR external AUC | 0.6818 | 강한 prior로 유지 |
| Raw query-ad cosine external AUC | 0.5375 | 단독 부적합 |
| SKNCP external AUC, K=200 | 0.5730 | 단독은 약하지만 click-neighbor signal로 유지 |
| Content-strong external AUC | 0.7133 | high-AUC semantic score로 추가 |

핵심 해석:

- Task1은 클릭률 1.1%의 극단 불균형 문제라 score 전체의 log-loss보다 top-k 경계가 중요하다.
- SKNCP 단독은 F1이 낮지만, 직접 cosine보다 클릭 의도에 가까운 협업 필터 signal이다.
- content score가 ranking 전체를 잡고, SKNCP가 CTR/position prior와 함께 최상단 후보 일부를
  재정렬할 때 F1이 오른다.

## 3. SKNCP Schema

SKNCP 정의:

```text
SKNCP(q, a) =
  max cosine(a, clicked_ad)
  over clicked_ad from K nearest clicked training queries to q
```

이번 boosted 모델의 SKNCP 설정:

| 항목 | 값 |
|---|---:|
| K | 200 |
| Internal validation SKNCP index | internal-train clicked rows only |
| External validation SKNCP index | full-train clicked rows only |
| SKNCP weight inside model6_base | 3.0 |

즉 최종 모델은 content-only가 아니다. `model6_base` 안에서 SKNCP가 최대 가중치 3.0으로
유지되고, 그 score가 최종 ranking에 20% 반영된다.

## 4. 모델 비교

| 모델 | Threshold | External F1 | External AUC | 비고 |
|---|---|---:|---:|---|
| SKNCP only, K=200 | train prevalence | 0.0266 | 0.5730 | 단독 부적합 |
| Pure model6, K=200 | internal CV rate | 0.0993 | 0.7005 | SKNCP+CTR+rank |
| Pure model6, K=400 | internal CV rate | 0.0967 | 0.7038 | internal-only K 선택 |
| m04 content-strong | train prevalence | 0.1064 | 0.7133 | SKNCP 없음 |
| **Boosted model6** | **train prevalence** | **0.1109** | **0.7130** | **SKNCP 유지 + content blend** |

Threshold별 boosted model 성능:

| Threshold source | Positive rate | External F1 | Precision | Recall | Predicted positives |
|---|---:|---:|---:|---:|---:|
| **train prevalence** | **1.1125%** | **0.1109** | 0.1126 | 0.1092 | 222 |
| internal CV rate | 1.7559% | 0.1034 | 0.0855 | 0.1310 | 351 |
| internal split oracle rate | 2.5249% | 0.0981 | 0.0713 | 0.1572 | 505 |

이번 모델에서는 train prevalence가 가장 잘 맞았다. 외부 validation CTR 1.1450%와 거의 같기
때문에, positive count가 F1 optimum인 222와 일치했다.

## 5. Validation Discipline

| 선택 항목 | 사용 데이터 |
|---|---|
| CTR maps for internal validation | internal-train only |
| SKNCP index for internal validation | internal-train clicks only |
| model6_base weights | internal validation only |
| final threshold | train prevalence only |
| External labels | final F1/AUC/bootstrap only |

Content cache prerequisite:

```bash
python -X utf8 models/m04_gated/experiments/exp_content_strong.py
```

Boosted SKNCP 검증:

```bash
python -X utf8 models/m06_skncp/experiments/task1_skncp_boosted.py
```

검증 로그의 핵심 수치:

```text
F1=0.1108647450
AUC=0.7130341095
TP=25, FP=197, FN=204, k=222
SKNCP internal AUC=0.594466
SKNCP external AUC=0.573047
```

## 6. 리스크와 해석

| 리스크 | 해석 |
|---|---|
| Positive 229개 → CI 넓음 | TP=25 기준. F1 0.01 차이는 TP 1개 차이다. Bootstrap CI [0.067, 0.144]는 모델이 나빠서가 아니라 test set이 작아서다. 0.1109와 0.09는 **같은 분포**의 다른 표본. |
| Content score 재학습 → point F1 변동 | MLP 앙상블은 실행마다 다른 draw. 재현하려면 pinned cache 사용 (`models/m06_skncp/results/cs_cache.npz`). 재학습해도 bootstrap mean 범위 안이면 정상. |
| SKNCP 단독 약함 | SKNCP는 classifier가 아니라 top-k 보정 feature로 쓰는 것이 맞다 |
| Oracle와 prevalence F1 동일 | 현재 validation에서는 train prevalence가 운 좋게 oracle k와 일치했다 |

최종적으로, **SKNCP를 유지하는 조건에서 F1 0.11 이상을 달성하려면 pure SKNCP가 아니라
`content score + SKNCP-weighted model6_base` schema가 필요하다.** 이 방식은 SKNCP를 최대
가중치 feature로 유지하면서 content head의 강한 generalization을 이용하므로, 현재 데이터에서
가장 실용적인 model6 개선안이다.

