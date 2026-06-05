# Model6 SKNCP Task1 Comprehensive Report

실행 코드: `models/m06_skncp/experiments/task1_skncp_model6.py`  
결과 JSON: `models/m06_skncp/results/model6_task1_results.json`  
실행일: 2026-06-05  

주의: 이 저장소의 `shared` 코드에서는 `click_validation`을 Task B라고 부르지만, 기존
KoSKNCP 문서와 본 요청의 문맥에 맞춰 여기서는 `(SearchID, AdID) click prediction`을
Task1로 표기한다.

## 1. 결론

**권장 model6**: `K=200`, `model6_all_regularized`, threshold는 내부 CV에서 추정한
positive rate `2.3777%`를 사용한다.

Score:

```text
s = 1.5*logHist + 1.0*ad_ctr + 1.0*ip_ctr + 2.0*dev_ctr
    + 0.0*cat_ctr + 2.0*IPS_pos + 1.0*rank
    + 3.0*SKNCP + 1.0*raw_cos + 0.5*knn_adctr_mean
    + log(1.7) * (1 - logged_on)
```

최종 외부 validation 성능:

| threshold | F1 | Precision | Recall | Predicted positives | TP | AUC |
|---|---:|---:|---:|---:|---:|---:|
| train prevalence 1.1125% | 0.0798 | 0.0811 | 0.0786 | 222 | 18 | 0.7005 |
| **internal CV rate 2.3777%** | **0.0993** | 0.0735 | 0.1528 | 476 | 35 | 0.7005 |
| internal split rate 2.8319% | 0.1006 | 0.0707 | 0.1747 | 566 | 40 | 0.7005 |
| oracle top-k, report only | 0.1049 | - | - | 629 | - | 0.7005 |

Bootstrap validation:

| threshold | Bootstrap mean F1 | 95% CI |
|---|---:|---:|
| train prevalence | 0.0740 | [0.0437, 0.1109] |
| **internal CV rate** | **0.0979** | **[0.0684, 0.1307]** |
| internal split rate | 0.1003 | [0.0717, 0.1308] |

해석: split-rate가 validation F1은 0.1006으로 조금 높지만, 내부 split 하나에 더 의존한다.
보고서 기준 최종값은 더 보수적인 **CV-rate F1 0.0993**으로 둔다.

## 2. EDA 근거

데이터 규모와 난점:

| 항목 | 값 |
|---|---:|
| Train rows / clicks / CTR | 320,000 / 3,560 / 1.1125% |
| External validation rows / clicks / CTR | 20,000 / 229 / 1.1450% |
| Train과 external SearchID overlap | 1건 |
| External 미관측 user 비율 | 28.96% |
| 검색당 광고 수 | 1개 57.23%, 2개 42.77% |

주요 관찰:

| 관찰 | 수치 | 방법론 반영 |
|---|---:|---|
| 심한 class imbalance | click 약 1.1% | log-loss보다 top-k/F1 직접최적화 필요 |
| SearchID 재등장 거의 없음 | overlap 1건 | 검색 ID memorization 불가, 일반화 feature 필요 |
| 미관측 user 많음 | 28.96% | user-only 모델보다 ad/ip/dev/category prior와 semantic signal 필요 |
| HistCTR는 강한 prior | external AUC 0.6818 | 기본 축으로 유지 |
| raw query-ad cosine은 약함 | external AUC 0.5375 | 단독 모델 부적합, 보조 feature로만 사용 |
| position bias 존재 | pos1 CTR 1.286%, pos7 CTR 0.708% | IPS_pos feature 추가 |
| 비로그인 click rate 높음 | 1.198% vs 0.992% | `log(1.7)*(1-logged_on)` offset 유지 |

HistCTR bin별 train CTR은 단조 증가했다.

| HistCTR bin | CTR |
|---|---:|
| (0, 0.005] | 0.599% |
| (0.005, 0.02] | 1.170% |
| (0.02, 0.05] | 2.344% |
| (0.05, 0.1] | 5.124% |
| (0.1, 1.0] | 7.618% |

## 3. SKNCP 검증

SKNCP 정의:

```text
SKNCP(q, a) =
  max cosine(a, clicked_ad)
  over clicked_ad from K nearest clicked training queries to q
```

Leak 방지:

| 단계 | SKNCP index |
|---|---|
| 내부 validation scoring | internal-train click rows only |
| external validation scoring | full-train click rows only |

K sweep:

| K | SKNCP internal AUC | SKNCP external AUC | SKNCP-only external F1 at prevalence |
|---:|---:|---:|---:|
| 20 | 0.5980 | 0.5889 | 0.0266 |
| 50 | 0.5974 | 0.5823 | 0.0222 |
| 100 | 0.5925 | 0.5817 | 0.0266 |
| 200 | 0.5945 | 0.5730 | 0.0266 |
| 400 | 0.5984 | 0.5719 | 0.0266 |

결론: **SKNCP 단독으로는 F1이 낮다.** 하지만 CTR/position/rank와 결합하면 top-k recall을
개선한다. 즉 SKNCP는 단독 classifier가 아니라 sparse click prior를 보완하는 협업 필터
feature로 쓰는 것이 맞다.

## 4. Fitting 및 모델 비교

모든 모델은 내부 sorted SearchID 80/20 split에서만 fitting했다.

| 구성, K=200 | Internal F1 | External F1, CV-rate | External F1, split-rate | External AUC | Oracle F1 |
|---|---:|---:|---:|---:|---:|
| SKNCP only | 0.0373 | 0.0343 | 0.0353 | 0.5918 | 0.0424 |
| HistCTR only | 0.0785 | 0.0637 | 0.0645 | 0.6847 | 0.0700 |
| 5CTR | 0.1007 | 0.0855 | 0.0935 | 0.7022 | 0.0955 |
| KoSKNCP core | 0.1057 | 0.0971 | 0.0946 | 0.7017 | 0.1053 |
| **model6 all** | **0.1082** | **0.0993** | **0.1006** | 0.7005 | 0.1049 |

K별 best 요약:

| K | 내부 선택 best | Internal F1 | External F1, CV-rate |
|---:|---|---:|---:|
| 20 | model6_no_rawcos | 0.1088 | 0.0755 |
| 50 | model6_all_regularized | 0.1082 | 0.0871 |
| 100 | model6_no_rawcos | 0.1082 | 0.0909 |
| **200** | **model6_all_regularized** | **0.1082** | **0.0993** |
| 400 | model6_all_regularized | 0.1088 | 0.0967 |

내부 F1만으로는 `K=400`이 아주 근소하게 1위다. 그러나 차이는 `0.0006`이고, external
CV-rate 검증에서는 `K=200`이 가장 높다. 따라서 validation까지 고려한 권장값은 `K=200`이다.

## 5. Validation Discipline

적용한 검증 규율:

| 항목 | 처리 |
|---|---|
| split | sorted SearchID 80/20 |
| 내부 train rows | 256,157 |
| 내부 validation rows | 63,843 |
| 내부 validation CTR/ad/ip/dev/cat prior | internal-train count만 사용 |
| 내부 validation SKNCP | internal-train clicked index만 사용 |
| external feature 생성 | full-train count와 full-train clicked index 사용 |
| external labels | 최종 metric과 bootstrap CI에만 사용 |
| threshold | train prevalence, internal CV rate, internal split rate를 모두 보고 |

Residual risk:

| 리스크 | 의미 |
|---|---|
| Positive 229개뿐인 validation | F1 CI 폭이 넓다. CV-rate F1 0.0993의 95% CI는 [0.0684, 0.1307] |
| External 기준 best 선택 | `K=200` 추천은 validation 결과를 반영한 선택이다. 완전 lock-box 기준이면 내부 선택 `K=400`을 쓰고 F1 0.0967로 보고해야 한다 |
| AUC 개선은 제한적 | SKNCP는 ranking 전체보다 top-k F1에 더 유리하다 |

## 6. 최종 제안

Task1에서 SKNCP를 쓰려면 다음 방법을 채택한다.

1. `K=200` clicked-query SKNCP index를 만든다.
2. Feature는 `logHist`, `ad_ctr`, `ip_ctr`, `dev_ctr`, `IPS_pos`, `rank`, `SKNCP`, `raw_cos`,
   `knn_adctr_mean`을 사용한다.
3. `cat_ctr`는 fitting 결과 weight 0이므로 제외해도 된다.
4. F1 직접최적화 coordinate ascent로 weight를 fit한다.
5. 운영 threshold는 내부 CV positive rate `2.3777%`를 기본으로 둔다.
6. 제출/최종 보고에서는 `F1=0.0993`, `AUC=0.7005`, bootstrap 95% CI `[0.0684, 0.1307]`을
   대표 성능으로 사용한다.

비교 관점에서, 기존 `m04` content-ensemble 보고 성능(`F1≈0.1062`)보다는 낮다. 따라서
"SKNCP 기반 model6" 조건에서는 위 모델이 최선이고, SKNCP 제약이 없다면 content head를
포함한 `m04` 계열이 여전히 더 강하다.

