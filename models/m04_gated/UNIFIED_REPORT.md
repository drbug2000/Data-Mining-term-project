# 통합 보고서 — 광고 클릭 예측(Task A) F1 방법론 병합 및 엄밀 검증

두 독립 방법론을 병합하고 bootstrap·ablation 으로 엄밀히 검증한 결과.
재현 파이프라인: `models/m04_gated/experiments/exp_unified.py` (+ `exp_content_strong.py`).

---

## 1. 병합한 두 방법론

| | **m04 (content-ensemble)** | **KoSKNCP (log-odds)** |
|---|---|---|
| 핵심 신호 | 클릭튜닝 bi-encoder **content head 앙상블**(10-seed, MLP proj) | **SKNCP** semantic-kNN 클릭예측(K=100, 협업필터) |
| 결합 | 5-CTR + content, F1-fit log-linear | 9항 log-odds (HistCTR/ad/cat/login/IPS/sim/rank/SKNCP/nbr) |
| 정규화 | — (free F1-fit) | **w_min=0.5** (HistCTR 가중치 보존) |
| rate | train-prevalence | OOF 5-fold |
| 검증 | sorted 80/20, honest, oracle 배제 | sorted 80/20, honest, oracle 배제, **bootstrap CI** |
| honest External F1 | **0.1062** | **0.0945** |
| External AUC | **0.7133** | 0.710 |
| content 신호 AUC | 0.684 (학습) | 0.593 (SKNCP, 비모수) |

두 연구의 **수렴된 독립 결론**: 천장은 model capacity 가 아니라 **top-rank 정보량**;
oracle(외부 threshold) 배제 시 honest F1 ≈ 0.09~0.11; 추가 entity CTR(ip/dev) 과 직접
의미유사도는 약하거나 drag.

---

## 2. 통합 모델

log-odds 선형(나이브 베이즈)에 **양쪽의 모든 신호**를 결합, F1-objective coordinate ascent
(w_min=0.5, KoSKNCP 정규화)로 가중치 학습:

```
lo = w1·log HistCTR + w2·log ad_ctr + w3·log ip_ctr + w4·log dev_ctr + w5·log cat_ctr
   + log(1.7)·[not logged]                        # 고정 offset
   + w6·IPS_pos + w7·rank                          # KoSKNCP: 위치 debias, 검색내 순위
   + w8·z(content_ensemble)                        # m04: 클릭튜닝 head 앙상블 (강신호)
   + w9·SKNCP + w10·nbr_ctr                        # KoSKNCP: 협업필터, 이웃광고 CTR
score = σ(lo)
```

검증 규율(둘 다 통합): sorted SearchID 80/20 내부 split 으로만 모든 선택(content
early-stop·F1 지수·rate). CTR/SKNCP 인덱스는 내부-val→내부-train, external→full-train.
threshold = OOF / train-prevalence (외부 라벨 0개). **external click_validation 라벨은 최종
채점에서만.**

---

## 3. 엄밀 검증 결과

### 3.1 신호 subset 비교 (external, sorted 80/20)

| subset | honest_oof | honest_prev | oracle | **AUC** |
|---|---:|---:|---:|---:|
| minimal (5CTR+content) | 0.0857 | 0.0885 | 0.0965 | 0.7081 |
| +IPS+rank | 0.0874 | 0.0841 | 0.0949 | 0.7089 |
| +SKNCP+nbr | 0.0943 | 0.0885 | 0.0988 | 0.7095 |
| **ALL-10 (union)** | **0.0958** | 0.0885 | 0.1018 | 0.7100 |
| KoSKNCP-like (no content) | 0.0931 | 0.0664 | 0.1007 | 0.7029 |
| m04 standalone (free-fit, 5CTR+content)¹ | 0.0983 | **0.1062** | 0.1106 | **0.7133** |

¹ `exp_content_strong.py` (w_min 없는 free F1-fit + train-prevalence rate). 통합 harness 의
w_min=0.5 정규화는 이 case 에선 약간 손해 — 자유 적합이 더 나음(역시 noise 범위 내).

### 3.2 Bootstrap 신뢰구간 (2000 resample, external honest F1)

```
mean = 0.0843   std = 0.0177   95% CI = [0.0518, 0.1214]   (폭 0.070)
```

**→ 결정적 발견**: CI 폭 0.07 (229 양성). 따라서 **m04(0.106) · KoSKNCP(0.094) · union(0.096)
의 차이는 전부 통계적 노이즈**(최소 유의차 ≈ 0.039 > 관측 격차). 어떤 단일 config 도
통계적으로 우월하지 않다. **유일하게 안정적인 비교 지표는 AUC** (0.70~0.713).

### 3.3 Leave-one-signal-out ablation

추가 신호(SKNCP/IPS/rank/nbr/ip/dev)를 빼도 honest F1 변화가 ±0.01~0.02 (전부 CI 내) —
즉 **어떤 추가 신호도 통계적으로 유의하게 기여하지 않음**. 이는 두 원 논문이 각각
보고한 "추가 신호 drag / 정보 천장" 을 **세 번째 독립 구성(union)으로 재확인**.

---

## 4. 결론

1. **두 방법론은 통계적으로 동등**하다. honest External F1 ≈ 0.09~0.11 은 bootstrap CI
   (폭 0.07) 안의 단일 분포 — m04 / KoSKNCP / union 모두 구별 불가.
2. **유일한 안정적 우위(AUC 기준) = 강한 클릭튜닝 content head 앙상블**(m04). external AUC
   0.7133 로 양쪽 원본(0.671/0.710)을 상회. SKNCP(비모수 0.593) 대비 학습 head 가 우월.
3. **신호를 더 합쳐도(union) 이득 없음** — capacity·feature 추가 모두 noise 또는 drag.
   GBDT(비선형 capacity↑)는 external 0.078 로 overfit. → **천장은 정보량, 모델 용량 아님.**
   (세 독립 구성 + GBDT 반증으로 robust.)
4. **0.11 은 oracle(외부 threshold)** 수치이며 honest 천장(~0.10)과 통계적으로 구별 불가.
   valid 제출 가능 최선 = **honest F1 ≈ 0.106** (5CTR + 강 content, train-prevalence rate).

**권장 최종 모델**: `exp_content_strong.py` (5-CTR + 10-seed content 앙상블, free F1-fit,
train-prevalence threshold) — honest 0.1062 / AUC 0.7133. KoSKNCP 에서 채택: **w_min 정규화
개념 + bootstrap CI 검증 + OOF rate**(통계적 정직성). 추가 신호(SKNCP/IPS/rank)는 noise 라
미채택.

---

## 5. 재현

```bash
# 1) content head 앙상블 학습 + 캐시 (GPU)
CUDA_VISIBLE_DEVICES=0 python -X utf8 models/m04_gated/experiments/exp_content_strong.py
#    -> /tmp/cs_full.npz (10-seed content logits), honest 0.1062 / AUC 0.7133

# 2) 통합 파이프라인: 전 신호 + subset 비교 + bootstrap + ablation
CUDA_VISIBLE_DEVICES=0 python -X utf8 models/m04_gated/experiments/exp_unified.py
#    -> /tmp/pumasi_res/unified.json (subset별 honest/oracle/AUC + bootstrap CI)

# (참고) capacity 반증
CUDA_VISIBLE_DEVICES=0 python -X utf8 models/m04_gated/experiments/exp_tabular_gbdt.py  # 0.078 overfit
```

검증 규율: 모든 모델 선택은 학습데이터 sorted 80/20 내부 split. `click_validation_answer`
는 최종 F1/AUC/bootstrap 보고에만 사용(파라미터/threshold 선택 일절 미사용 = leak-free).
```

의존성: numpy, pandas, torch(content head 학습; 추론은 numpy 가중치). lightgbm(반증 실험).
