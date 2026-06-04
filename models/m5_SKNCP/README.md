# Task 1: Advertisement Click Prediction

**AI506 Data Mining — Term Project**

## Overview

Predicts whether a user will click an advertisement given a search event context.  
Evaluation metric: **F1 score** (click class).

### Results

| Model | External Val F1 | AUC |
|-------|----------------|-----|
| HistCTR baseline | 0.0436 | 0.669 |
| **This method** | **0.0945–0.0949** | **0.710** |

---

## Method

### Core Components

#### 1. Log-linear Scoring in Log-odds Space

All signals are combined in log-odds (logit) space — the natural domain for
independent evidence under the Naive Bayes assumption:

```
lo(q, a, pos) = w1·log(HistCTR)
              + w2·log(ad_ctr)
              + w3·log(cat_ctr)
              + LOGIN_OFFSET · not_logged          # fixed
              + w4·log(pos_ctr / global_ctr)       # position debiasing
              + w5·(semantic_sim − 0.5) / 0.22
              + w6·(0.5 − within_search_rank)
              + w7·(SKNCP − 0.30)
              + w8·(logit(nbhd_ctr) − logit(GCT))

score = sigmoid(lo)
```

#### 2. SKNCP — Semantic K-Nearest-Neighbor Click Prediction

Direct query–ad cosine similarity is nearly useless (Cohen's d = 0.04).
Instead, SKNCP retrieves the K=100 most similar training queries and asks:
*"How similar is the current ad to the ads that were actually clicked in those similar searches?"*

```
SKNCP(q, a) = max_{c ∈ clicked_ads(KNN(q))} cosine(embed(a), c)
```

Improvement over direct similarity: Cohen's d = 0.04 → **0.44** (11×).

#### 3. Laplace-smoothed CTR Features

```
ctr(entity) = (n_clicks + k · global_ctr) / (n_impressions + k),  k = 20
```

Unseen entities fall back to global CTR.

#### 4. F1-objective Coordinate Ascent

Standard log-loss optimisation misaligns with F1 at 1.1% CTR.
Coordinate Ascent directly maximises Top-K F1 on the internal validation set:

```
for each weight w_j in [0.5, 3.0] step 0.1:
    w_j ← argmax F1_topk(score(X_ival, w), y_ival)
```

Weight floor `w_min = 0.5` prevents the strongest signal (HistCTR, AUC=0.669)
from being zeroed out due to local distributional quirks in the internal val set.

#### 5. OOF Threshold

The positive prediction rate is estimated by 5-fold Out-of-Fold averaging on
the internal validation set, then applied directly to the test set.

---

## Validation Methodology

```
Training 320K  ──(chronological 80/20)──>  Internal Train (80%)
                                            Internal Val  (20%)   ← parameter fitting
External Val 20K  ──────────────────────>  Final reporting only  ← no fitting
Test 20K         ──────────────────────>  Submission
```

**Why chronological split?**
The dataset was collected chronologically (confirmed by the project spec),
so a time-based split correctly simulates deployment conditions.

---

## Project Structure

```
task1_github/
├── main.py           # entry point — run this
├── config.py         # all hyperparameters in one place
├── data_utils.py     # data loading, CTR statistics, SKNCP index
├── skncp.py          # Semantic KNN Click Prediction
├── features.py       # 9-dimensional feature engineering
├── model.py          # log-linear scoring + Coordinate Ascent + OOF threshold
├── evaluate.py       # AUC, F1, precision/recall
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Dataset

Set `DATASET_DIR` in `config.py` to point to the folder containing:
- `searchinfo.csv`, `adinfo.csv`, `userinfo.csv`
- `search_stream_training.csv`
- `click_validation_query.csv`, `click_validation_answer.csv`
- `click_test_query.csv`
- `searchinfo_text_embs.npy`, `adinfo_title_embs.npy`

### Run

```bash
python main.py
```

Outputs:
- Console: External Val F1 / AUC
- File: `click_test_answer.csv` (submission file)

Runtime: ~5 minutes (dominated by SKNCP KNN computation)

---

## Key References

| Component | Reference |
|-----------|-----------|
| KNN collaborative filtering | Sarwar et al. (2001), WWW |
| kNN augmented prediction | Khandelwal et al. (2020), ICLR |
| Historical CTR | Richardson et al. (2007), WWW |
| Sentence embeddings | Reimers & Gurevych (2019), EMNLP |
| IPS position debiasing | Joachims et al. (2017), SIGIR |
