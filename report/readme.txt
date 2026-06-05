AI506 Data Mining and Search — Term Project
Team: Sangjune Kim, Gyuchan An, Minseo Kang (KAIST)

we will use one late pass. 
Sangjune Kim : second time using  
Gyuchan An, Minseo Kang : first time using 

================================================================================
HELP RECEIVED
================================================================================

We received no help from friends, classmates, lab TAs, or course staff beyond
the publicly posted lecture slides and the provided dataset.

Large language models (ChatGPT, Claude) were used for LaTeX editing, wording
refinement in the report, and minor Python syntax questions. All experimental
design, modeling decisions, numerical results, and final submissions were
produced and verified by the team.

================================================================================
COMMENT / EVALUATION
================================================================================

We found this project well-designed: having two related but methodologically
distinct tasks forced us to think carefully about when text similarity is and
is not an informative signal. The key insight — that direct query-ad cosine
barely separates clicks among displayed impressions (Cohen's d ≈ 0.11) but
strongly separates clicked from random ads (d ≈ 1.56) — only became clear
after careful EDA, and it drove the two-track design. The main difficulty was
the extreme label sparsity on Task 1 (1.11% CTR, 229 positives on validation),
which makes F1 estimates very noisy and hyper-parameter selection unreliable
without a stable proxy signal (we used Cohen's d on the internal split).

================================================================================
VALIDATION RESULTS
================================================================================

Task 1 (Click Prediction):
  External validation F1 (clicked-class) : 0.1109
  External validation AUC                : 0.713
  Bootstrap mean F1 (2000 resamples)     : 0.104
  Bootstrap 95% CI                       : [0.067, 0.144]
  Baseline provided (HistCTR > 0.01)     : 0.0436
  Relative improvement                   : +154%

Task 2 (Ad Prediction):
  Validation NDCG@3                      : 0.1538
  Query-only baseline NDCG@3             : 0.0989
  Relative improvement                   : +41%

================================================================================
HOW TO RUN
================================================================================

Requirements
------------
Python 3.10+, PyTorch >= 2.0, numpy, pandas.
Embeddings are precomputed (.npy files); sentence-transformers not required.

Submission package layout
-------------------------
This readme assumes the current directory is the submission root:

  final_report.pdf
  final_report.tex
  readme.txt
  code/task1/
  test_prediction/

Dataset layout
--------------
Place the provided dataset files in one directory and pass that directory with
--dataset-dir. Alternatively, set the DATASET_DIR environment variable.

  searchinfo.csv, searchinfo_text_embs.npy
  adinfo.csv, adinfo_title_embs.npy
  userinfo.csv
  search_stream_training.csv
  click_validation_query.csv, click_validation_answer.csv
  ad_validation_query.csv, ad_validation_answer.csv
  click_test_query.csv, ad_test_query.csv

Task 1 — reproduce click_test_answer.csv
-----------------------------------------
The standalone Task 1 code is in code/task1/. From the submission root:

  cd code/task1
  python predict_test.py --dataset-dir /path/to/datasets

This trains the content-head ensemble and applies the SKNCP-boosted pipeline
with frozen fitted weights. Output: click_test_answer.csv at the submission
root.
Runtime: ~3 minutes on GPU, ~30 minutes on CPU.

To reproduce the exact validation score (F1 0.1109) with the frozen
content-score cache included in code/task1/cs_cache.npz:

  cd code/task1
  python evaluate.py --frozen --dataset-dir /path/to/datasets

To retrain the content head from scratch and then evaluate:

  cd code/task1
  python evaluate.py --retrain --dataset-dir /path/to/datasets

Task 2 — reproduce ad_test_answer.csv
---------------------------------------
Run from the project root:

  python models/m01_interest/experiments/generate_submission.py

Trains the SOTA multi-interest model (k=5, τ=0.1, γ=0.7, γ_s=0.5) on the
full training stream, scores all 17,518 ad candidates for each of the 214
test queries, and writes the top-3 ranked AdIDs.
Output: ad_test_answer.csv at the project root.
Runtime: ~20 seconds on CPU (no GPU required).


Task 2 — reproduce validation score (NDCG@3 = 0.1538)
--------------------------------------------------------
Run from the project root:

  python models/m01_interest/experiments/baseline_eval.py

Trains the SOTA model on the training stream, then evaluates on both
validation sets. Prints a comparison table against Query-only and Random
baselines.

Expected output:
  Task B  NDCG@3 = 0.1538  (rank1=28, rank2=7, rank3=2)
  Task A  F1     = 0.0260   AUC = 0.5435
Runtime: ~20 seconds on CPU.


Task 2 — ablation study (Section 4-3 in report)
-------------------------------------------------
Run from the project root:

  python models/m01_interest/experiments/ablation_task_b.py

Removes each component one at a time (interest vectors, tiered gamma, search
signal, click signal) and sweeps k ∈ {1,3,5,10} and τ ∈ {0.01,0.1,1.0,100}
to measure Task B NDCG@3 impact against the SOTA baseline.
Runtime: ~3 minutes on CPU.


================================================================================
LABOR DIVISION
================================================================================

- Dataset audit and EDA: Sangjune Kim, Gyuchan An, Minseo Kang
- Task 1 SKNCP hypothesis, feature design, and validation: Minseo Kang
- Task 1 content head, content-strong baseline, boosted final model, and
  reproducibility cache: Gyuchan An
- Task 2 multi-interest model (design, streaming update, validation): Sangjune Kim
- Report writing, integration, LaTeX formatting, consistency checks:
  Sangjune Kim, Gyuchan An, Minseo Kang
