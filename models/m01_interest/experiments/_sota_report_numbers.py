"""
_sota_report_numbers.py
REPORT.md 업데이트용 SOTA 수치 전부 산출.
Config: gamma=0.7, gamma_search=0.5  (tiered_gamma_eval 최적)
"""
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.data.dataset import RecoDataset
from models.m01_interest import ModelConfig, MultiInterestModel
from shared.eval.predictor import (
    evaluate_task_a, evaluate_task_b_ndcg,
    score_task_a, score_task_b, train,
)

DATASET_DIR = ROOT / "../datasets"
SEED = 42

SOTA_CFG = ModelConfig(
    k=5, alpha_search=0.01, alpha_click=0.5,
    alpha_neg=0.0, temperature=0.1,
    gamma=0.7, gamma_search=0.5, threshold=0.5,
)
HISTCTR_BASELINE = dict(task_a_f1=0.0436, task_b_ndcg=0.0211)

def sep(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

np.random.seed(SEED)
ds = RecoDataset(DATASET_DIR).load()
cand_embs, cand_ids = ds.all_ad_embs()
val_clk_q   = ds.val_click_queries()
val_clk_ans = ds.val_click_answers()
val_ad_q    = ds.val_ad_queries()
val_ad_ans  = ds.val_ad_answers()

# ── 훈련 ────────────────────────────────────────────────
sep("Training")
model = MultiInterestModel(SOTA_CFG)
t0 = time.time()
train(model, ds.training_stream())
print(f"  elapsed: {time.time()-t0:.1f}s")
print(f"  Config: {SOTA_CFG.to_dict()}")

# ── Task B ───────────────────────────────────────────────
sep("Task B: Ad Recommendation (NDCG@3)")
sc_b = score_task_b(model, val_ad_q, cand_embs)
mb   = evaluate_task_b_ndcg(sc_b, val_ad_ans, cand_ids)
rd   = mb["rank_dist"]

# query-only baseline
qonly = MultiInterestModel(ModelConfig(gamma=0.0, gamma_search=0.0))
sc_qonly = score_task_b(qonly, val_ad_q, cand_embs)
mb_qonly = evaluate_task_b_ndcg(sc_qonly, val_ad_ans, cand_ids)

# random baseline
rng = np.random.default_rng(SEED)
sc_rand = {ev.search_id: rng.random(len(cand_ids)).astype(np.float32) for ev in val_ad_q}
mb_rand = evaluate_task_b_ndcg(sc_rand, val_ad_ans, cand_ids)

print(f"\n  {'Model':<30} {'NDCG@3':>8}  rank1  rank2  rank3   >3")
print(f"  {'-'*30}  {'-'*6}  {'-----'*4}")

def taskb_row(label, m):
    r = m['rank_dist']
    print(f"  {label:<30} {m['ndcg@3']:>8.4f}  {r[1]:>5}  {r[2]:>5}  {r[3]:>5}  {r['>3']:>5}")

taskb_row("MultiInterest (SOTA)", mb)
taskb_row("Query-only (gamma=0)", mb_qonly)
taskb_row("Random", mb_rand)
print(f"  {'HistCTR baseline':<30} {HISTCTR_BASELINE['task_b_ndcg']:>8.4f}  (external)")

gain = mb['ndcg@3'] / HISTCTR_BASELINE['task_b_ndcg']
gain_qonly = mb['ndcg@3'] / mb_qonly['ndcg@3']
print(f"\n  vs HistCTR baseline : x{gain:.1f}")
print(f"  vs Query-only       : x{gain_qonly:.2f}")

# ── Task A ───────────────────────────────────────────────
sep("Task A: Click Prediction (F1, threshold sweep)")
sc_a = score_task_a(model, val_clk_q)

n_clicks = val_clk_ans["IsClick"].sum()
print(f"  label dist: click={n_clicks} ({n_clicks/len(val_clk_ans):.2%})  no-click={len(val_clk_ans)-n_clicks}")

print(f"\n  {'thr':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")

best_f1, best_thr, best_ma = 0.0, 0.5, None
for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    ma = evaluate_task_a(sc_a, val_clk_ans, thr)
    marker = " <-- best F1" if ma['f1'] > best_f1 else ""
    print(f"  {thr:>5.1f} {ma['accuracy']:>7.4f} {ma['precision']:>7.4f} "
          f"{ma['recall']:>7.4f} {ma['f1']:>7.4f} {ma['auc']:>7.4f}{marker}")
    if ma['f1'] > best_f1:
        best_f1, best_thr, best_ma = ma['f1'], thr, ma

# Always-0 / Always-1
ma0 = evaluate_task_a([0.0]*len(val_clk_q), val_clk_ans, 0.5)
ma1 = evaluate_task_a([1.0]*len(val_clk_q), val_clk_ans, 0.5)
print(f"\n  Always-0    F1={ma0['f1']:.4f}  AUC={ma0['auc']:.4f}")
print(f"  Always-1    F1={ma1['f1']:.4f}  AUC={ma1['auc']:.4f}")
print(f"  HistCTR baseline F1={HISTCTR_BASELINE['task_a_f1']}")

sep(f"Best Task A: threshold={best_thr}")
pc = best_ma['per_class']
c0 = pc.get(0, {}); c1 = pc.get(1, {})
print(f"  Accuracy={best_ma['accuracy']:.4f}  Precision={best_ma['precision']:.4f}  "
      f"Recall={best_ma['recall']:.4f}  F1={best_ma['f1']:.4f}  AUC={best_ma['auc']:.4f}")
print(f"\n  Per-class (IsClick=1 기준)")
print(f"  {'Class':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Sup':>8}")
for cls, c in sorted(pc.items()):
    print(f"  {cls:>8} {c['tp']:>6} {c['fp']:>6} {c['fn']:>6} "
          f"{c['precision']:>8.4f} {c['recall']:>8.4f} {c['f1']:>8.4f} {c['support']:>8}")

TP=c1.get('tp',0); FP=c1.get('fp',0); FN=c1.get('fn',0); TN=c0.get('tp',0)
print(f"\n  Confusion Matrix (threshold={best_thr})")
print(f"              Pred 0   Pred 1")
print(f"  True 0  :  {TN:>7}  {FP:>7}  (support={c0.get('support',0)})")
print(f"  True 1  :  {FN:>7}  {TP:>7}  (support={c1.get('support',0)})")

sep("SUMMARY FOR REPORT.md")
print(f"""
Config (SOTA)
  k={SOTA_CFG.k}, alpha_search={SOTA_CFG.alpha_search}, alpha_click={SOTA_CFG.alpha_click},
  alpha_neg={SOTA_CFG.alpha_neg}, temperature={SOTA_CFG.temperature},
  gamma={SOTA_CFG.gamma}, gamma_search={SOTA_CFG.gamma_search}, threshold={SOTA_CFG.threshold}

Task B (214 queries, 17518 candidates)
  MultiInterest NDCG@3 = {mb['ndcg@3']:.4f}
    rank1={rd[1]}, rank2={rd[2]}, rank3={rd[3]}, >3={rd['>3']}
  Query-only    NDCG@3 = {mb_qonly['ndcg@3']:.4f}
  Random        NDCG@3 = {mb_rand['ndcg@3']:.4f}
  HistCTR       NDCG@3 = {HISTCTR_BASELINE['task_b_ndcg']}  (x{gain:.1f} vs ours)

Task A (20000 pairs, CTR {n_clicks/len(val_clk_ans):.2%}, best threshold={best_thr})
  Accuracy={best_ma['accuracy']:.4f}, Precision={best_ma['precision']:.4f},
  Recall={best_ma['recall']:.4f}, F1={best_ma['f1']:.4f}, AUC={best_ma['auc']:.4f}
  AUC at thr=0.5: {evaluate_task_a(sc_a, val_clk_ans, 0.5)['auc']:.4f}
  HistCTR F1 = {HISTCTR_BASELINE['task_a_f1']}
""")
