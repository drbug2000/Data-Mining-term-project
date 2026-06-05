"""
Check claims in the paper:
  "clicked vs non-clicked, Cohen's d ≈ 0.11, AUC ≈ 0.53"

Verifies on:
  (1) Training data  — same as _quick_check.py
  (2) Val Task-A     — 20,000 pairs, pure cos-sim(query, ad) as predictor
"""
import sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from shared.data.dataset import RecoDataset

ds = RecoDataset(ROOT / "../datasets").load()

def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

def cohens_d(a, b):
    p = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return float((a.mean() - b.mean()) / (p + 1e-12))

def binary_auc(scores, labels):
    s = np.array(scores); l = np.array(labels)
    pos = s[l == 1]; neg = s[l == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    return float((pos[:, None] > neg[None, :]).sum() + 0.5*(pos[:, None] == neg[None, :]).sum()) / (len(pos)*len(neg))

# ── (1) Training data ────────────────────────────────────────
print("=== (1) Training data: clicked vs non-clicked ===")
c_train, nc_train = [], []
for event in ds.training_stream():
    q = l2(event.search_emb)
    for ad in event.ads:
        sim = float(q @ l2(ad.ad_emb))
        if ad.is_click:
            c_train.append(sim)
        else:
            nc_train.append(sim)

c  = np.array(c_train)
nc = np.array(nc_train)
print(f"  clicked    mean={c.mean():.4f}  std={c.std():.4f}  n={len(c)}")
print(f"  non-click  mean={nc.mean():.4f}  std={nc.std():.4f}  n={len(nc)}")
print(f"  Cohen's d  = {cohens_d(c, nc):.4f}   (claimed: ~0.11)")
all_sims   = np.concatenate([c, nc])
all_labels = np.concatenate([np.ones(len(c)), np.zeros(len(nc))])
print(f"  AUC (cos-sim predictor) = {binary_auc(all_sims, all_labels):.4f}   (claimed: ~0.53)")

# ── (2) Val Task-A ───────────────────────────────────────────
print("\n=== (2) Val Task-A: clicked vs non-clicked ===")
val_pairs = ds.val_click_queries()
val_ans   = ds.val_click_answers()
labels    = val_ans["IsClick"].tolist()

scores_val = []
for (ev, ad) in val_pairs:
    q = l2(ev.search_emb)
    a = l2(ad.ad_emb)
    scores_val.append(float(q @ a))

s    = np.array(scores_val)
lbl  = np.array(labels[:len(s)])
c_v  = s[lbl == 1]
nc_v = s[lbl == 0]

print(f"  clicked    mean={c_v.mean():.4f}  std={c_v.std():.4f}  n={len(c_v)}")
print(f"  non-click  mean={nc_v.mean():.4f}  std={nc_v.std():.4f}  n={len(nc_v)}")
print(f"  Cohen's d  = {cohens_d(c_v, nc_v):.4f}   (claimed: ~0.11)")
print(f"  AUC (cos-sim predictor) = {binary_auc(scores_val, labels):.4f}   (claimed: ~0.53)")
