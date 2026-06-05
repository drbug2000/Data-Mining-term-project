import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from shared.data.dataset import RecoDataset

ds = RecoDataset(ROOT / "../datasets").load()

def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

def cohens_d(a, b):
    p = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return float((a.mean() - b.mean()) / (p + 1e-12))

rng = np.random.default_rng(42)
cand_embs, cand_ids = ds.all_ad_embs()
cand_n = l2(cand_embs)

click_sims, noclick_sims, random_sims = [], [], []
for event in ds.training_stream():
    q = l2(event.search_emb)
    for ad in event.ads:
        sim = float(q @ l2(ad.ad_emb))
        if ad.is_click:
            click_sims.append(sim)
            for ri in rng.choice(len(cand_ids), 5, replace=False):
                random_sims.append(float(q @ cand_n[ri]))
        else:
            noclick_sims.append(sim)

c  = np.array(click_sims)
nc = np.array(noclick_sims)
r  = np.array(random_sims)

print("=== clicked vs NON-CLICKED (노출된 광고, 실제 경쟁 후보) ===")
print(f"  clicked   mean={c.mean():.4f}  std={c.std():.4f}  n={len(c)}")
print(f"  non-click mean={nc.mean():.4f}  std={nc.std():.4f}  n={len(nc)}")
print(f"  Cohen d = {cohens_d(c, nc):.4f}")
print()
print("=== clicked vs RANDOM (전체 17,518개 중 무작위) ===")
print(f"  clicked mean={c.mean():.4f}  std={c.std():.4f}  n={len(c)}")
print(f"  random  mean={r.mean():.4f}  std={r.std():.4f}  n={len(r)}")
print(f"  Cohen d = {cohens_d(c, r):.4f}")
print()
print("=== non-clicked (노출) vs RANDOM ===")
print(f"  non-click mean={nc.mean():.4f}  std={nc.std():.4f}  n={len(nc)}")
print(f"  random    mean={r.mean():.4f}  std={r.std():.4f}  n={len(r)}")
print(f"  Cohen d = {cohens_d(nc, r):.4f}")
