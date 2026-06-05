"""
verify_cohens_d.py

Claim: "The cosine similarity between query embeddings and clicked
advertisement embeddings is substantially higher than that of randomly
sampled advertisements, with a large effect size (Cohen's d = 1.56)."

This script tests exactly that claim against all four interpretations:
  (A) Training data  : sim(query, clicked_ad) vs sim(query, random_ad)
  (B) Training data  : sim(query, clicked_ad) vs sim(query, non-clicked_ad)
  (C) Val Task-B     : sim(query, correct_ad) vs sim(query, random_ad)  [214 queries]
  (D) Training data  : using ALL random samples per clicked event

Run from project root:
    python models/m01_interest/experiments/verify_cohens_d.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.data.dataset import RecoDataset

DATASET_DIR = ROOT / "../datasets"
SEED = 42
N_RANDOM_PER_CLICK = 5   # random ads sampled per click event (same as hypothesis_analysis.py)


def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d: (mean_a - mean_b) / pooled_std"""
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def sep(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def stats_line(arr: np.ndarray, label: str) -> None:
    print(f"  {label:<40}  n={len(arr):>6}  mean={arr.mean():.4f}  std={arr.std():.4f}")


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("Loading dataset...")
    t0 = time.time()
    ds = RecoDataset(DATASET_DIR).load()
    cand_embs, cand_ids = ds.all_ad_embs()          # (17518, 384)
    cand_embs_n = l2_norm(cand_embs)                # pre-normalized
    id_to_idx = {aid: i for i, aid in enumerate(cand_ids)}
    print(f"  loaded in {time.time()-t0:.2f}s")

    # ------------------------------------------------------------------ #
    # (A) Training: sim(query, clicked) vs sim(query, random)             #
    # ------------------------------------------------------------------ #
    sep("(A) Training data: clicked ads vs RANDOM ads")

    click_sims_A, random_sims_A = [], []

    for event in ds.training_stream():
        q = l2_norm(event.search_emb)
        clicked_ads = [ad for ad in event.ads if ad.is_click == 1]
        if not clicked_ads:
            continue
        for ad in clicked_ads:
            a = l2_norm(ad.ad_emb)
            click_sims_A.append(float(q @ a))
            # sample N_RANDOM_PER_CLICK random ads from the full corpus
            ridxs = rng.choice(len(cand_ids), size=N_RANDOM_PER_CLICK, replace=False)
            for ri in ridxs:
                random_sims_A.append(float(q @ cand_embs_n[ri]))

    click_A  = np.array(click_sims_A)
    random_A = np.array(random_sims_A)

    stats_line(click_A,  "clicked ad sim(query, ad)")
    stats_line(random_A, "random  ad sim(query, ad)")
    d_A = cohens_d(click_A, random_A)
    t_stat, p_val = stats.ttest_ind(click_A, random_A, equal_var=False)
    print(f"\n  Gap (clicked - random)  = {click_A.mean() - random_A.mean():+.4f}")
    print(f"  Cohen's d               = {d_A:.4f}   <-- claimed: 1.56")
    print(f"  Welch t-test: t={t_stat:.2f}, p={p_val:.2e}")

    # ------------------------------------------------------------------ #
    # (B) Training: sim(query, clicked) vs sim(query, non-clicked)        #
    # ------------------------------------------------------------------ #
    sep("(B) Training data: clicked ads vs NON-CLICKED ads (exposed)")

    click_sims_B, noclick_sims_B = [], []

    for event in ds.training_stream():
        q = l2_norm(event.search_emb)
        for ad in event.ads:
            sim = float(q @ l2_norm(ad.ad_emb))
            if ad.is_click == 1:
                click_sims_B.append(sim)
            else:
                noclick_sims_B.append(sim)

    click_B   = np.array(click_sims_B)
    noclick_B = np.array(noclick_sims_B)

    stats_line(click_B,   "clicked    ad sim(query, ad)")
    stats_line(noclick_B, "non-clicked ad sim(query, ad)")
    d_B = cohens_d(click_B, noclick_B)
    t_stat_B, p_val_B = stats.ttest_ind(click_B, noclick_B, equal_var=False)
    print(f"\n  Gap (clicked - non-clicked) = {click_B.mean() - noclick_B.mean():+.4f}")
    print(f"  Cohen's d                   = {d_B:.4f}   <-- claimed: 1.56")
    print(f"  Welch t-test: t={t_stat_B:.2f}, p={p_val_B:.2e}")

    # ------------------------------------------------------------------ #
    # (C) Val Task-B: correct ads vs random ads  (214 queries)           #
    # ------------------------------------------------------------------ #
    sep("(C) Val Task-B: correct ads vs RANDOM ads (214 queries)")

    val_ad_q   = ds.val_ad_queries()
    val_ad_ans = ds.val_ad_answers()

    correct_sims_C, random_sims_C = [], []

    for ev in val_ad_q:
        correct_aid = val_ad_ans.get(ev.search_id)
        if correct_aid is None or correct_aid not in id_to_idx:
            continue
        q = l2_norm(ev.search_emb)
        correct_sims_C.append(float(q @ cand_embs_n[id_to_idx[correct_aid]]))
        ridxs = rng.choice(len(cand_ids), size=N_RANDOM_PER_CLICK, replace=False)
        for ri in ridxs:
            random_sims_C.append(float(q @ cand_embs_n[ri]))

    correct_C = np.array(correct_sims_C)
    random_C  = np.array(random_sims_C)

    stats_line(correct_C, "correct ad sim(query, ad)")
    stats_line(random_C,  "random  ad sim(query, ad)")
    d_C = cohens_d(correct_C, random_C)
    t_stat_C, p_val_C = stats.ttest_ind(correct_C, random_C, equal_var=False)
    print(f"\n  Gap (correct - random)  = {correct_C.mean() - random_C.mean():+.4f}")
    print(f"  Cohen's d               = {d_C:.4f}   <-- claimed: 1.56")
    print(f"  Welch t-test: t={t_stat_C:.2f}, p={p_val_C:.2e}")

    # ------------------------------------------------------------------ #
    # (D) Sensitivity: vary N_RANDOM_PER_CLICK for interpretation (A)    #
    # ------------------------------------------------------------------ #
    sep("(D) Sensitivity: Cohen's d vs N_random_per_click  [interpretation A]")

    print(f"  {'N_random':>10}  {'d':>8}  {'gap':>8}  {'click_mean':>12}  {'rand_mean':>12}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}")
    for n_rand in [1, 5, 20, 100]:
        r2 = np.random.default_rng(SEED)
        c_s, r_s = [], []
        for event in ds.training_stream():
            q2 = l2_norm(event.search_emb)
            for ad in event.ads:
                if ad.is_click != 1:
                    continue
                a2 = l2_norm(ad.ad_emb)
                c_s.append(float(q2 @ a2))
                ridxs2 = r2.choice(len(cand_ids), size=n_rand, replace=False)
                for ri in ridxs2:
                    r_s.append(float(q2 @ cand_embs_n[ri]))
        ca = np.array(c_s); ra = np.array(r_s)
        d_val = cohens_d(ca, ra)
        print(f"  {n_rand:>10}  {d_val:>8.4f}  {ca.mean()-ra.mean():>+8.4f}  "
              f"{ca.mean():>12.4f}  {ra.mean():>12.4f}")

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    sep("SUMMARY")
    print(f"  Claimed Cohen's d = 1.56")
    print()
    print(f"  (A) train clicked  vs random       : d = {d_A:.4f}")
    print(f"  (B) train clicked  vs non-clicked  : d = {d_B:.4f}")
    print(f"  (C) val correct    vs random        : d = {d_C:.4f}")
    print()

    if abs(d_A - 1.56) < 0.05:
        verdict_A = "MATCHES (A)"
    elif abs(d_B - 1.56) < 0.05:
        verdict_A = "MATCHES (B)"
    elif abs(d_C - 1.56) < 0.05:
        verdict_A = "MATCHES (C)"
    else:
        verdict_A = "NO MATCH in any interpretation"

    print(f"  Claim verification: {verdict_A}")
    closest = min([(abs(d_A-1.56), 'A', d_A),
                   (abs(d_B-1.56), 'B', d_B),
                   (abs(d_C-1.56), 'C', d_C)], key=lambda x: x[0])
    print(f"  Closest match: interpretation ({closest[1]}), d={closest[2]:.4f}, "
          f"delta={closest[0]:.4f}")


if __name__ == "__main__":
    main()
