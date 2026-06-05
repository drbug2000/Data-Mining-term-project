"""
verify_user_history_sim.py

New claim basis:
  "max cosine similarity between a target ad and ALL of a user's
   clicked search queries + clicked ad embeddings from training history"

Compares this signal for:
  (A) Training data  : clicked ads  vs random ads     (per-event)
  (B) Val Task-B     : correct ads  vs random ads     (214 queries)
  (C) Val Task-A     : clicked ads  vs non-clicked ads (20k pairs)

For each interpretation, reports:
  - distribution stats (mean, std, n)
  - Cohen's d
  - Welch t-test p-value
  - overlap coefficient

Run from project root:
    python models/m01_interest/experiments/verify_user_history_sim.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.data.dataset import RecoDataset

DATASET_DIR = ROOT / "../datasets"
SEED = 42
N_RANDOM = 5   # random ads sampled per query


def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def overlap_coeff(a: np.ndarray, b: np.ndarray, n_bins: int = 60) -> float:
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    return float(np.sum(np.minimum(ha, hb)) * (bins[1] - bins[0]))


def sep(title: str) -> None:
    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")


def report(label_pos: str, pos: np.ndarray,
           label_neg: str, neg: np.ndarray) -> float:
    d = cohens_d(pos, neg)
    t, p = stats.ttest_ind(pos, neg, equal_var=False)
    ov   = overlap_coeff(pos, neg)
    print(f"  {label_pos:<45}  n={len(pos):>6}  mean={pos.mean():.4f}  std={pos.std():.4f}")
    print(f"  {label_neg:<45}  n={len(neg):>6}  mean={neg.mean():.4f}  std={neg.std():.4f}")
    print(f"\n  Gap              = {pos.mean() - neg.mean():+.4f}")
    print(f"  Cohen's d        = {d:.4f}")
    print(f"  Welch t          = {t:.2f},  p = {p:.2e}")
    print(f"  Overlap coeff    = {ov:.4f}  (0=no overlap, 1=identical)")
    return d


def max_sim_to_history(
    ad_emb_n: np.ndarray,          # (dim,)  L2-normalised target ad
    history: np.ndarray,           # (H, dim) L2-normalised history embeddings
) -> float:
    """max cos-sim between target ad and any history embedding."""
    if len(history) == 0:
        return 0.0
    return float((history @ ad_emb_n).max())


def mean_sim_to_history(
    ad_emb_n: np.ndarray,
    history: np.ndarray,
) -> float:
    """mean cos-sim between target ad and all history embeddings."""
    if len(history) == 0:
        return 0.0
    return float((history @ ad_emb_n).mean())


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("Loading dataset...")
    t0 = time.time()
    ds = RecoDataset(DATASET_DIR).load()
    cand_embs, cand_ids = ds.all_ad_embs()
    cand_embs_n = l2_norm(cand_embs)            # (17518, 384) pre-normalised
    id_to_idx = {aid: i for i, aid in enumerate(cand_ids)}
    print(f"  loaded in {time.time()-t0:.2f}s")

    # ------------------------------------------------------------------
    # Build user history from training stream
    # history[uid] = stack of normalised embeddings (searches + clicked ads)
    # ------------------------------------------------------------------
    print("Building user click histories from training stream...")
    t0 = time.time()

    raw_history: dict[int, list[np.ndarray]] = defaultdict(list)

    for event in ds.training_stream():
        q_n = l2_norm(event.search_emb)
        has_click = any(ad.is_click for ad in event.ads)
        if has_click:
            raw_history[event.user_id].append(q_n)   # clicked search
        for ad in event.ads:
            if ad.is_click:
                raw_history[event.user_id].append(l2_norm(ad.ad_emb))

    user_history: dict[int, np.ndarray] = {
        uid: np.stack(embs) for uid, embs in raw_history.items()
    }
    n_users_with_history = len(user_history)
    print(f"  users with click history: {n_users_with_history:,}  "
          f"({time.time()-t0:.2f}s)")

    # ------------------------------------------------------------------
    # (A) Training: clicked ads vs random ads  — TEMPORAL leave-out
    #     history for event t = only clicks strictly BEFORE event t
    #     (avoids trivial self-match: clicked ad is NOT yet in history)
    # ------------------------------------------------------------------
    sep("(A) Training: clicked ads vs RANDOM ads  [max-sim, temporal leave-out]")

    click_sims_A_max, random_sims_A_max = [], []
    click_sims_A_mean, random_sims_A_mean = [], []

    # Rebuild history incrementally: at each event we use the history
    # accumulated up to (but not including) the current event.
    incremental: dict[int, list[np.ndarray]] = defaultdict(list)

    for event in ds.training_stream():
        uid  = event.user_id
        q_n  = l2_norm(event.search_emb)
        hist_so_far = incremental[uid]  # history BEFORE this event

        for ad in event.ads:
            if ad.is_click != 1:
                continue
            if len(hist_so_far) == 0:
                continue
            hist_arr = np.stack(hist_so_far)
            a_n = l2_norm(ad.ad_emb)
            click_sims_A_max.append(max_sim_to_history(a_n, hist_arr))
            click_sims_A_mean.append(mean_sim_to_history(a_n, hist_arr))

            ridxs = rng.choice(len(cand_ids), size=N_RANDOM, replace=False)
            for ri in ridxs:
                random_sims_A_max.append(max_sim_to_history(cand_embs_n[ri], hist_arr))
                random_sims_A_mean.append(mean_sim_to_history(cand_embs_n[ri], hist_arr))

        # Update history AFTER processing this event
        has_click = any(ad.is_click for ad in event.ads)
        if has_click:
            incremental[uid].append(q_n)
        for ad in event.ads:
            if ad.is_click:
                incremental[uid].append(l2_norm(ad.ad_emb))

    print("  -- max --")
    d_A_max = report(
        "max_sim(clicked_ad,  user_history)", np.array(click_sims_A_max),
        "max_sim(random_ad,   user_history)", np.array(random_sims_A_max),
    )
    print("\n  -- mean --")
    d_A_mean = report(
        "mean_sim(clicked_ad,  user_history)", np.array(click_sims_A_mean),
        "mean_sim(random_ad,   user_history)", np.array(random_sims_A_mean),
    )

    # ------------------------------------------------------------------
    # (B) Val Task-B: correct ads vs random ads  (214 queries)
    # ------------------------------------------------------------------
    sep("(B) Val Task-B: correct ads vs RANDOM ads  [max-sim to user history]")

    val_ad_q   = ds.val_ad_queries()
    val_ad_ans = ds.val_ad_answers()

    correct_sims_B_max, random_sims_B_max = [], []
    correct_sims_B_mean, random_sims_B_mean = [], []

    for ev in val_ad_q:
        hist = user_history.get(ev.user_id)
        if hist is None:
            continue
        correct_aid = val_ad_ans.get(ev.search_id)
        if correct_aid is None or correct_aid not in id_to_idx:
            continue

        c_n = cand_embs_n[id_to_idx[correct_aid]]
        correct_sims_B_max.append(max_sim_to_history(c_n, hist))
        correct_sims_B_mean.append(mean_sim_to_history(c_n, hist))

        ridxs = rng.choice(len(cand_ids), size=N_RANDOM, replace=False)
        for ri in ridxs:
            random_sims_B_max.append(max_sim_to_history(cand_embs_n[ri], hist))
            random_sims_B_mean.append(mean_sim_to_history(cand_embs_n[ri], hist))

    print("  -- max --")
    d_B_max = report(
        "max_sim(correct_ad,  user_history)", np.array(correct_sims_B_max),
        "max_sim(random_ad,   user_history)", np.array(random_sims_B_max),
    )
    print("\n  -- mean --")
    d_B_mean = report(
        "mean_sim(correct_ad,  user_history)", np.array(correct_sims_B_mean),
        "mean_sim(random_ad,   user_history)", np.array(random_sims_B_mean),
    )

    # ------------------------------------------------------------------
    # (C) Val Task-A: clicked vs non-clicked ads  (20k pairs)
    # ------------------------------------------------------------------
    sep("(C) Val Task-A: clicked ads vs NON-CLICKED ads  [max-sim to user history]")

    val_clk_q   = ds.val_click_queries()
    val_clk_ans = ds.val_click_answers()
    labels_C    = val_clk_ans["IsClick"].tolist()

    click_sims_C_max, noclick_sims_C_max = [], []
    click_sims_C_mean, noclick_sims_C_mean = [], []

    for (ev, ad), label in zip(val_clk_q, labels_C):
        hist = user_history.get(ev.user_id)
        if hist is None:
            continue
        a_n = l2_norm(ad.ad_emb)
        sig_max  = max_sim_to_history(a_n, hist)
        sig_mean = mean_sim_to_history(a_n, hist)
        if label == 1:
            click_sims_C_max.append(sig_max)
            click_sims_C_mean.append(sig_mean)
        else:
            noclick_sims_C_max.append(sig_max)
            noclick_sims_C_mean.append(sig_mean)

    print("  -- max --")
    d_C_max = report(
        "max_sim(clicked_ad,    user_history)", np.array(click_sims_C_max),
        "max_sim(non-clicked_ad, user_history)", np.array(noclick_sims_C_max),
    )
    print("\n  -- mean --")
    d_C_mean = report(
        "mean_sim(clicked_ad,    user_history)", np.array(click_sims_C_mean),
        "mean_sim(non-clicked_ad, user_history)", np.array(noclick_sims_C_mean),
    )

    # ------------------------------------------------------------------
    # Baseline comparison: plain sim(query, ad) for same splits
    # ------------------------------------------------------------------
    sep("Baseline: plain cos-sim(query, ad) for same splits (no history)")

    plain_click_A, plain_rand_A = [], []
    for event in ds.training_stream():
        q_n = l2_norm(event.search_emb)
        for ad in event.ads:
            if ad.is_click != 1:
                continue
            plain_click_A.append(float(q_n @ l2_norm(ad.ad_emb)))
            ridxs = rng.choice(len(cand_ids), size=N_RANDOM, replace=False)
            for ri in ridxs:
                plain_rand_A.append(float(q_n @ cand_embs_n[ri]))

    d_plain = report(
        "cos_sim(query, clicked_ad)",  np.array(plain_click_A),
        "cos_sim(query, random_ad)",   np.array(plain_rand_A),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    sep("SUMMARY")
    print(f"""
  user_history = all clicked searches + clicked ads from training
  (A) n_click={len(click_sims_A_max)}, temporal leave-out
  (B) n_correct={len(correct_sims_B_max)} (val Task-B queries with history)
  (C) n_click={len(click_sims_C_max)} vs n_noclick={len(noclick_sims_C_max)} (val Task-A)

  {'':40s}  {'max-d':>8}  {'mean-d':>8}
  {'':40s}  {'------':>8}  {'-------':>8}
  (A) Training  clicked vs random           {d_A_max:>8.4f}  {d_A_mean:>8.4f}
  (B) Val Task-B correct vs random          {d_B_max:>8.4f}  {d_B_mean:>8.4f}
  (C) Val Task-A clicked vs non-clicked     {d_C_max:>8.4f}  {d_C_mean:>8.4f}
  Baseline plain cos-sim(query,ad)          {d_plain:>8.4f}  {'  N/A':>8}
""")


if __name__ == "__main__":
    main()
