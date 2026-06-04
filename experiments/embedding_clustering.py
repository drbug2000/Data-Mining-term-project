"""
Embedding Clustering Analysis
==============================
1. Search embeddings clustering (sample from 276K)
2. Ad embeddings clustering (all 17,518)
3. Joint search+ad embedding analysis

Methods:
- K-Means (k=5,10,20)
- Silhouette score selection
- PCA → 2D visualization
- KNN inter-cluster analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from pathlib import Path

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

DATASET_DIR = Path(r"D:\lecture\2026_1_1-1\data mining and search\term project\datasets")
OUT_DIR = Path(__file__).parent.parent / "clustering_results"
OUT_DIR.mkdir(exist_ok=True)

SEARCH_SAMPLE = 15_000   # sample size for search embeddings (full 276K too slow for t-SNE)
RANDOM_SEED = 42
K_VALUES = [5, 10, 15, 20]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading embeddings...")
    search_info = pd.read_csv(DATASET_DIR / "searchinfo.csv")
    ad_info     = pd.read_csv(DATASET_DIR / "adinfo.csv")

    search_embs = np.load(DATASET_DIR / "searchinfo_text_embs.npy")  # (276807, 384)
    ad_embs     = np.load(DATASET_DIR / "adinfo_title_embs.npy")     # (17518, 384)

    print(f"  Search embs : {search_embs.shape}")
    print(f"  Ad embs     : {ad_embs.shape}")
    return search_info, ad_info, search_embs, ad_embs


def sample_search(search_info, search_embs, n=SEARCH_SAMPLE, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(search_embs), size=n, replace=False)
    idx.sort()
    return search_info.iloc[idx].reset_index(drop=True), search_embs[idx], idx


def pca_reduce(embs, n_components=50):
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    return pca.fit_transform(embs), pca


def pca_2d(embs, n_components=2):
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    return pca.fit_transform(embs), pca


def best_k_by_silhouette(embs_reduced, k_values=K_VALUES, sample_for_sil=3000):
    """Run K-Means for each k, return (labels_dict, sil_scores_dict, best_k)."""
    labels_dict = {}
    sil_dict    = {}
    print(f"  Testing k = {k_values} ...")
    for k in k_values:
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=5, batch_size=4096)
        labs = km.fit_predict(embs_reduced)
        labels_dict[k] = labs
        # Silhouette on a subsample for speed
        n = len(embs_reduced)
        if n > sample_for_sil:
            rng = np.random.default_rng(RANDOM_SEED)
            sel = rng.choice(n, size=sample_for_sil, replace=False)
            sil = silhouette_score(embs_reduced[sel], labs[sel], metric="euclidean")
        else:
            sil = silhouette_score(embs_reduced, labs, metric="euclidean")
        sil_dict[k] = round(sil, 4)
        print(f"    k={k:2d}  silhouette={sil:.4f}")
    best_k = max(sil_dict, key=sil_dict.get)
    return labels_dict, sil_dict, best_k


def plot_silhouette_curve(sil_dict, title, filename):
    ks   = sorted(sil_dict)
    vals = [sil_dict[k] for k in ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, vals, "o-", color="steelblue", lw=2)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(title)
    ax.set_xticks(ks)
    best_k = max(sil_dict, key=sil_dict.get)
    ax.axvline(best_k, color="tomato", ls="--", alpha=0.7, label=f"Best k={best_k}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150)
    plt.close()


def plot_2d_clusters(coords2d, labels, title, filename, alpha=0.15, s=4, cat_ids=None):
    unique = np.unique(labels)
    cmap = cm.get_cmap("tab20", len(unique))
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, lab in enumerate(unique):
        mask = labels == lab
        ax.scatter(coords2d[mask, 0], coords2d[mask, 1],
                   color=cmap(i), alpha=alpha, s=s, label=f"C{lab}")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # legend only if ≤20 clusters
    if len(unique) <= 20:
        leg = ax.legend(markerscale=3, fontsize=7, loc="upper right",
                        ncol=2, framealpha=0.6)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150)
    plt.close()


def category_distribution(labels, cat_ids, top_n=5):
    """For each cluster, show top_n most frequent category IDs."""
    df = pd.DataFrame({"cluster": labels, "cat": cat_ids})
    result = {}
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]["cat"]
        top = sub.value_counts().head(top_n)
        result[c] = top.to_dict()
    return result


def cluster_stats(labels, name="cluster"):
    """Print basic size statistics per cluster."""
    vc = pd.Series(labels).value_counts().sort_index()
    total = len(labels)
    rows = [(f"C{i}", cnt, f"{cnt/total*100:.1f}%") for i, cnt in vc.items()]
    df = pd.DataFrame(rows, columns=[name, "count", "pct"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Search Embedding Clustering
# ─────────────────────────────────────────────────────────────────────────────

def analyze_search(search_info, search_embs):
    print("\n" + "="*60)
    print("PART 1: Search Embedding Clustering")
    print("="*60)

    sinfo_s, embs_s, _ = sample_search(search_info, search_embs)
    embs_s_norm = normalize(embs_s, norm="l2")

    # PCA to 50D for clustering, 2D for visualization
    embs_50d, pca50 = pca_reduce(embs_s_norm, n_components=50)
    embs_2d,  pca2  = pca_2d(embs_s_norm, n_components=2)

    explained = pca50.explained_variance_ratio_.sum()
    print(f"  PCA 50D explains {explained*100:.1f}% variance")

    labels_dict, sil_dict, best_k = best_k_by_silhouette(embs_50d)

    print(f"\n  Best k = {best_k}  (silhouette={sil_dict[best_k]:.4f})")
    best_labels = labels_dict[best_k]

    plot_silhouette_curve(sil_dict, "Search Embeddings — Silhouette vs k",
                          "search_silhouette.png")
    plot_2d_clusters(embs_2d, best_labels,
                     f"Search Embeddings (PCA 2D) — k={best_k}",
                     "search_clusters_2d.png", alpha=0.2, s=3)

    stats_df = cluster_stats(best_labels, "cluster")
    print("\n  Cluster size distribution:")
    print(stats_df.to_string(index=False))

    if "CategoryID" in sinfo_s.columns:
        cat_dist = category_distribution(best_labels, sinfo_s["CategoryID"].values)
        print("\n  Top-3 CategoryIDs per cluster:")
        for c, top in cat_dist.items():
            top3 = list(top.items())[:3]
            print(f"    C{c}: {top3}")

    # KNN inter-cluster reachability
    print("\n  KNN (k=10) inter-cluster analysis (on 2000 samples)...")
    rng = np.random.default_rng(RANDOM_SEED)
    sel = rng.choice(len(embs_50d), size=2000, replace=False)
    nbrs = NearestNeighbors(n_neighbors=11, metric="euclidean").fit(embs_50d[sel])
    dists, inds = nbrs.kneighbors(embs_50d[sel])

    same_cluster = []
    for i, neighbors in enumerate(inds[:, 1:]):  # skip self
        lab_i = best_labels[sel[i]]
        lab_n = best_labels[sel[neighbors]]
        same_cluster.append((lab_n == lab_i).mean())
    sc_arr = np.array(same_cluster)
    print(f"  Mean same-cluster fraction among 10-NN: {sc_arr.mean():.3f} (±{sc_arr.std():.3f})")
    print(f"  → Higher = tighter clusters in embedding space")

    return best_k, sil_dict, stats_df


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — Ad Embedding Clustering
# ─────────────────────────────────────────────────────────────────────────────

def analyze_ads(ad_info, ad_embs):
    print("\n" + "="*60)
    print("PART 2: Ad Embedding Clustering")
    print("="*60)

    embs_norm = normalize(ad_embs, norm="l2")
    embs_50d, pca50 = pca_reduce(embs_norm, n_components=50)
    embs_2d,  _     = pca_2d(embs_norm, n_components=2)

    explained = pca50.explained_variance_ratio_.sum()
    print(f"  PCA 50D explains {explained*100:.1f}% variance (all {len(ad_embs)} ads)")

    labels_dict, sil_dict, best_k = best_k_by_silhouette(embs_50d)

    print(f"\n  Best k = {best_k}  (silhouette={sil_dict[best_k]:.4f})")
    best_labels = labels_dict[best_k]

    plot_silhouette_curve(sil_dict, "Ad Embeddings — Silhouette vs k",
                          "ad_silhouette.png")
    plot_2d_clusters(embs_2d, best_labels,
                     f"Ad Embeddings (PCA 2D) — k={best_k}",
                     "ad_clusters_2d.png", alpha=0.35, s=6)

    stats_df = cluster_stats(best_labels, "cluster")
    print("\n  Cluster size distribution:")
    print(stats_df.to_string(index=False))

    if "CategoryID" in ad_info.columns:
        cat_dist = category_distribution(best_labels, ad_info["CategoryID"].values)
        print("\n  Top-3 CategoryIDs per cluster:")
        for c, top in cat_dist.items():
            top3 = list(top.items())[:3]
            print(f"    C{c}: {top3}")

    # Price distribution per cluster
    if "Price" in ad_info.columns:
        price_df = pd.DataFrame({"cluster": best_labels, "price": ad_info["Price"].values})
        price_stats = price_df.groupby("cluster")["price"].agg(["mean", "median", "std"])
        price_stats.columns = ["mean_price", "median_price", "std_price"]
        price_stats = price_stats.round(1)
        print("\n  Price statistics per cluster:")
        print(price_stats.to_string())

        # Boxplot of price per cluster
        fig, ax = plt.subplots(figsize=(max(8, best_k), 5))
        groups = [price_df[price_df["cluster"] == c]["price"].values
                  for c in sorted(price_df["cluster"].unique())]
        ax.boxplot(groups, labels=[f"C{c}" for c in sorted(price_df["cluster"].unique())],
                   showfliers=False)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Price")
        ax.set_title(f"Ad Price Distribution per Cluster (k={best_k})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "ad_price_by_cluster.png", dpi=150)
        plt.close()

    # Category purity analysis
    if "CategoryID" in ad_info.columns:
        df_cat = pd.DataFrame({"cluster": best_labels, "cat": ad_info["CategoryID"].values})
        purity_per_cluster = []
        for c in sorted(df_cat["cluster"].unique()):
            sub = df_cat[df_cat["cluster"] == c]["cat"]
            purity = sub.value_counts().iloc[0] / len(sub)
            purity_per_cluster.append(purity)
        mean_purity = np.mean(purity_per_cluster)
        print(f"\n  Mean category purity per cluster: {mean_purity:.3f}")
        print(f"  → Fraction of most-common category in each cluster")

    return best_k, sil_dict, best_labels, embs_50d, embs_2d, ad_info


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — Joint Search + Ad Embedding Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_joint(search_info, search_embs, ad_info, ad_embs,
                  search_best_k, ad_best_k):
    print("\n" + "="*60)
    print("PART 3: Joint Search + Ad Embedding Analysis")
    print("="*60)

    # Sample searches for joint analysis
    sinfo_s, embs_s, _ = sample_search(search_info, search_embs, n=5000)
    all_ad_embs  = normalize(ad_embs, norm="l2")  # all 17,518
    all_srch_embs = normalize(embs_s, norm="l2")  # 5,000 sample

    # ── 3a. Shared PCA space ──
    combined = np.vstack([all_srch_embs, all_ad_embs])
    combined_2d, _ = pca_2d(combined, n_components=2)
    s_2d = combined_2d[:len(all_srch_embs)]
    a_2d = combined_2d[len(all_srch_embs):]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(s_2d[:, 0], s_2d[:, 1], c="steelblue", alpha=0.12, s=3, label="Search")
    ax.scatter(a_2d[:, 0], a_2d[:, 1], c="tomato",    alpha=0.35, s=7, label="Ad")
    ax.set_title("Joint PCA: Search vs Ad Embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "joint_pca_2d.png", dpi=150)
    plt.close()
    print("  Saved joint PCA plot.")

    # ── 3b. Joint K-Means ──
    combined_50d, _ = pca_reduce(combined, n_components=50)
    joint_k = max(search_best_k, ad_best_k)
    print(f"  Running joint K-Means with k={joint_k} ...")
    km_joint = MiniBatchKMeans(n_clusters=joint_k, random_state=RANDOM_SEED,
                               n_init=5, batch_size=4096)
    joint_labels = km_joint.fit_predict(combined_50d)

    search_labels = joint_labels[:len(all_srch_embs)]
    ad_labels     = joint_labels[len(all_srch_embs):]

    # Overlap: which clusters contain both search and ad items?
    joint_s_set = set(np.unique(search_labels))
    joint_a_set = set(np.unique(ad_labels))
    shared = joint_s_set & joint_a_set
    print(f"  Search uses {len(joint_s_set)}/{joint_k} clusters")
    print(f"  Ads use     {len(joint_a_set)}/{joint_k} clusters")
    print(f"  Shared clusters (both present): {len(shared)}/{joint_k}")

    # Project 50D to 2D for plotting
    pca2_joint = PCA(n_components=2, random_state=RANDOM_SEED).fit(combined_50d)
    combined_2d_joint = pca2_joint.transform(combined_50d)
    centroids_2d = pca2_joint.transform(km_joint.cluster_centers_)

    # Plot: color by type (search=blue, ad=red)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(combined_2d_joint[:len(all_srch_embs), 0],
               combined_2d_joint[:len(all_srch_embs), 1],
               c="steelblue", alpha=0.12, s=3, label="Search")
    ax.scatter(combined_2d_joint[len(all_srch_embs):, 0],
               combined_2d_joint[len(all_srch_embs):, 1],
               c="tomato", alpha=0.35, s=7, label="Ad")
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
               c="black", s=80, marker="x", zorder=5, label="Centroid")
    ax.set_title(f"Joint Clusters (k={joint_k}) — Search & Ad overlap")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "joint_clusters_2d.png", dpi=150)
    plt.close()

    # ── 3c. KNN: for each search, find nearest ads ──
    print("\n  KNN analysis: search → nearest ads (k=5)...")
    nn_ad = NearestNeighbors(n_neighbors=5, metric="cosine").fit(all_ad_embs)
    dists_knn, inds_knn = nn_ad.kneighbors(all_srch_embs)

    mean_dist = dists_knn.mean()
    print(f"  Mean cosine distance (search → nearest-5 ads): {mean_dist:.4f}")
    print(f"  → 0=identical, 2=opposite; lower = embeddings are aligned")

    # Category match: does nearest ad share category with search?
    if "CategoryID" in sinfo_s.columns and "CategoryID" in ad_info.columns:
        ad_cats = ad_info["CategoryID"].values
        srch_cats = sinfo_s["CategoryID"].values
        category_hits = []
        for i, neighbors in enumerate(inds_knn):
            s_cat = srch_cats[i]
            n_cats = ad_cats[neighbors]
            category_hits.append((n_cats == s_cat).mean())
        cat_hit_arr = np.array(category_hits)
        print(f"\n  Category match rate (search→NN-5 ads same category):")
        print(f"    Mean={cat_hit_arr.mean():.3f}  Std={cat_hit_arr.std():.3f}")
        random_baseline = (sinfo_s["CategoryID"].value_counts(normalize=True) *
                           ad_info["CategoryID"].value_counts(normalize=True)).sum()
        print(f"    Random baseline ~= {random_baseline:.3f}")
        lift = cat_hit_arr.mean() / (random_baseline + 1e-9)
        print(f"    Category match lift over random: {lift:.2f}×")

    # ── 3d. Cosine similarity distribution: click vs non-click ──
    print("\n  Cosine sim distribution (clicked vs shown ads in training)...")
    try:
        train = pd.read_csv(DATASET_DIR / "search_stream_training.csv")
        # Sample 20K rows for speed
        train_s = train.sample(n=min(20000, len(train)), random_state=RANDOM_SEED)
        search_emb_dict = dict(zip(
            pd.read_csv(DATASET_DIR / "searchinfo.csv")["SearchID"].tolist(),
            np.load(DATASET_DIR / "searchinfo_text_embs.npy")
        ))
        ad_emb_dict = dict(zip(
            pd.read_csv(DATASET_DIR / "adinfo.csv")["AdID"].tolist(),
            np.load(DATASET_DIR / "adinfo_title_embs.npy")
        ))
        sims_click = []
        sims_noclick = []
        for row in train_s.itertuples(index=False):
            se = search_emb_dict.get(row.SearchID)
            ae = ad_emb_dict.get(row.AdID)
            if se is None or ae is None:
                continue
            sim = float(np.dot(se, ae) / (np.linalg.norm(se) * np.linalg.norm(ae) + 1e-9))
            if row.IsClick == 1:
                sims_click.append(sim)
            else:
                sims_noclick.append(sim)

        print(f"  Clicked pairs   n={len(sims_click):,}  "
              f"mean sim={np.mean(sims_click):.4f}  std={np.std(sims_click):.4f}")
        print(f"  Non-click pairs n={len(sims_noclick):,}  "
              f"mean sim={np.mean(sims_noclick):.4f}  std={np.std(sims_noclick):.4f}")
        gap = np.mean(sims_click) - np.mean(sims_noclick)
        pool_std = np.sqrt((np.std(sims_click)**2 + np.std(sims_noclick)**2) / 2)
        cohens_d = gap / (pool_std + 1e-9)
        print(f"  Gap = {gap:.4f}  Cohen's d = {cohens_d:.3f}")

        # Histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(-0.2, 1.0, 80)
        ax.hist(sims_noclick, bins=bins, alpha=0.6, color="steelblue",
                label=f"Non-click (n={len(sims_noclick):,})", density=True)
        ax.hist(sims_click, bins=bins, alpha=0.7, color="tomato",
                label=f"Click (n={len(sims_click):,})", density=True)
        ax.axvline(np.mean(sims_noclick), color="steelblue", ls="--", lw=1.5)
        ax.axvline(np.mean(sims_click),   color="tomato",    ls="--", lw=1.5)
        ax.set_xlabel("Cosine Similarity (search, ad)")
        ax.set_ylabel("Density")
        ax.set_title(f"Click vs Non-click Cosine Similarity  (Cohen's d={cohens_d:.3f})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "joint_click_sim_distribution.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [skip cosine sim analysis: {e}]")

    return joint_labels, shared


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(search_best_k, search_sil, ad_best_k, ad_sil,
                  joint_labels, shared_clusters, joint_k):
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    print(f"""
[1] Search Embedding Clustering
    Best k = {search_best_k}  |  Silhouette = {search_sil[search_best_k]:.4f}
    Silhouette by k: { {k: search_sil[k] for k in sorted(search_sil)} }

[2] Ad Embedding Clustering
    Best k = {ad_best_k}  |  Silhouette = {ad_sil[ad_best_k]:.4f}
    Silhouette by k: { {k: ad_sil[k] for k in sorted(ad_sil)} }

[3] Joint Analysis
    Joint k used = {joint_k}
    Clusters shared by both search & ad = {len(shared_clusters)}/{joint_k}
    → Overlap indicates semantic alignment between query & ad space

Output files saved to: {OUT_DIR}
  search_silhouette.png        — Silhouette score curve for search
  search_clusters_2d.png       — PCA 2D scatter, search clusters
  ad_silhouette.png            — Silhouette score curve for ads
  ad_clusters_2d.png           — PCA 2D scatter, ad clusters
  ad_price_by_cluster.png      — Boxplot: price per ad cluster
  joint_pca_2d.png             — Shared PCA space: search vs ad
  joint_clusters_2d.png        — Joint clustering with centroids
  joint_click_sim_distribution — Click vs non-click cosine sim histogram
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    search_info, ad_info, search_embs, ad_embs = load_data()

    s_best_k, s_sil, s_stats       = analyze_search(search_info, search_embs)
    a_best_k, a_sil, _, _, _, _    = analyze_ads(ad_info, ad_embs)
    joint_k_used = max(s_best_k, a_best_k)
    j_labels, shared                = analyze_joint(
        search_info, search_embs, ad_info, ad_embs, s_best_k, a_best_k
    )

    print_summary(s_best_k, s_sil, a_best_k, a_sil, j_labels, shared, joint_k_used)
