"""
Cluster vs Category Alignment Analysis
Metrics: NMI, ARI, purity, heatmap
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (normalized_mutual_info_score,
                             adjusted_rand_score,
                             homogeneity_score, completeness_score)
from sklearn.preprocessing import normalize

DATASET_DIR = Path(r"D:\lecture\2026_1_1-1\data mining and search\term project\datasets")
OUT_DIR = Path(__file__).parent.parent / "clustering_results"
OUT_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42
BEST_K = 15
SEARCH_SAMPLE = 15_000


def load():
    search_info = pd.read_csv(DATASET_DIR / "searchinfo.csv")
    ad_info     = pd.read_csv(DATASET_DIR / "adinfo.csv")
    search_embs = np.load(DATASET_DIR / "searchinfo_text_embs.npy")
    ad_embs     = np.load(DATASET_DIR / "adinfo_title_embs.npy")
    return search_info, ad_info, search_embs, ad_embs


def cluster(embs, k, n_components=50):
    embs_n = normalize(embs, norm="l2")
    pca    = PCA(n_components=n_components, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(embs_n)
    km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED,
                         n_init=5, batch_size=4096)
    labels = km.fit_predict(reduced)
    return labels


def alignment_metrics(labels, cat_ids, name):
    # filter out CategoryID == 0 (unknown) for cleaner analysis
    mask_all  = np.ones(len(labels), dtype=bool)
    mask_known = cat_ids != 0

    for mask, tag in [(mask_all, "all cats"), (mask_known, "known cats only")]:
        l = labels[mask]
        c = cat_ids[mask]
        nmi  = normalized_mutual_info_score(c, l)
        ari  = adjusted_rand_score(c, l)
        hom  = homogeneity_score(c, l)
        comp = completeness_score(c, l)

        # cluster purity (dominant category fraction per cluster)
        df = pd.DataFrame({"cluster": l, "cat": c})
        purities = []
        for cl in sorted(df["cluster"].unique()):
            sub = df[df["cluster"] == cl]["cat"]
            purities.append(sub.value_counts().iloc[0] / len(sub))
        mean_purity = np.mean(purities)

        print(f"  [{name} | {tag}]  n={mask.sum():,}")
        print(f"    NMI         = {nmi:.4f}   (0=no align, 1=perfect)")
        print(f"    ARI         = {ari:.4f}   (0=random, 1=perfect)")
        print(f"    Homogeneity = {hom:.4f}   (each cluster = 1 category)")
        print(f"    Completeness= {comp:.4f}   (each category = 1 cluster)")
        print(f"    Mean purity = {mean_purity:.4f}   (dominant cat fraction)")
        print()
    return labels, cat_ids


def heatmap(labels, cat_ids, title, filename, topn_cats=15):
    df = pd.DataFrame({"cluster": labels, "cat": cat_ids})
    # keep only topN most frequent categories for readability
    top_cats = df["cat"].value_counts().head(topn_cats).index.tolist()
    df2 = df[df["cat"].isin(top_cats)]

    ct = pd.crosstab(df2["cluster"], df2["cat"])
    # normalize each cluster row -> fraction
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(10, topn_cats * 0.7),
                                    max(6, BEST_K * 0.45)))
    im = ax.imshow(ct_norm.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=ct_norm.values.max())
    ax.set_xticks(range(len(ct_norm.columns)))
    ax.set_xticklabels([f"Cat{c}" for c in ct_norm.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(ct_norm.index)))
    ax.set_yticklabels([f"C{i}" for i in ct_norm.index])
    ax.set_xlabel("CategoryID (top 15 most frequent)")
    ax.set_ylabel("Cluster")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Fraction within cluster")
    # annotate cells
    for i in range(ct_norm.shape[0]):
        for j in range(ct_norm.shape[1]):
            val = ct_norm.values[i, j]
            if val > 0.05:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if val < 0.5 else "white")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


def dominant_category_per_cluster(labels, cat_ids, name):
    df = pd.DataFrame({"cluster": labels, "cat": cat_ids})
    print(f"  [{name}] Dominant category per cluster:")
    print(f"  {'Cluster':>8} {'DomCat':>8} {'Purity':>8} {'Size':>6}")
    for cl in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cl]["cat"]
        vc  = sub.value_counts()
        dom = vc.index[0]
        pur = vc.iloc[0] / len(sub)
        print(f"  {'C'+str(cl):>8} {'Cat'+str(dom):>8} {pur:>8.3f} {len(sub):>6}")
    print()


def category_spread(labels, cat_ids, name):
    """How many clusters does each category spread across?"""
    df = pd.DataFrame({"cluster": labels, "cat": cat_ids})
    spread = df.groupby("cat")["cluster"].nunique()
    print(f"  [{name}] Category spread across clusters:")
    print(f"    Mean clusters per category : {spread.mean():.2f}")
    print(f"    Max clusters per category  : {spread.max()}  (Cat {spread.idxmax()})")
    print(f"    Categories in only 1 cluster: {(spread == 1).sum()}")
    print()


if __name__ == "__main__":
    print("Loading data...")
    search_info, ad_info, search_embs, ad_embs = load()

    # ── Sample search ──
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(search_embs), size=SEARCH_SAMPLE, replace=False)
    idx.sort()
    sinfo_s    = search_info.iloc[idx].reset_index(drop=True)
    embs_s     = search_embs[idx]
    search_cats = sinfo_s["CategoryID"].values

    print("\nClustering search embeddings...")
    s_labels = cluster(embs_s, BEST_K)

    print("\nClustering ad embeddings...")
    a_labels = cluster(ad_embs, BEST_K)
    ad_cats  = ad_info["CategoryID"].values

    # ── Alignment metrics ──
    print("\n" + "="*60)
    print("ALIGNMENT METRICS")
    print("="*60)
    alignment_metrics(s_labels, search_cats, "Search")
    alignment_metrics(a_labels, ad_cats, "Ad")

    # ── Dominant category per cluster ──
    print("="*60)
    print("DOMINANT CATEGORY PER CLUSTER")
    print("="*60)
    dominant_category_per_cluster(s_labels, search_cats, "Search", )
    dominant_category_per_cluster(a_labels, ad_cats, "Ad")

    # ── Category spread ──
    print("="*60)
    print("CATEGORY SPREAD")
    print("="*60)
    category_spread(s_labels, search_cats, "Search")
    category_spread(a_labels, ad_cats, "Ad")

    # ── Heatmaps ──
    print("="*60)
    print("GENERATING HEATMAPS")
    print("="*60)
    heatmap(s_labels, search_cats,
            "Search: Cluster vs Category (row-normalized)",
            "search_cluster_category_heatmap.png")
    heatmap(a_labels, ad_cats,
            "Ad: Cluster vs Category (row-normalized)",
            "ad_cluster_category_heatmap.png")

    print("\nDone. Results in:", OUT_DIR)
