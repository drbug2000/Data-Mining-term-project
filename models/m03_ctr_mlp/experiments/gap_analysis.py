"""
gap_analysis.py — Train / Val feature 분포 및 leakage 진단.

분석 항목
─────────────────────────────────────────────────────────
  1. 각 피처의 train/val 평균·표준편차 비교
  2. 각 피처와 IsClick 레이블의 Pearson 상관계수 (train vs val)
  3. 훈련 통계 피처의 train/val coverage (val에서 0 또는 prior인 샘플 비율)
  4. 클릭/비클릭 그룹별 피처 평균 비교 (train vs val)

실행: python -X utf8 experiments/gap_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.data.dataset import RecoDataset
from shared.data.graph import build_graph
from models.m02_gnn import GNNConfig, GNNModel
from models.m03_ctr_mlp import CTRConfig, CTRPredictor

DATASET_DIR = ROOT / "../datasets"
SEED = 42

FEAT_NAMES = [
    "sim_sa", "sim_ua", "sim_su",
    "log_pos", "hist_ctr", "log_srch_adcnt",
    "log_ad_show", "log_ad_click", "ad_train_ctr",
    "log_user_srch", "user_ctr", "user_cat_ctr",
    "cat_match", "cat_ctr",
    "log_price", "is_logged_on",
]

# 훈련 통계 피처 인덱스 (coverage 분석 대상)
STAT_FEAT_IDX = {
    "log_ad_show":    6,
    "log_ad_click":   7,
    "ad_train_ctr":   8,
    "log_user_srch":  9,
    "user_ctr":      10,
    "user_cat_ctr":  11,
    "cat_ctr":       13,
    "log_srch_adcnt": 5,
}


def sep(title=""):
    print(f"\n{'─'*62}")
    if title:
        print(f"  {title}")
        print(f"{'─'*62}")


def main():
    np.random.seed(SEED)

    # ── 데이터 로드 & GNN ──────────────────────────────────────────────
    print("[1] 데이터 로드 & GNN 학습...")
    ds    = RecoDataset(DATASET_DIR).load()
    graph = build_graph(ds, verbose=False, transductive=True)
    gnn   = GNNModel(GNNConfig(n_layers=2, agg_fn="mean")).fit(graph)

    # ── CTRPredictor로 통계 집계 (feature 추출용) ─────────────────────
    print("[2] 통계 집계 & feature 행렬 구성...")
    ctr = CTRPredictor(CTRConfig())
    ctr._gnn = gnn

    # 통계 직접 집계 (fit 내부 로직 재현)
    from collections import defaultdict
    user_s   = defaultdict(int); user_c   = defaultdict(int)
    u_cat_c  = defaultdict(int); u_cat_s  = defaultdict(int)
    ad_show  = defaultdict(int); ad_click = defaultdict(int)
    cat_show = defaultdict(int); cat_click= defaultdict(int)
    srch_cnt = defaultdict(int)
    total_show = total_click = 0

    for ev in ds.training_stream():
        user_s[ev.user_id] += 1
        srch_cnt[ev.search_id] = len(ev.ads)
        for ad in ev.ads:
            cid = ad.category_id
            ad_show[ad.ad_id]   += 1; cat_show[cid]       += 1
            u_cat_s[(ev.user_id, cid)] += 1; total_show  += 1
            if ad.is_click:
                user_c[ev.user_id]     += 1; ad_click[ad.ad_id] += 1
                cat_click[cid]         += 1; u_cat_c[(ev.user_id, cid)] += 1
                total_click            += 1

    ctr._user_search_cnt = dict(user_s); ctr._user_click_cnt  = dict(user_c)
    ctr._user_cat_click  = dict(u_cat_c); ctr._user_cat_show   = dict(u_cat_s)
    ctr._ad_show_cnt     = dict(ad_show); ctr._ad_click_cnt    = dict(ad_click)
    ctr._cat_show_cnt    = dict(cat_show); ctr._cat_click_cnt  = dict(cat_click)
    ctr._search_ad_cnt   = dict(srch_cnt)
    ctr._global_ctr      = total_click / max(total_show, 1)

    # ── 16d 피처 추출 함수 (통계 포함 전체 버전) ─────────────────────
    def feat16(ev, ad):
        g  = gnn
        si = g._search_id_to_idx.get(ev.search_id)
        ai = g._ad_id_to_idx.get(ad.ad_id)
        if si is None or ai is None:
            return None
        from models.m03_ctr_mlp import _unit
        h_s = _unit(g._search_repr[si])
        h_a = _unit(g._ad_feat[ai])
        ui  = g._user_id_to_idx.get(ev.user_id)
        h_u = _unit(g._user_repr[ui]) if ui is not None else np.zeros_like(h_s)

        sim_sa = float(h_s @ h_a)
        sim_ua = float(h_u @ h_a)
        sim_su = float(h_s @ h_u)
        log_pos        = float(np.log1p(ad.position))
        hist_ctr_f     = float(ad.hist_ctr) if ad.hist_ctr is not None else 0.0
        log_srch_adcnt = float(np.log1p(srch_cnt.get(ev.search_id, 1)))

        m    = CTRConfig().smooth_prior
        gctr = ctr._global_ctr
        a_show  = ad_show.get(ad.ad_id, 0)
        a_click = ad_click.get(ad.ad_id, 0)
        log_ad_show  = float(np.log1p(a_show))
        log_ad_click = float(np.log1p(a_click))
        ad_train_ctr = (a_click + m * gctr) / (a_show + m)

        ns = user_s.get(ev.user_id, 0)
        nc = user_c.get(ev.user_id, 0)
        log_user_srch = float(np.log1p(ns))
        user_ctr_f    = (nc + m * gctr) / (ns + m)
        key_uc  = (ev.user_id, ad.category_id)
        u_cat_s_v = u_cat_s.get(key_uc, 0)
        u_cat_c_v = u_cat_c.get(key_uc, 0)
        user_cat_ctr  = (u_cat_c_v + m * gctr) / (u_cat_s_v + m)

        cat_match = float(ev.category_id == ad.category_id and ev.category_id != -1)
        cs  = cat_show.get(ad.category_id, 0)
        cc  = cat_click.get(ad.category_id, 0)
        cat_ctr_f = (cc + m * gctr) / (cs + m)

        log_price    = float(np.log1p(ad.price))
        is_logged_on = float(ev.is_logged_on)

        return np.array([
            sim_sa, sim_ua, sim_su,
            log_pos, hist_ctr_f, log_srch_adcnt,
            log_ad_show, log_ad_click, ad_train_ctr,
            log_user_srch, user_ctr_f, user_cat_ctr,
            cat_match, cat_ctr_f,
            log_price, is_logged_on,
        ], dtype=np.float32)

    # 훈련 피처 행렬
    X_tr_list, y_tr_list = [], []
    for ev in ds.training_stream():
        for ad in ev.ads:
            f = feat16(ev, ad)
            if f is not None:
                X_tr_list.append(f); y_tr_list.append(float(ad.is_click))
    X_tr = np.array(X_tr_list, np.float32)
    y_tr = np.array(y_tr_list, np.float32)

    # 검증 피처 행렬
    val_pairs = ds.val_click_queries()
    val_ans   = ds.val_click_answers()
    y_val_all = val_ans["IsClick"].to_numpy(dtype=np.float32)

    X_val_list = []
    for ev, ad in val_pairs:
        f = feat16(ev, ad)
        X_val_list.append(f if f is not None else np.zeros(len(FEAT_NAMES), np.float32))
    X_val = np.array(X_val_list, np.float32)
    y_val = y_val_all[:len(val_pairs)]

    print(f"  Train: {X_tr.shape}  pos={int(y_tr.sum())}  CTR={y_tr.mean():.4f}")
    print(f"  Val  : {X_val.shape}  pos={int(y_val.sum())}  CTR={y_val.mean():.4f}")

    # ══════════════════════════════════════════════════════════════════
    sep("분석 1 — Train / Val 피처 분포 비교 (mean ± std)")
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  {'피처':<18} {'tr_mean':>9} {'tr_std':>8}   {'val_mean':>9} {'val_std':>8}   {'shift':>8}")
    print("  " + "─" * 68)
    for i, name in enumerate(FEAT_NAMES):
        tr_m, tr_s   = X_tr[:,i].mean(),  X_tr[:,i].std()
        val_m, val_s = X_val[:,i].mean(), X_val[:,i].std()
        # 정규화된 분포 이동량: |mean 차이| / train_std
        shift = abs(tr_m - val_m) / (tr_s + 1e-8)
        marker = " ◀ LARGE" if shift > 1.0 else ""
        print(f"  {name:<18} {tr_m:9.4f} {tr_s:8.4f}   {val_m:9.4f} {val_s:8.4f}   {shift:8.3f}{marker}")

    # ══════════════════════════════════════════════════════════════════
    sep("분석 2 — 피처 × IsClick Pearson 상관계수 (train vs val)")
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  {'피처':<18} {'tr_corr':>9}   {'val_corr':>9}   {'corr_drop':>10}")
    print("  " + "─" * 54)
    for i, name in enumerate(FEAT_NAMES):
        tr_corr  = float(np.corrcoef(X_tr[:,i],  y_tr)[0,1])
        val_corr = float(np.corrcoef(X_val[:,i], y_val)[0,1])
        drop     = tr_corr - val_corr
        marker   = " ◀ LEAK" if abs(tr_corr) > 0.05 and abs(drop) > 0.05 else ""
        print(f"  {name:<18} {tr_corr:9.4f}   {val_corr:9.4f}   {drop:10.4f}{marker}")

    # ══════════════════════════════════════════════════════════════════
    sep("분석 3 — 훈련 통계 피처 Val Coverage")
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  {'피처':<18} {'val=0 비율':>12}   설명")
    print("  " + "─" * 56)
    coverage_info = {
        "log_ad_show" : "val ad가 훈련에서 노출된 적 없음",
        "log_ad_click": "val ad가 훈련에서 클릭된 적 없음",
        "ad_train_ctr": "ad CTR = smoothed prior (노출 없음)",
        "log_user_srch": "val user가 훈련에 없음",
        "user_ctr"    : "user CTR = smoothed prior",
        "user_cat_ctr": "user × category = smoothed prior",
        "cat_ctr"     : "category 훈련 데이터 없음",
        "log_srch_adcnt": "search가 훈련에 없어 fallback=1",
    }
    m = CTRConfig().smooth_prior
    gctr = ctr._global_ctr
    prior_ad  = m * gctr / m          # = gctr (ad_show=0일 때)
    prior_usr = m * gctr / m

    for feat, idx in STAT_FEAT_IDX.items():
        col = X_val[:, idx]
        if feat in ("log_ad_show", "log_ad_click", "log_user_srch"):
            zero_rate = (col == 0.0).mean()
        elif feat == "ad_train_ctr":
            zero_rate = (col == prior_ad).mean()
        elif feat in ("user_ctr", "user_cat_ctr", "cat_ctr"):
            zero_rate = (col == prior_usr).mean()
        elif feat == "log_srch_adcnt":
            zero_rate = (col == np.log1p(1)).mean()
        else:
            zero_rate = 0.0
        print(f"  {feat:<18} {zero_rate:12.1%}   {coverage_info.get(feat,'')}")

    # ══════════════════════════════════════════════════════════════════
    sep("분석 4 — 클릭/비클릭 그룹별 피처 평균 (train vs val)")
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  {'피처':<18}  {'--- train ---':^25}   {'--- val ---':^25}")
    print(f"  {'':18}  {'click':>10} {'no-click':>10} {'diff':>6}   "
          f"{'click':>10} {'no-click':>10} {'diff':>6}")
    print("  " + "─" * 78)
    for i, name in enumerate(FEAT_NAMES):
        tr_c  = X_tr[y_tr==1, i].mean();   tr_n  = X_tr[y_tr==0, i].mean()
        val_c = X_val[y_val==1, i].mean(); val_n = X_val[y_val==0, i].mean()
        print(f"  {name:<18}  {tr_c:10.4f} {tr_n:10.4f} {tr_c-tr_n:6.3f}   "
              f"{val_c:10.4f} {val_n:10.4f} {val_c-val_n:6.3f}")


if __name__ == "__main__":
    main()
