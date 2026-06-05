"""
ctr_mlp.py — Task A CTR 예측 MLP.

아키텍처
─────────────────────────────────────────────────────────
  Input features (14d, user embedding 제거 + 훈련 통계 포함 → z-score 정규화):
    ── GNN 코사인 유사도 (1d) ──────────────────────────
    sim(h_s, h_a)          query-ad 코사인 유사도  (user repr 제외)
    ── 노출/위치 (3d) ──────────────────────────────────
    log1p(position)        광고 노출 위치 (position bias)
    hist_ctr               광고의 과거 CTR (CSV 제공)
    log1p(search_ad_cnt)   검색의 동시 노출 광고 수
    ── 광고 통계 (3d, Bayesian smoothing) ──────────────
    log1p(ad_show_cnt)     훈련 내 ad 총 노출 수
    log1p(ad_click_cnt)    훈련 내 ad 총 클릭 수
    ad_train_ctr_smooth    훈련 내 ad CTR (pseudo-count smoothing)
    ── 유저 통계 (3d, Bayesian smoothing) ──────────────
    log1p(user_search_cnt) 유저 검색 횟수
    user_ctr_smooth        유저 전체 클릭률
    user_cat_ctr_smooth    유저의 해당 카테고리 클릭률
    ── 카테고리 (2d, Bayesian smoothing) ───────────────
    category_match         search.CategoryID == ad.CategoryID
    cat_ctr_smooth         카테고리 전체 CTR
    ── 광고 속성 (2d) ──────────────────────────────────
    log1p(price)           광고 가격 (log-scaled)
    is_logged_on           검색 시 유저 로그인 여부

  제거된 피처: sim(h_u, h_a), sim(h_s, h_u)
    → val의 76%가 cold-start user라 h_u 신호가 noise에 가까움

  z-score 정규화: 훈련 셋의 mean/std로 표준화 (std=0이면 1로 대체)

  MLP: Linear(16, 64) → ReLU → Dropout(0.4)
       Linear(64, 32)  → ReLU → Dropout(0.4)
       Linear(32,  1)  → Sigmoid

정규화:
  - AdamW (L2 weight decay λ=1e-3)
  - Dropout 0.4
  - Early stopping: val F1 기준 patience=5 에폭 (best weight 복원)

손실: Focal Loss  FL = -(1-p_t)^γ * log(p_t)
  backward (안정 근사): dz = (1-p_t)^γ * (p-y) / B
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class CTRConfig:
    hidden_dim:   int   = 64     # 128→64: 데이터 규모 대비 용량 축소
    n_epochs:     int   = 50     # early stopping이 실제로 멈추므로 넉넉하게
    batch_size:   int   = 512
    lr:           float = 1e-3
    dropout:      float = 0.4    # 0.2→0.4: 정규화 강화
    weight_decay: float = 1e-3   # AdamW L2 regularization
    focal_gamma:  float = 2.0
    smooth_prior: int   = 10     # Bayesian smoothing pseudo-count
    patience:     int   = 5      # early stopping: val F1 기준
    seed:         int   = 42


class CTRPredictor:

    def __init__(self, config: CTRConfig | None = None) -> None:
        self.config   = config or CTRConfig()
        self._gnn     = None
        self._fitted  = False
        self.W1 = self.b1 = None
        self.W2 = self.b2 = None
        self.W3 = self.b3 = None

        # 훈련 통계 (fit 후 채워짐)
        self._user_search_cnt: dict[int, int]   = {}
        self._user_click_cnt:  dict[int, int]   = {}
        self._user_cat_click:  dict[tuple, int] = {}
        self._user_cat_show:   dict[tuple, int] = {}
        self._ad_show_cnt:     dict[int, int]   = {}
        self._ad_click_cnt:    dict[int, int]   = {}
        self._cat_show_cnt:    dict[int, int]   = {}
        self._cat_click_cnt:   dict[int, int]   = {}
        self._search_ad_cnt:   dict[int, int]   = {}
        self._global_ctr:      float            = 0.0

        # z-score 정규화 파라미터
        self._feat_mean: np.ndarray | None = None
        self._feat_std:  np.ndarray | None = None

    # ── 학습 ───────────────────────────────────────────────────────────────

    def fit(self,
            ds,
            gnn_model,
            val_pairs=None,          # list[(SearchEvent, AdRecord)] — 선택
            val_answers_df=None,     # DataFrame with "IsClick" column — 선택
            ) -> "CTRPredictor":
        cfg = self.config
        np.random.seed(cfg.seed)
        self._gnn = gnn_model

        # ── feature 추출 (raw) ───────────────────────────────────────────
        print("[CTR] feature 추출...")
        X_list, y_list = [], []
        for ev in ds.training_stream():
            for ad in ev.ads:
                f = self._feat_raw(ev, ad)
                if f is not None:
                    X_list.append(f)
                    y_list.append(float(ad.is_click))

        X_raw = np.array(X_list, dtype=np.float32)
        y     = np.array(y_list,  dtype=np.float32)
        N, d_in = X_raw.shape
        n_pos   = int(y.sum())
        print(f"  N={N:,}  d_in={d_in}  CTR={n_pos/N:.4f}")

        # ── z-score 정규화 파라미터 계산 (훈련 셋 기준) ──────────────────
        self._feat_mean = X_raw.mean(axis=0)
        std             = X_raw.std(axis=0)
        self._feat_std  = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        X = (X_raw - self._feat_mean) / self._feat_std

        # val feature 행렬을 미리 구성 (checkpoint 평가에 재사용)
        X_val, y_val = None, None
        if val_pairs is not None and val_answers_df is not None:
            vf_list = []
            for ev, ad in val_pairs:
                f = self._feat_raw(ev, ad)
                vf_list.append(
                    f if f is not None else np.zeros(d_in, np.float32))
            X_val_raw = np.array(vf_list, np.float32)
            X_val     = (X_val_raw - self._feat_mean) / self._feat_std
            y_val     = val_answers_df["IsClick"].to_numpy(
                dtype=np.float32)[: len(val_pairs)]

        # ── MLP 초기화 (He init) ─────────────────────────────────────────
        h, h2 = cfg.hidden_dim, cfg.hidden_dim // 2
        def he(a, b):
            return (np.random.randn(a, b) * np.sqrt(2 / a)).astype(np.float32)

        self.W1 = he(d_in, h);  self.b1 = np.zeros(h,  np.float32)
        self.W2 = he(h,   h2);  self.b2 = np.zeros(h2, np.float32)
        self.W3 = he(h2,   1);  self.b3 = np.zeros(1,  np.float32)

        params     = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        # weight 행렬만 decay 적용 (bias 제외)
        decay_mask = [True, False, True, False, True, False]
        m_adam     = [np.zeros_like(p) for p in params]
        v_adam     = [np.zeros_like(p) for p in params]
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        t_adam = 0

        has_val = X_val is not None
        hdr = (f"  {'ep':>3}  {'loss':>8}  "
               f"{'tr_F1':>7} {'tr_AUC':>7}  "
               + (f"{'val_F1':>7} {'val_AUC':>7}  {'ES':>4}" if has_val else ""))
        print(f"[CTR] 학습 (max_epochs={cfg.n_epochs}, lr={cfg.lr}, "
              f"wd={cfg.weight_decay}, dropout={cfg.dropout}, "
              f"focal_γ={cfg.focal_gamma}, patience={cfg.patience})")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        t0 = time.time()

        # Early stopping 상태
        best_val_f1   = -1.0
        best_weights  = None
        patience_left = cfg.patience

        for epoch in range(cfg.n_epochs):
            perm    = np.random.permutation(N)
            ep_loss = 0.0

            for i in range(0, N, cfg.batch_size):
                idx = perm[i: i + cfg.batch_size]
                B   = len(idx)
                Xb  = X[idx]
                yb  = y[idx, None]

                # Forward
                z1 = Xb @ self.W1 + self.b1;  a1 = np.maximum(0, z1)
                if cfg.dropout > 0:
                    dm1 = (np.random.rand(*a1.shape) >
                           cfg.dropout).astype(np.float32) / (1 - cfg.dropout)
                    a1 = a1 * dm1
                z2 = a1 @ self.W2 + self.b2;  a2 = np.maximum(0, z2)
                if cfg.dropout > 0:
                    dm2 = (np.random.rand(*a2.shape) >
                           cfg.dropout).astype(np.float32) / (1 - cfg.dropout)
                    a2 = a2 * dm2
                z3 = a2 @ self.W3 + self.b3
                p  = _sigmoid(z3)

                # Focal Loss
                p_t  = p * yb + (1 - p) * (1 - yb)
                w_fl = (1 - p_t) ** cfg.focal_gamma
                loss = -(w_fl * np.log(p_t + 1e-8)).mean()
                ep_loss += float(loss) * B

                # Backward
                dz3 = w_fl * (p - yb) / B
                dW3 = a2.T @ dz3;  db3 = dz3.sum(0)
                da2 = dz3 @ self.W3.T
                if cfg.dropout > 0: da2 = da2 * dm2
                dz2 = da2 * (z2 > 0)
                dW2 = a1.T @ dz2;  db2 = dz2.sum(0)
                da1 = dz2 @ self.W2.T
                if cfg.dropout > 0: da1 = da1 * dm1
                dz1 = da1 * (z1 > 0)
                dW1 = Xb.T @ dz1;  db1 = dz1.sum(0)

                grads = [dW1, db1, dW2, db2, dW3, db3]
                t_adam += 1
                for pi, (p_, g_, do_decay) in enumerate(
                        zip(params, grads, decay_mask)):
                    # AdamW: weight decay를 gradient update 전에 직접 적용
                    if do_decay and cfg.weight_decay > 0:
                        p_ *= (1 - cfg.lr * cfg.weight_decay)
                    m_adam[pi] = beta1 * m_adam[pi] + (1 - beta1) * g_
                    v_adam[pi] = beta2 * v_adam[pi] + (1 - beta2) * g_ * g_
                    mh = m_adam[pi] / (1 - beta1 ** t_adam)
                    vh = v_adam[pi] / (1 - beta2 ** t_adam)
                    p_ -= cfg.lr * mh / (np.sqrt(vh) + eps_adam)

            # ── 매 epoch 평가 (dropout 없이) ─────────────────────────────
            sc_tr  = self._forward(X)
            tr_f1  = _best_f1(sc_tr, y)
            tr_auc = _binary_auc(sc_tr, y)

            if has_val:
                sc_val  = self._forward(X_val)
                val_f1  = _best_f1(sc_val, y_val)
                val_auc = _binary_auc(sc_val, y_val)

                # Early stopping 업데이트
                if val_f1 > best_val_f1 + 1e-6:
                    best_val_f1  = val_f1
                    best_weights = [p.copy() for p in params]
                    patience_left = cfg.patience
                    es_tag = "✓"
                else:
                    patience_left -= 1
                    es_tag = f"{patience_left}"

                print(f"  {epoch+1:3d}  {ep_loss/N:8.5f}  "
                      f"{tr_f1:7.4f} {tr_auc:7.4f}  "
                      f"{val_f1:7.4f} {val_auc:7.4f}  {es_tag:>4}  "
                      f"t={time.time()-t0:.0f}s")

                if patience_left <= 0:
                    print(f"[CTR] Early stopping at epoch {epoch+1}  "
                          f"(best val_F1={best_val_f1:.4f})")
                    break
            else:
                print(f"  {epoch+1:3d}  {ep_loss/N:8.5f}  "
                      f"{tr_f1:7.4f} {tr_auc:7.4f}  "
                      f"t={time.time()-t0:.0f}s")

        # Best weight 복원 (val 데이터가 있고 저장된 경우)
        if best_weights is not None:
            for p_, best in zip(params, best_weights):
                p_[:] = best
            print(f"[CTR] Best weight 복원 (val_F1={best_val_f1:.4f})")

        self._fitted = True
        print(f"[CTR] 완료  {time.time()-t0:.1f}s")
        return self

    # ── 추론 ───────────────────────────────────────────────────────────────

    def score_pairs(self, pairs) -> np.ndarray:
        assert self._fitted
        d_in = self.W1.shape[0]
        feats, n_oov = [], 0
        for ev, ad in pairs:
            f = self._feat_raw(ev, ad)
            if f is None:
                n_oov += 1
                f = np.zeros(d_in, np.float32)
            feats.append(f)
        if n_oov:
            print(f"[CTR] OOV {n_oov}/{len(feats)} 샘플 → zero vector 대체")
        X_raw = np.array(feats, np.float32)
        X     = (X_raw - self._feat_mean) / self._feat_std
        return self._forward(X)

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """dropout 없는 순전파. (N,) score 반환."""
        a1 = np.maximum(0, X  @ self.W1 + self.b1)
        a2 = np.maximum(0, a1 @ self.W2 + self.b2)
        return _sigmoid(a2 @ self.W3 + self.b3)[:, 0]

    def _feat_raw(self, ev, ad) -> np.ndarray | None:
        """SearchEvent × AdRecord → 9d raw (정규화 전) feature vector.

        최고 baseline(10d)에서 hist_ctr 제거한 버전.
        """
        g  = self._gnn
        si = g._search_id_to_idx.get(ev.search_id)
        ai = g._ad_id_to_idx.get(ad.ad_id)
        if si is None or ai is None:
            return None

        # ── 코사인 유사도 (1) ─────────────────────────────────────────────
        h_s = _unit(g._search_repr[si])
        h_a = _unit(g._ad_feat[ai])
        sim_sa = float(h_s @ h_a)

        # ── 노출/위치 (3, hist_ctr 제외) ─────────────────────────────────
        pos     = ad.position
        log_pos = float(np.log1p(pos))
        inv_pos = float(1.0 / (1.0 + pos))
        is_top1 = float(pos == 1)

        # ── 카테고리 (3) ──────────────────────────────────────────────────
        cat_match     = float(ev.category_id == ad.category_id
                              and ev.category_id != -1)
        search_cat_id = float(ev.category_id)
        ad_cat_id     = float(ad.category_id)

        # ── 광고 속성 (2) ─────────────────────────────────────────────────
        log_price    = float(np.log1p(ad.price))
        is_logged_on = float(ev.is_logged_on)

        return np.array([
            sim_sa,
            log_pos, inv_pos, is_top1,
            cat_match, search_cat_id, ad_cat_id,
            log_price, is_logged_on,
        ], dtype=np.float32)  # (9,)  hist_ctr 제외


# ── 모듈 유틸 ──────────────────────────────────────────────────────────────

def _unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x) + eps)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    return float(
        (np.sum(pos[:, None] > neg[None, :]) +
         0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))
    )

def _best_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    """0.05 간격 threshold sweep으로 최고 F1 반환."""
    best = 0.0
    for thr in np.arange(0.05, 0.96, 0.05):
        preds = (scores > 1 - thr).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        if f1 > best:
            best = f1
    return best
