"""
gated_ctr.py — GatedCTRModel: F1-objective log-linear(신뢰 CTR) + 클릭튜닝 content head.

원본 방법론(AI506 Task1, honest F1 ≈ 0.11)을 협업 프레임워크에 이식한 모델.

핵심 (원본 "5+content" 재현)
─────────────────────────────────────────────────────────
  s = Σ wᵢ·log(CTRᵢ)  +  w_c·z(content)  +  LO·[not logged_on]
      CTRᵢ ∈ {HistCTR, ad_ctr, ip_ctr, dev_ctr, cat_ctr}   (원본 5개 신뢰 CTR)
      content = 클릭튜닝 저차원 bilinear (Uᵀq̂)·(Vᵀâ) + b   (+ 유저 interest max-cos)
      지수 wᵢ 는 **ranking-F1 을 직접 최대화**하는 좌표상승으로 적합 (log-loss 아님 — 이
        과제의 1번 레버: MLE 0.046 vs F1-fit 0.103).  LO = log(login_boost).

선택: support 게이트 (config.use_gate=True) — warm=entity / cold=content 시그모이드 결합.
기본은 위 log-linear(원본 0.1106 재현 경로).

검증 규율 (honest 재현 핵심)
─────────────────────────────────────────────────────────
fit() 는 학습 스트림만 본다. SearchID 80/20 내부 split:
  * content head: 내부-train 학습 / 내부-val AUC early-stop,
  * F1 지수: 내부-val 에서 적합하되 **내부-val CTR feature 는 내부-train 카운트로만**
    (자기 클릭 leak 방지 — 원본 st_tr 규율).
최종 외부 점수는 full-train 카운트로 feature 생성. IPID/device 키는 SearchEvent 에
없어 searchinfo.csv·userinfo.csv 를 ds.dir 에서 직접 읽는다(shared 미수정).
외부 click_validation 라벨은 최종 F1/AUC 보고에만 쓴다.

score_pairs 는 평가의 threshold sweep(score>1-thr)이 top-k 선택이 되도록 rank-정규화
[0,1) 점수를 반환한다(순위 보존 → AUC 불변, F1 sweep 깔끔).
─────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from shared.base import BaseRecoModel
from models.m04_gated.config import GateConfig

EPS = 1e-6
CTR_KEYS = ("ad", "ip", "dv", "ca", "us", "uc")   # + user_ctr, user_cat_ctr

try:                       # content head 학습만 torch 사용(있으면). 추론은 항상 numpy.
    import torch as _torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class GatedCTRModel(BaseRecoModel):

    def __init__(self, config: GateConfig | None = None):
        super().__init__(config or GateConfig())
        self._cnt_full: dict | None = None
        self._sid2ip: dict[int, int] = {}
        self._uid2dev: dict[int, int] = {}
        self._user_protos: dict[int, np.ndarray] = {}
        self._U = self._V = None; self._bU = self._bV = None
        self._b_head = 0.0; self._scale = 10.0
        self._w: np.ndarray | None = None          # F1 지수 [5 CTR, content]
        self._con_mu = self._con_sd = 0.0
        self._lo = 0.0                              # login offset log(login_boost)
        # 선택 게이트용
        self._ent_mu = self._ent_sd = 0.0
        self._gate_t = 0.0
        self._fitted = False

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def fit(self, ds, val_pairs=None, val_answers_df=None, split_seed=None,
            split_mode="sorted") -> "GatedCTRModel":
        """학습 스트림으로 content head + F1 지수(+선택 게이트)를 적합. val_* 는 선택에 미사용.

        내부-val split (모두 leak-free group split):
          split_mode='sorted' (기본): 정렬 SearchID 뒤 20% (repo 방식).
          split_mode='random': SearchID 랜덤 20% (split_seed).
          split_mode='user'  : UserID 랜덤 20% → 그 유저 행을 내부-val 로 (미관측 유저 ≈ external
                                cold-start 모사). split_seed 로 seed 지정.
        """
        cfg = self.config
        np.random.seed(cfg.seed)
        self._lo = float(np.log(cfg.login_boost))
        t0 = time.time()

        # ── 1. 스트림 수집 ─────────────────────────────────────────────
        Q, A, Y = [], [], []
        ad_ids, cat_ids, user_ids, hist, logged, search_ids = [], [], [], [], [], []
        pos_feat, cat_match = [], []
        for ev in ds.training_stream():
            uid, sid = ev.user_id, ev.search_id
            for ad in ev.ads:
                y = int(ad.is_click)
                Q.append(ev.search_emb); A.append(ad.ad_emb); Y.append(y)
                ad_ids.append(ad.ad_id); cat_ids.append(ad.category_id); user_ids.append(uid)
                search_ids.append(sid)
                hist.append(float(ad.hist_ctr) if ad.hist_ctr is not None else 0.0)
                logged.append(float(ev.is_logged_on))
                pos_feat.append(-np.log1p(float(ad.position)))
                cat_match.append(float(ev.category_id == ad.category_id and ev.category_id != -1))

        Q = _l2_normalize(np.asarray(Q, np.float32)); A = _l2_normalize(np.asarray(A, np.float32))
        Y = np.asarray(Y, np.float32)
        ad_ids = np.asarray(ad_ids); cat_ids = np.asarray(cat_ids); user_ids = np.asarray(user_ids)
        hist = np.asarray(hist, np.float32); logged = np.asarray(logged, np.float32)
        pos_feat = np.asarray(pos_feat, np.float32); cat_match = np.asarray(cat_match, np.float32)
        print(f"[gated] train rows={len(Y):,}  global CTR={Y.mean():.4f}  ({time.time()-t0:.1f}s)")

        # ── 1b. IPID/device (CSV 직접 로드) ────────────────────────────
        sinfo = pd.read_csv(ds.dir / "searchinfo.csv"); uinfo = pd.read_csv(ds.dir / "userinfo.csv")
        self._sid2ip = dict(zip(sinfo["SearchID"].tolist(), sinfo["IPID"].tolist()))
        self._uid2dev = dict(zip(uinfo["UserID"].tolist(), uinfo["UserDeviceID"].tolist()))
        ip_ids = np.array([self._sid2ip.get(int(s), -1) for s in search_ids])
        dev_ids = np.array([self._uid2dev.get(int(u), -1) for u in user_ids.tolist()])

        # ── 2. 내부 80/20 group split (leak 방지) ──────────────────────
        sid_arr = np.asarray(search_ids)
        if split_mode == "user":
            uu = np.unique(user_ids); cutu = int(len(uu) * 0.8)
            va_users = set(np.random.RandomState(split_seed or 0).permutation(uu)[cutu:].tolist())
            mask_va = np.array([u in va_users for u in user_ids.tolist()], dtype=bool)
        elif split_mode == "random":
            uniq = np.unique(sid_arr); cut = int(len(uniq) * 0.8)
            va_sids = set(np.random.RandomState(split_seed or 0).permutation(uniq)[cut:].tolist())
            mask_va = np.array([s in va_sids for s in search_ids], dtype=bool)
        else:                                                          # sorted (repo)
            uniq = np.unique(sid_arr); cut = int(len(uniq) * 0.8)
            va_sids = set(np.sort(uniq)[cut:].tolist())
            mask_va = np.array([s in va_sids for s in search_ids], dtype=bool)
        mask_tr = ~mask_va

        # ── 3. content head: 내부-train 학습 / 내부-val AUC early-stop ──
        self._fit_head(Q[mask_tr], A[mask_tr], Y[mask_tr], Q[mask_va], A[mask_va], Y[mask_va])
        print(f"[gated] stage head  {time.time()-t0:.1f}s")
        proto_full = self._build_protos(user_ids, A, Y)
        self._user_protos = self._build_protos(user_ids[mask_tr], A[mask_tr], Y[mask_tr])
        print(f"[gated] stage protos  {time.time()-t0:.1f}s")
        con = self._content_logit(Q, A, user_ids)
        self._con_mu, self._con_sd = float(con[mask_tr].mean()), float(con[mask_tr].std() + 1e-9)
        con_z = (con - self._con_mu) / self._con_sd
        print(f"[gated] stage content  {time.time()-t0:.1f}s")

        # ── 4. 카운트 (내부-train: leak 방지 / full: 외부) ─────────────
        cnt_tr = self._count_subset(ad_ids[mask_tr], cat_ids[mask_tr], ip_ids[mask_tr], dev_ids[mask_tr],
                                    user_ids[mask_tr], Y[mask_tr])
        self._cnt_full = self._count_subset(ad_ids, cat_ids, ip_ids, dev_ids, user_ids, Y)
        print(f"[gated] stage counts  {time.time()-t0:.1f}s")

        # ── 5. 설계행렬 [CTR 통계, pos, cat-match, z(content)] + login offset ─
        X7_va = self._ctr7(ad_ids[mask_va], cat_ids[mask_va], ip_ids[mask_va], dev_ids[mask_va],
                           user_ids[mask_va], hist[mask_va], cnt_tr)
        print(f"[gated] stage ctr7-va  {time.time()-t0:.1f}s")
        X10_va = np.column_stack([X7_va, pos_feat[mask_va], cat_match[mask_va], con_z[mask_va]])
        base_va = self._lo * (1.0 - logged[mask_va])
        fit_idx = _sample_for_f1_fit(Y[mask_va], cfg.f1_fit_max_rows, cfg.seed)
        print(f"[gated] stage f1-sample  {time.time()-t0:.1f}s")
        grid = np.arange(0.0, 3.0 + cfg.f1_fit_grid_step * 0.5, cfg.f1_fit_grid_step)
        self._w = _fit_f1_exponents(
            X10_va[fit_idx], Y[mask_va][fit_idx], base=base_va[fit_idx],
            grid=grid, n_pass=cfg.f1_fit_passes,
        )
        print(f"[gated] stage f1-weights  {time.time()-t0:.1f}s")
        s_va = X10_va @ self._w + base_va
        self._sel_f1 = _best_f1_topk(s_va, Y[mask_va])[0]   # 내부-val 선택 추정 F1
        tr_users = set(user_ids[mask_tr].tolist())
        self._va_unseen_user = float(np.mean([u not in tr_users for u in user_ids[mask_va].tolist()]))
        print(f"[gated] F1-fit w={np.round(self._w,2)} "
              f"[logHist,ad,ip,dev,cat,user,userCat,negPos,catMatch,content]  "
              f"(internal-val F1={self._sel_f1:.4f}, val 미관측유저={self._va_unseen_user:.1%}, "
              f"fit_rows={len(fit_idx):,})")
        print(f"[gated] stage f1-fit  {time.time()-t0:.1f}s")

        # ── 6. (선택) support 게이트 — 내부-train z + 내부-val t ───────
        if cfg.use_gate:
            X7_tr = self._ctr7(ad_ids[mask_tr], cat_ids[mask_tr], ip_ids[mask_tr], dev_ids[mask_tr],
                               user_ids[mask_tr], hist[mask_tr], cnt_tr)
            ent_tr = np.column_stack([X7_tr, pos_feat[mask_tr], cat_match[mask_tr]]) @ self._w[:9] + self._lo * (1.0 - logged[mask_tr])
            self._ent_mu, self._ent_sd = float(ent_tr.mean()), float(ent_tr.std() + 1e-9)
            ent_va = np.column_stack([X7_va, pos_feat[mask_va], cat_match[mask_va]]) @ self._w[:9] + base_va
            ent_va_z = (ent_va - self._ent_mu) / self._ent_sd
            sup_va = self._support_vec(ad_ids[mask_va], user_ids[mask_va], cnt_tr)
            if cfg.gate_t is None:
                best_f1, best_t = -1.0, float(np.median(sup_va))
                for t in np.quantile(sup_va, np.linspace(0.05, 0.95, 19)):
                    g = _sigmoid((sup_va - t) / cfg.gate_s)
                    f1 = _best_f1_topk(g * ent_va_z + (1 - g) * con_z[mask_va], Y[mask_va])[0]
                    if f1 > best_f1:
                        best_f1, best_t = f1, float(t)
            else:
                best_t = float(cfg.gate_t)
                g = _sigmoid((sup_va - best_t) / cfg.gate_s)
                best_f1 = _best_f1_topk(g * ent_va_z + (1 - g) * con_z[mask_va], Y[mask_va])[0]
            self._gate_t = best_t
            print(f"[gated] gate t={best_t:.3f} (internal-val gated F1={best_f1:.4f})")

        self._user_protos = proto_full
        self._fitted = True
        print(f"[gated] fit 완료  {time.time()-t0:.1f}s")
        return self

    # ------------------------------------------------------------------
    # 스코어링 (full-feature — score_pairs)
    # ------------------------------------------------------------------

    def score_pairs(self, pairs) -> np.ndarray:
        """(SearchEvent, AdRecord) 쌍 → rank-정규화 [0,1) 클릭 점수 (N,)."""
        assert self._fitted, "fit() 먼저 호출"
        Q, A = self._pairs_embs(pairs)
        ad_ids = np.array([ad.ad_id for _, ad in pairs]); cat_ids = np.array([ad.category_id for _, ad in pairs])
        user_ids = np.array([ev.user_id for ev, _ in pairs])
        hist = np.array([float(ad.hist_ctr) if ad.hist_ctr is not None else 0.0 for _, ad in pairs], np.float32)
        logged = np.array([float(ev.is_logged_on) for ev, _ in pairs], np.float32)
        pos_feat = np.array([-np.log1p(float(ad.position)) for _, ad in pairs], np.float32)
        cat_match = np.array([float(ev.category_id == ad.category_id and ev.category_id != -1)
                              for ev, ad in pairs], np.float32)
        ip_ids = np.array([self._sid2ip.get(int(ev.search_id), -1) for ev, _ in pairs])
        dev_ids = np.array([self._uid2dev.get(int(ev.user_id), -1) for ev, _ in pairs])

        X7 = self._ctr7(ad_ids, cat_ids, ip_ids, dev_ids, user_ids, hist, self._cnt_full)
        con = self._content_logit(Q, A, list(user_ids))
        con_z = (con - self._con_mu) / self._con_sd
        base = self._lo * (1.0 - logged)

        if self.config.use_gate:
            ent = np.column_stack([X7, pos_feat, cat_match]) @ self._w[:9] + base
            ent_z = (ent - self._ent_mu) / self._ent_sd
            sup = self._support_vec(ad_ids, user_ids, self._cnt_full)
            g = _sigmoid((sup - self._gate_t) / self.config.gate_s)
            raw = g * ent_z + (1 - g) * con_z
        else:
            raw = np.column_stack([X7, pos_feat, cat_match, con_z]) @ self._w + base
        return _rank01(raw)

    # ------------------------------------------------------------------
    # 스코어링 (BaseRecoModel 폴백)
    # ------------------------------------------------------------------

    def score_click(self, user_id: int, query_emb: np.ndarray, ad_emb: np.ndarray) -> float:
        """Task A 폴백: AdID/HistCTR 없으면 content 경로만 (sigmoid)."""
        if not self._fitted:
            return float(_l2_normalize(query_emb) @ _l2_normalize(ad_emb))
        Q = _l2_normalize(query_emb[None, :]); A = _l2_normalize(ad_emb[None, :])
        con = self._content_logit(Q, A, [user_id])[0]
        return float(_sigmoid((con - self._con_mu) / self._con_sd))

    def score_ad_candidates(self, user_id: int, query_emb: np.ndarray,
                            candidate_embs: np.ndarray) -> np.ndarray:
        """Task B: raw query-ad retrieval + clicked-ad interest 점수 (N,).

        클릭튜닝 content head 는 Task A click prediction 에 맞춰져 있어 전체 광고 retrieval 에서는
        query-ad cosine 보다 불안정하다. Task B 는 검색어와 후보 광고의 원래 embedding 정렬을
        기본 신호로 쓰고, 유저 클릭 광고 prototype 이 있으면 개인화 보너스만 더한다.
        """
        Q = _l2_normalize(query_emb[None, :]); C = _l2_normalize(candidate_embs)
        score = C @ Q[0]
        protos = self._user_protos.get(user_id)
        beta = self.config.task_b_interest_beta
        if self._fitted and protos is not None and beta > 0:
            score = score + beta * (C @ protos.T).max(axis=1)
        return score

    def predict_click(self, user_id: int, query_emb: np.ndarray, ad_emb: np.ndarray) -> int:
        return int(self.score_click(user_id, query_emb, ad_emb) > 0.5)

    def predict_ad(self, user_id: int, query_emb: np.ndarray,
                   candidate_embs: np.ndarray, candidate_ids: list[int]) -> int:
        return candidate_ids[int(np.argmax(self.score_ad_candidates(user_id, query_emb, candidate_embs)))]

    def update_search(self, user_id: int, search_emb: np.ndarray) -> None:
        """배치 fit(ds) 학습. 스트리밍 train() 호환용 no-op."""
        return None

    def update_click(self, user_id: int, ad_emb: np.ndarray, clicked: bool) -> None:
        """배치 fit(ds) 학습. 스트리밍 train() 호환용 no-op."""
        return None

    # ------------------------------------------------------------------
    # 내부 — content head (low-rank bilinear)
    # ------------------------------------------------------------------

    def _fit_head(self, Q, A, Y, vQ, vA, vY) -> None:
        """content bi-encoder head 적합. torch 가 있으면 원본 head 로 학습(권장), 없으면 numpy fallback.
        어느 쪽이든 학습된 가중치를 numpy(U,V,bU,bV,scale,b)로 추출해 저장 → 추론은 torch 불필요."""
        if _HAS_TORCH:
            return self._fit_head_torch(Q, A, Y, vQ, vA, vY)
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)
        d, p, n = cfg.dim, cfg.proj_dim, len(Y)
        U = (rng.randn(d, p) / np.sqrt(d)).astype(np.float32)
        V = (rng.randn(d, p) / np.sqrt(d)).astype(np.float32)
        bU = np.zeros(p, np.float32); bV = np.zeros(p, np.float32)   # projection bias (원본 Linear 와 동일)
        b = 0.0; scale = 10.0
        pos_w = float((Y == 0).sum() / max(1, (Y == 1).sum()))
        sw = np.where(Y > 0.5, pos_w, 1.0).astype(np.float32)
        mU = np.zeros_like(U); vU = np.zeros_like(U); mV = np.zeros_like(V); vV = np.zeros_like(V)
        mbU = np.zeros(p); vbU = np.zeros(p); mbV = np.zeros(p); vbV = np.zeros(p)
        mb = vb = ms = vs = 0.0; b1, b2, ea, t = 0.9, 0.999, 1e-8, 0
        best_auc, best, bad = -1.0, None, 0

        def cos_parts(Qm, Am):
            up = Qm @ U + bU; vp = Am @ V + bV
            nu = np.linalg.norm(up, axis=1, keepdims=True) + 1e-8
            nv = np.linalg.norm(vp, axis=1, keepdims=True) + 1e-8
            uh = up / nu; vh = vp / nv
            c = (uh * vh).sum(1)
            return up, vp, nu[:, 0], nv[:, 0], uh, vh, c

        def adam(P, gP, m, v):
            m = b1 * m + (1 - b1) * gP; v = b2 * v + (1 - b2) * (gP * gP)
            P -= cfg.head_lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + ea)
            return P, m, v

        for ep in range(cfg.head_epochs):
            perm = rng.permutation(n)
            for i in range(0, n, cfg.head_batch):
                idx = perm[i:i + cfg.head_batch]
                Qb, Ab, yb, swb = Q[idx], A[idx], Y[idx], sw[idx]
                _, _, nu, nv, uh, vh, c = cos_parts(Qb, Ab)
                logit = scale * c + b
                g = swb * (_sigmoid(logit) - yb)               # dL/dlogit
                du = (g[:, None] * scale) * (vh - c[:, None] * uh) / nu[:, None]   # dL/dup
                dv = (g[:, None] * scale) * (uh - c[:, None] * vh) / nv[:, None]   # dL/dvp
                t += 1
                U, mU, vU = adam(U, Qb.T @ du + cfg.head_wd * U, mU, vU)
                V, mV, vV = adam(V, Ab.T @ dv + cfg.head_wd * V, mV, vV)
                bU, mbU, vbU = adam(bU, du.sum(0), mbU, vbU)
                bV, mbV, vbV = adam(bV, dv.sum(0), mbV, vbV)
                ms = b1 * ms + (1 - b1) * float((g * c).sum()); vs = b2 * vs + (1 - b2) * float((g * c).sum()) ** 2
                scale -= cfg.head_lr * (ms / (1 - b1 ** t)) / (np.sqrt(vs / (1 - b2 ** t)) + ea)
                mb = b1 * mb + (1 - b1) * float(g.sum()); vb = b2 * vb + (1 - b2) * float(g.sum()) ** 2
                b -= cfg.head_lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + ea)
            _, _, _, _, _, _, cv = cos_parts(vQ, vA)
            auc = _binary_auc(scale * cv + b, vY)
            if auc > best_auc:
                best_auc, best, bad = auc, (U.copy(), V.copy(), bU.copy(), bV.copy(), float(scale), float(b)), 0
            else:
                bad += 1
                if bad >= cfg.head_patience:
                    break
        if best is not None:
            U, V, bU, bV, scale, b = best
            print(f"[gated] content head internal-val AUC={best_auc:.4f}")
        self._U, self._V, self._bU, self._bV = U, V, bU, bV
        self._scale, self._b_head = float(scale), float(b)

    def _fit_head_torch(self, Q, A, Y, vQ, vA, vY) -> None:
        """원본 bi-encoder(Linear 384→p + L2norm + scale·cos + bias)를 torch 로 학습.
        학습 후 가중치를 numpy(U=lsᵀ, bU=ls.bias, …)로 추출 → 추론은 순수 numpy."""
        import torch
        import torch.nn as nn
        cfg = self.config
        torch.manual_seed(cfg.seed)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _Head(nn.Module):
            def __init__(s):
                super().__init__()
                s.ls = nn.Linear(cfg.dim, cfg.proj_dim); s.la = nn.Linear(cfg.dim, cfg.proj_dim)
                s.scale = nn.Parameter(torch.tensor(10.0)); s.bias = nn.Parameter(torch.tensor(0.0))

            def forward(s, q, a):
                sp = nn.functional.normalize(s.ls(q), dim=1)
                ap = nn.functional.normalize(s.la(a), dim=1)
                return s.scale * (sp * ap).sum(1) + s.bias

        m = _Head().to(dev)
        qt = torch.as_tensor(Q, device=dev); at = torch.as_tensor(A, device=dev)
        yt = torch.as_tensor(Y, device=dev)
        vqt = torch.as_tensor(vQ, device=dev); vat = torch.as_tensor(vA, device=dev)
        pos_w = torch.tensor((Y == 0).sum() / max(1, (Y == 1).sum()), device=dev)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.Adam(m.parameters(), lr=cfg.head_lr, weight_decay=cfg.head_wd)
        n, bs = len(Y), cfg.head_batch
        best_auc, best_state, bad = -1.0, None, 0
        for ep in range(cfg.head_epochs):
            m.train(); perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                lossf(m(qt[idx], at[idx]), yt[idx]).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                auc = _binary_auc(m(vqt, vat).cpu().numpy(), vY)
            if auc > best_auc:
                best_auc, best_state, bad = auc, {k: v.detach().clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= cfg.head_patience:
                    break
        if best_state is not None:
            m.load_state_dict(best_state)
        print(f"[gated] content head (torch) internal-val AUC={best_auc:.4f}")
        sd = m.state_dict()
        # Linear: y = x·Wᵀ + b  →  numpy 추론은 x·U + bU 이므로 U = Wᵀ
        self._U = sd["ls.weight"].cpu().numpy().T.astype(np.float32)
        self._V = sd["la.weight"].cpu().numpy().T.astype(np.float32)
        self._bU = sd["ls.bias"].cpu().numpy().astype(np.float32)
        self._bV = sd["la.bias"].cpu().numpy().astype(np.float32)
        self._scale = float(sd["scale"].cpu()); self._b_head = float(sd["bias"].cpu())

    def _content_logit(self, Q: np.ndarray, A: np.ndarray, user_ids) -> np.ndarray:
        up = Q @ self._U + self._bU; vp = A @ self._V + self._bV
        uh = up / (np.linalg.norm(up, axis=1, keepdims=True) + 1e-8)
        vh = vp / (np.linalg.norm(vp, axis=1, keepdims=True) + 1e-8)
        logit = self._scale * (uh * vh).sum(1) + self._b_head
        beta = self.config.interest_beta
        if beta > 0:
            bonus = np.zeros(len(A), dtype=np.float32)
            uid_arr = np.asarray(user_ids)
            order = np.argsort(uid_arr, kind="stable")
            uid_sorted = uid_arr[order]
            start = 0
            while start < len(order):
                end = start + 1
                uid = int(uid_sorted[start])
                while end < len(order) and uid_sorted[end] == uid:
                    end += 1
                protos = self._user_protos.get(int(uid))
                if protos is not None:
                    idx = order[start:end]
                    bonus[idx] = (A[idx] @ protos.T).max(axis=1).astype(np.float32)
                start = end
            logit = logit + beta * bonus
        return logit

    # ------------------------------------------------------------------
    # 내부 — CTR feature / 카운트 / support
    # ------------------------------------------------------------------

    def _count_subset(self, ad_ids, cat_ids, ip_ids, dev_ids, user_ids, Y) -> dict:
        """행 부분집합으로 ad/ip/dev/cat/user/user-cat CTR 카운트 + global."""
        uc_keys = list(zip(user_ids.tolist(), cat_ids.tolist()))
        keys = {"ad": ad_ids, "ip": ip_ids, "dv": dev_ids, "ca": cat_ids, "us": user_ids, "uc": uc_keys}
        out = {"g": float(Y.mean())}
        yl = Y.tolist()
        for name, arr in keys.items():
            s, c = {}, {}
            values = arr.tolist() if hasattr(arr, "tolist") else arr
            for k, y in zip(values, yl):
                s[k] = s.get(k, 0) + 1
                if y:
                    c[k] = c.get(k, 0) + 1
            out[name + "_s"], out[name + "_c"] = s, c
        return out

    def _ctr7(self, ad_ids, cat_ids, ip_ids, dev_ids, user_ids, hist, cnt) -> np.ndarray:
        """[log HistCTR, log ad_ctr, log ip_ctr, log dev_ctr, log cat_ctr, log user_ctr, log user_cat_ctr]."""
        g, k = cnt["g"], self.config.k_smooth
        m = len(ad_ids)
        rows = np.empty((m, 7), dtype=np.float32)
        ad_l = ad_ids.tolist(); ca_l = cat_ids.tolist(); ip_l = ip_ids.tolist(); dv_l = dev_ids.tolist(); us_l = user_ids.tolist()
        for i in range(m):
            rows[i, 0] = np.log(max(float(hist[i]), EPS))
            rows[i, 1] = np.log(_ctr(cnt["ad_s"], cnt["ad_c"], ad_l[i], g, k))
            rows[i, 2] = np.log(_ctr(cnt["ip_s"], cnt["ip_c"], ip_l[i], g, k))
            rows[i, 3] = np.log(_ctr(cnt["dv_s"], cnt["dv_c"], dv_l[i], g, k))
            rows[i, 4] = np.log(_ctr(cnt["ca_s"], cnt["ca_c"], ca_l[i], g, k))
            rows[i, 5] = np.log(_ctr(cnt["us_s"], cnt["us_c"], us_l[i], g, k))
            rows[i, 6] = np.log(_ctr(cnt["uc_s"], cnt["uc_c"], (us_l[i], ca_l[i]), g, k))
        return rows

    def _support_vec(self, ad_ids, user_ids, cnt) -> np.ndarray:
        """광고 관측량 support = log1p(ad_show). (게이트 경로에서만 사용)"""
        ad_s = cnt["ad_s"]
        return np.array([np.log1p(ad_s.get(int(a), 0)) for a in ad_ids.tolist()], dtype=np.float32)

    def _add_proto(self, user_id: int, ad_emb: np.ndarray) -> None:
        v = _l2_normalize(ad_emb.astype(np.float32))[None, :]
        cur = self._user_protos.get(user_id)
        if cur is None:
            self._user_protos[user_id] = v
        elif len(cur) < self.config.interest_cap:
            self._user_protos[user_id] = np.vstack([cur, v])

    def _build_protos(self, user_ids: np.ndarray, ad_embs: np.ndarray, clicks: np.ndarray) -> dict[int, np.ndarray]:
        protos: dict[int, np.ndarray] = {}
        old = self._user_protos
        self._user_protos = protos
        for uid, emb, clicked in zip(user_ids.tolist(), ad_embs, clicks.tolist()):
            if clicked:
                self._add_proto(int(uid), emb)
        out = self._user_protos
        self._user_protos = old
        return out

    def _pairs_embs(self, pairs):
        Q = _l2_normalize(np.asarray([ev.search_emb for ev, _ in pairs], dtype=np.float32))
        A = _l2_normalize(np.asarray([ad.ad_emb for _, ad in pairs], dtype=np.float32))
        return Q, A


# ------------------------------------------------------------------
# 모듈 공통 유틸
# ------------------------------------------------------------------

def _ctr(show: dict, click: dict, key, g: float, k: int) -> float:
    s = show.get(key, 0)
    if s == 0:
        return max(g, EPS)
    return max((click.get(key, 0) + k * g) / (s + k), EPS)


def _rank01(x: np.ndarray) -> np.ndarray:
    """순위 정규화 → [0,1). threshold sweep(score>1-thr)이 top-k 선택이 되게 한다."""
    order = np.argsort(np.argsort(x))
    return (order / max(1, len(x))).astype(np.float64)


def _best_f1_topk(scores: np.ndarray, y: np.ndarray):
    order = np.argsort(-scores)
    ys = y[order].astype(np.float64)
    tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1); P = ys.sum()
    if P == 0:
        return 0.0, 1
    f1 = 2 * tp / (k + P)
    j = int(np.argmax(f1))
    return float(f1[j]), int(k[j])


def _fit_f1_exponents(X: np.ndarray, y: np.ndarray, base=0.0, grid=None, n_pass: int = 4) -> np.ndarray:
    """좌표상승으로 ranking-F1 직접 최대화하는 비음 지수 w 적합. score = X@w + base."""
    if grid is None:
        grid = np.arange(0.0, 3.05, 0.5)
    nfeat = X.shape[1]
    w = np.ones(nfeat, dtype=np.float64)
    best = _best_f1_topk(X @ w + base, y)[0]
    for _ in range(n_pass):
        improved = False
        for j in range(nfeat):
            bj, bf = w[j], best
            for gv in grid:
                w[j] = gv
                f = _best_f1_topk(X @ w + base, y)[0]
                if f > bf:
                    bf, bj = f, gv
            w[j] = bj
            if bf > best + 1e-12:
                best, improved = bf, True
        if not improved:
            break
    return w.astype(np.float32)


def _sample_for_f1_fit(y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or len(y) <= max_rows:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    pos = np.flatnonzero(y > 0.5)
    neg = np.flatnonzero(y <= 0.5)
    n_pos = min(len(pos), max_rows // 2)
    n_neg = min(len(neg), max_rows - n_pos)
    if n_pos + n_neg < max_rows:
        extra = max_rows - (n_pos + n_neg)
        if len(pos) - n_pos >= len(neg) - n_neg:
            n_pos = min(len(pos), n_pos + extra)
        else:
            n_neg = min(len(neg), n_neg + extra)
    take = np.concatenate([
        rng.choice(pos, size=n_pos, replace=False) if n_pos else np.empty(0, dtype=int),
        rng.choice(neg, size=n_neg, replace=False) if n_neg else np.empty(0, dtype=int),
    ])
    rng.shuffle(take)
    return np.sort(take)


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    return float((np.sum(pos[:, None] > neg[None, :]) +
                  0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))
