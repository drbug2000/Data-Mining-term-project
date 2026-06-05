"""
Dataset loader for the AI506 ad recommendation task.

Directory layout (../datasets/ by default):
    searchinfo.csv              SearchID, UserID, IPID, IsUserLoggedOn, CategoryID
    searchinfo_text_embs.npy    (N_searches, 384)  — row i matches searchinfo row i
    adinfo.csv                  AdID, CategoryID, Price
    adinfo_title_embs.npy       (N_ads, 384)       — row i matches adinfo row i
    userinfo.csv                UserID, UserAgentID, ...
    search_stream_training.csv  SearchID, AdID, Position, HistCTR, IsClick
    ad_validation_query.csv     SearchID
    ad_validation_answer.csv    SearchID, AdID
    click_validation_query.csv  SearchID, AdID, Position, HistCTR
    click_validation_answer.csv SearchID, AdID, Position, HistCTR, IsClick
    ad_test_query.csv           SearchID
    click_test_query.csv        SearchID, AdID, Position, HistCTR

Quick start:
    from data.dataset import RecoDataset

    ds = RecoDataset("../datasets").load()

    # Training (stream ordered by SearchID — temporal proxy)
    for event in ds.training_stream():
        # event.search_emb : (384,) ndarray
        # event.user_id    : int
        # event.ads        : list[AdRecord]  (sorted by position)
        for ad in event.ads:
            # ad.ad_emb   : (384,)
            # ad.is_click : 0 or 1
            pass

    # Task A validation
    emb_matrix, ad_ids = ds.all_ad_embs()   # (17518, 384), list[int]
    for event in ds.val_ad_queries():
        predicted_ad_id = ...
    answers = ds.val_ad_answers()            # dict[SearchID -> AdID]

    # Task B validation
    for event, ad in ds.val_click_queries():
        predicted_click = ...
    answers_df = ds.val_click_answers()      # DataFrame with IsClick column
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

_DEFAULT_DIR = Path(__file__).parent.parent / "../datasets"


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class AdRecord:
    ad_id: int
    ad_emb: np.ndarray   # shape (dim,)
    position: int
    hist_ctr: float
    is_click: int        # 0 or 1 for training; -1 for test (unknown)
    category_id: int = -1
    price: float = 0.0


@dataclass
class SearchEvent:
    search_id: int
    user_id: int
    search_emb: np.ndarray   # shape (dim,)
    ads: list[AdRecord] = field(default_factory=list)
    category_id: int = -1
    is_logged_on: int = 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RecoDataset:
    """Loads and exposes all dataset splits. Call .load() before use."""

    def __init__(self, dataset_dir: str | Path = _DEFAULT_DIR):
        self.dir = Path(dataset_dir)
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "RecoDataset":
        """Load all files into memory. Returns self for chaining."""
        d = self.dir

        # --- search embeddings (row-aligned with searchinfo.csv) ---
        searchinfo = pd.read_csv(d / "searchinfo.csv")
        search_embs = np.load(d / "searchinfo_text_embs.npy")  # (N, 384)
        self._search_emb: dict[int, np.ndarray] = dict(
            zip(searchinfo["SearchID"].tolist(), search_embs)
        )
        self._search_user: dict[int, int] = dict(
            zip(searchinfo["SearchID"].tolist(), searchinfo["UserID"].tolist())
        )
        self._search_cat: dict[int, int] = dict(
            zip(searchinfo["SearchID"].tolist(), searchinfo["CategoryID"].tolist())
        )
        self._search_logged_on: dict[int, int] = dict(
            zip(searchinfo["SearchID"].tolist(),
                searchinfo["IsUserLoggedOn"].tolist())
        )

        # --- ad embeddings (row-aligned with adinfo.csv) ---
        adinfo = pd.read_csv(d / "adinfo.csv")
        ad_embs = np.load(d / "adinfo_title_embs.npy")         # (M, 384)
        self._ad_ids_ordered: list[int] = adinfo["AdID"].tolist()
        self._ad_embs_matrix: np.ndarray = ad_embs             # kept for all_ad_embs()
        self._ad_emb: dict[int, np.ndarray] = dict(
            zip(self._ad_ids_ordered, ad_embs)
        )
        self._ad_cat: dict[int, int] = dict(
            zip(adinfo["AdID"].tolist(), adinfo["CategoryID"].tolist())
        )
        self._ad_price: dict[int, float] = dict(
            zip(adinfo["AdID"].tolist(), adinfo["Price"].tolist())
        )

        # --- user info ---
        self._userinfo = pd.read_csv(d / "userinfo.csv").set_index("UserID")

        # --- training stream ---
        train = pd.read_csv(d / "search_stream_training.csv")
        # Sort by SearchID (temporal proxy), then by position within each search
        self._train = train.sort_values(["SearchID", "Position"], ignore_index=True)

        # --- validation ---
        self._val_ad_q   = pd.read_csv(d / "ad_validation_query.csv")
        self._val_ad_a   = pd.read_csv(d / "ad_validation_answer.csv")
        self._val_click_q = pd.read_csv(d / "click_validation_query.csv")
        self._val_click_a = pd.read_csv(d / "click_validation_answer.csv")

        # --- test ---
        self._test_ad_q   = pd.read_csv(d / "ad_test_query.csv")
        self._test_click_q = pd.read_csv(d / "click_test_query.csv")

        self._loaded = True
        return self

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call .load() first.")

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_search_emb(self, search_id: int) -> np.ndarray:
        return self._search_emb[search_id]

    def get_ad_emb(self, ad_id: int) -> np.ndarray:
        return self._ad_emb[ad_id]

    def get_user_id(self, search_id: int) -> int:
        return self._search_user[search_id]

    def all_ad_embs(self) -> tuple[np.ndarray, list[int]]:
        """Returns (emb_matrix, ad_ids) in adinfo.csv row order.

        emb_matrix : np.ndarray  shape (N_ads, dim)
        ad_ids     : list[int]   len N_ads, matches rows of emb_matrix
        """
        self._check_loaded()
        return self._ad_embs_matrix, self._ad_ids_ordered

    # ------------------------------------------------------------------
    # Training stream
    # ------------------------------------------------------------------

    def training_stream(self) -> Iterator[SearchEvent]:
        """Yields one SearchEvent per unique SearchID, in ascending SearchID order.

        Within each SearchEvent, ads are sorted by position (1 → 7).
        Searches or ads missing from lookup tables are silently skipped.
        """
        self._check_loaded()

        # Precompute numpy arrays for speed
        sids   = self._train["SearchID"].to_numpy()
        adids  = self._train["AdID"].to_numpy()
        pos    = self._train["Position"].to_numpy()
        hctr   = self._train["HistCTR"].to_numpy()
        clicks = self._train["IsClick"].to_numpy()

        i = 0
        n = len(sids)
        while i < n:
            sid = sids[i]
            user_id = self._search_user.get(int(sid))
            search_emb = self._search_emb.get(int(sid))

            # collect all rows for this SearchID
            j = i
            while j < n and sids[j] == sid:
                j += 1

            if user_id is None or search_emb is None:
                i = j
                continue

            ads: list[AdRecord] = []
            for k in range(i, j):
                ad_id = int(adids[k])
                ad_emb = self._ad_emb.get(ad_id)
                if ad_emb is None:
                    continue
                ads.append(AdRecord(
                    ad_id=ad_id,
                    ad_emb=ad_emb,
                    position=int(pos[k]),
                    hist_ctr=float(hctr[k]),
                    is_click=int(clicks[k]),
                    category_id=self._ad_cat.get(ad_id, -1),
                    price=float(self._ad_price.get(ad_id, 0.0)),
                ))

            yield SearchEvent(
                search_id=int(sid),
                user_id=int(user_id),
                search_emb=search_emb,
                ads=ads,
                category_id=self._search_cat.get(int(sid), -1),
                is_logged_on=int(self._search_logged_on.get(int(sid), 0)),
            )
            i = j

    # ------------------------------------------------------------------
    # Validation — Task A (Ad Recommendation)
    # ------------------------------------------------------------------

    def val_ad_queries(self) -> list[SearchEvent]:
        """214 SearchEvents for Task A validation. ads list is empty."""
        self._check_loaded()
        return self._build_search_events(self._val_ad_q["SearchID"])

    def val_ad_answers(self) -> dict[int, int]:
        """Returns dict mapping SearchID -> correct AdID."""
        self._check_loaded()
        return dict(zip(
            self._val_ad_a["SearchID"].tolist(),
            self._val_ad_a["AdID"].tolist(),
        ))

    # ------------------------------------------------------------------
    # Validation — Task B (Click Prediction)
    # ------------------------------------------------------------------

    def val_click_queries(self) -> list[tuple[SearchEvent, AdRecord]]:
        """20 000 (SearchEvent, AdRecord) pairs for Task B validation.

        AdRecord.is_click is set to -1 (unknown — check val_click_answers()).
        """
        self._check_loaded()
        return self._build_click_pairs(self._val_click_q, is_click_val=-1)

    def val_click_answers(self) -> pd.DataFrame:
        """DataFrame with columns [SearchID, AdID, Position, HistCTR, IsClick]."""
        self._check_loaded()
        return self._val_click_a.copy()

    # ------------------------------------------------------------------
    # Test — Task A
    # ------------------------------------------------------------------

    def test_ad_queries(self) -> list[SearchEvent]:
        """214 SearchEvents for Task A test. ads list is empty."""
        self._check_loaded()
        return self._build_search_events(self._test_ad_q["SearchID"])

    # ------------------------------------------------------------------
    # Test — Task B
    # ------------------------------------------------------------------

    def test_click_queries(self) -> list[tuple[SearchEvent, AdRecord]]:
        """20 000 (SearchEvent, AdRecord) pairs for Task B test."""
        self._check_loaded()
        return self._build_click_pairs(self._test_click_q, is_click_val=-1)

    # ------------------------------------------------------------------
    # Dataset summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        self._check_loaded()
        n_clicks = self._train["IsClick"].sum()
        n_total  = len(self._train)
        lines = [
            "=== RecoDataset Summary ===",
            f"  Dataset dir      : {self.dir}",
            f"  Users            : {len(self._userinfo):,}",
            f"  Unique searches  : {len(self._search_emb):,}  (emb dim={self._ad_embs_matrix.shape[1]})",
            f"  Unique ads       : {len(self._ad_emb):,}",
            f"  Training rows    : {n_total:,}  (clicks={n_clicks:,}, CTR={n_clicks/n_total:.2%})",
            f"  Val Task-A       : {len(self._val_ad_q):,} queries",
            f"  Val Task-B       : {len(self._val_click_q):,} queries",
            f"  Test Task-A      : {len(self._test_ad_q):,} queries",
            f"  Test Task-B      : {len(self._test_click_q):,} queries",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_search_events(self, search_ids) -> list[SearchEvent]:
        events = []
        for sid in search_ids:
            sid = int(sid)
            user_id   = self._search_user.get(sid)
            search_emb = self._search_emb.get(sid)
            if user_id is None or search_emb is None:
                continue
            events.append(SearchEvent(
                search_id=sid,
                user_id=user_id,
                search_emb=search_emb,
                ads=[],
                category_id=self._search_cat.get(sid, -1),
                is_logged_on=int(self._search_logged_on.get(sid, 0)),
            ))
        return events

    def _build_click_pairs(
        self, df: pd.DataFrame, is_click_val: int
    ) -> list[tuple[SearchEvent, AdRecord]]:
        pairs = []
        for row in df.itertuples(index=False):
            sid   = int(row.SearchID)
            ad_id = int(row.AdID)
            user_id    = self._search_user.get(sid)
            search_emb = self._search_emb.get(sid)
            ad_emb     = self._ad_emb.get(ad_id)
            if user_id is None or search_emb is None or ad_emb is None:
                continue
            event = SearchEvent(
                search_id=sid,
                user_id=user_id,
                search_emb=search_emb,
                ads=[],
                category_id=self._search_cat.get(sid, -1),
            )
            ad_rec = AdRecord(
                ad_id=ad_id,
                ad_emb=ad_emb,
                position=int(row.Position),
                hist_ctr=float(row.HistCTR),
                is_click=is_click_val,
                category_id=self._ad_cat.get(ad_id, -1),
                price=float(self._ad_price.get(ad_id, 0.0)),
            )
            pairs.append((event, ad_rec))
        return pairs
