"""Dataset loader for the AI506 ad recommendation task.

Directory layout (../../../datasets/ by default):
    searchinfo.csv              SearchID, UserID, IPID, IsUserLoggedOn, CategoryID
    searchinfo_text_embs.npy    (N_searches, 384)
    adinfo.csv                  AdID, CategoryID, Price
    adinfo_title_embs.npy       (N_ads, 384)
    userinfo.csv                UserID, UserAgentID, ...
    search_stream_training.csv  SearchID, AdID, Position, HistCTR, IsClick
    click_validation_query.csv  SearchID, AdID, Position, HistCTR
    click_validation_answer.csv SearchID, AdID, Position, HistCTR, IsClick
    click_test_query.csv        SearchID, AdID, Position, HistCTR
    ad_validation_query.csv     SearchID
    ad_validation_answer.csv    SearchID, AdID
    ad_test_query.csv           SearchID
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

_DEFAULT_DIR = Path(__file__).resolve().parents[3] / "../datasets"


@dataclass
class AdRecord:
    ad_id: int
    ad_emb: np.ndarray
    position: int
    hist_ctr: float
    is_click: int        # 0 or 1 for training; -1 for test
    category_id: int = -1
    price: float = 0.0


@dataclass
class SearchEvent:
    search_id: int
    user_id: int
    search_emb: np.ndarray
    ads: list[AdRecord] = field(default_factory=list)
    category_id: int = -1
    is_logged_on: int = 0


class RecoDataset:
    def __init__(self, dataset_dir: str | Path = _DEFAULT_DIR):
        self.dir = Path(dataset_dir)
        self._loaded = False

    def load(self) -> "RecoDataset":
        d = self.dir
        searchinfo = pd.read_csv(d / "searchinfo.csv")
        search_embs = np.load(d / "searchinfo_text_embs.npy")
        self._search_emb = dict(zip(searchinfo["SearchID"].tolist(), search_embs))
        self._search_user = dict(zip(searchinfo["SearchID"].tolist(), searchinfo["UserID"].tolist()))
        self._search_cat = dict(zip(searchinfo["SearchID"].tolist(), searchinfo["CategoryID"].tolist()))
        self._search_logged_on = dict(zip(searchinfo["SearchID"].tolist(), searchinfo["IsUserLoggedOn"].tolist()))

        adinfo = pd.read_csv(d / "adinfo.csv")
        ad_embs = np.load(d / "adinfo_title_embs.npy")
        self._ad_ids_ordered = adinfo["AdID"].tolist()
        self._ad_embs_matrix = ad_embs
        self._ad_emb = dict(zip(self._ad_ids_ordered, ad_embs))
        self._ad_cat = dict(zip(adinfo["AdID"].tolist(), adinfo["CategoryID"].tolist()))
        self._ad_price = dict(zip(adinfo["AdID"].tolist(), adinfo["Price"].tolist()))

        self._userinfo = pd.read_csv(d / "userinfo.csv").set_index("UserID")
        train = pd.read_csv(d / "search_stream_training.csv")
        self._train = train.sort_values(["SearchID", "Position"], ignore_index=True)

        self._val_click_q = pd.read_csv(d / "click_validation_query.csv")
        self._val_click_a = pd.read_csv(d / "click_validation_answer.csv")
        self._test_click_q = pd.read_csv(d / "click_test_query.csv")
        self._val_ad_q = pd.read_csv(d / "ad_validation_query.csv")
        self._val_ad_a = pd.read_csv(d / "ad_validation_answer.csv")
        self._test_ad_q = pd.read_csv(d / "ad_test_query.csv")
        self._loaded = True
        return self

    def training_stream(self) -> Iterator[SearchEvent]:
        sids = self._train["SearchID"].to_numpy()
        adids = self._train["AdID"].to_numpy()
        pos = self._train["Position"].to_numpy()
        hctr = self._train["HistCTR"].to_numpy()
        clicks = self._train["IsClick"].to_numpy()
        i = 0
        n = len(sids)
        while i < n:
            sid = sids[i]
            user_id = self._search_user.get(int(sid))
            search_emb = self._search_emb.get(int(sid))
            j = i
            while j < n and sids[j] == sid:
                j += 1
            if user_id is None or search_emb is None:
                i = j
                continue
            ads = []
            for k in range(i, j):
                ad_id = int(adids[k])
                ad_emb = self._ad_emb.get(ad_id)
                if ad_emb is None:
                    continue
                ads.append(AdRecord(
                    ad_id=ad_id, ad_emb=ad_emb, position=int(pos[k]),
                    hist_ctr=float(hctr[k]), is_click=int(clicks[k]),
                    category_id=self._ad_cat.get(ad_id, -1),
                    price=float(self._ad_price.get(ad_id, 0.0)),
                ))
            yield SearchEvent(
                search_id=int(sid), user_id=int(user_id), search_emb=search_emb,
                ads=ads, category_id=self._search_cat.get(int(sid), -1),
                is_logged_on=int(self._search_logged_on.get(int(sid), 0)),
            )
            i = j

    def val_click_queries(self) -> list[tuple[SearchEvent, AdRecord]]:
        return self._build_click_pairs(self._val_click_q, is_click_val=-1)

    def val_click_answers(self) -> pd.DataFrame:
        return self._val_click_a.copy()

    def test_click_queries(self) -> list[tuple[SearchEvent, AdRecord]]:
        return self._build_click_pairs(self._test_click_q, is_click_val=-1)

    def val_ad_queries(self) -> list[SearchEvent]:
        return self._build_search_events(self._val_ad_q["SearchID"])

    def val_ad_answers(self) -> dict[int, int]:
        return dict(zip(self._val_ad_a["SearchID"].tolist(), self._val_ad_a["AdID"].tolist()))

    def test_ad_queries(self) -> list[SearchEvent]:
        return self._build_search_events(self._test_ad_q["SearchID"])

    def all_ad_embs(self) -> tuple[np.ndarray, list[int]]:
        return self._ad_embs_matrix, self._ad_ids_ordered

    def _build_search_events(self, search_ids) -> list[SearchEvent]:
        events = []
        for sid in search_ids:
            sid = int(sid)
            user_id = self._search_user.get(sid)
            search_emb = self._search_emb.get(sid)
            if user_id is None or search_emb is None:
                continue
            events.append(SearchEvent(
                search_id=sid, user_id=user_id, search_emb=search_emb,
                ads=[], category_id=self._search_cat.get(sid, -1),
                is_logged_on=int(self._search_logged_on.get(sid, 0)),
            ))
        return events

    def _build_click_pairs(self, df: pd.DataFrame, is_click_val: int) -> list[tuple[SearchEvent, AdRecord]]:
        pairs = []
        for row in df.itertuples(index=False):
            sid = int(row.SearchID)
            ad_id = int(row.AdID)
            user_id = self._search_user.get(sid)
            search_emb = self._search_emb.get(sid)
            ad_emb = self._ad_emb.get(ad_id)
            if user_id is None or search_emb is None or ad_emb is None:
                continue
            event = SearchEvent(
                search_id=sid, user_id=user_id, search_emb=search_emb,
                ads=[], category_id=self._search_cat.get(sid, -1),
                is_logged_on=int(self._search_logged_on.get(sid, 0)),
            )
            ad_rec = AdRecord(
                ad_id=ad_id, ad_emb=ad_emb, position=int(row.Position),
                hist_ctr=float(row.HistCTR), is_click=is_click_val,
                category_id=self._ad_cat.get(ad_id, -1),
                price=float(self._ad_price.get(ad_id, 0.0)),
            )
            pairs.append((event, ad_rec))
        return pairs
