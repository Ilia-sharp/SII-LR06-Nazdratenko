from __future__ import annotations

import pandas as pd

from src.covisitation import build_covisitation
from src.popularity import build_popularity
from src.recommend import covisitation_recommend, popular_recommend
from src.split import assign_sessions


def test_popularity_uses_event_weights():
    events = pd.DataFrame(
        {
            "itemid": [1, 1, 2, 2, 3],
            "event": ["view", "addtocart", "transaction", "view", "view"],
        }
    )
    model = build_popularity(events)
    scores = dict(zip(model.item_ids, model.scores))
    assert scores[1] == 4.0   # 1 + 3
    assert scores[2] == 6.0   # 5 + 1
    assert scores[3] == 1.0
    assert model.item_ids[0] == 2


def test_popularity_filters_history():
    events = pd.DataFrame(
        {
            "itemid": [1, 2, 3],
            "event": ["transaction", "view", "view"],
        }
    )
    model = build_popularity(events)
    recs = popular_recommend(model, history={1, 2}, n=5)
    assert recs == [(3, 1.0)]


def _covis_frame():
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2015-01-01 10:00",
                    "2015-01-01 10:01",
                    "2015-01-01 10:02",
                    "2015-01-01 11:00",
                    "2015-01-01 11:01",
                ],
                utc=True,
            ),
            "visitorid": [1, 1, 1, 2, 2],
            "event": ["view", "view", "view", "view", "view"],
            "itemid": [1, 2, 3, 1, 2],
        }
    )
    return assign_sessions(events, gap_minutes=30)


def test_covisitation_counts_symmetric_pairs():
    model = build_covisitation(_covis_frame(), max_session_items=50)
    assert model.stats["pair_occurrences"] == 4  # AB дважды + AC + BC
    idx = {int(v): i for i, v in enumerate(model.item_ids)}

    def score(a, b):
        neighbors, scores = model.neighbor_slice(a)
        lookup = dict(zip(neighbors, scores))
        return float(lookup[idx[b]])

    assert score(1, 2) == 2.0
    assert score(2, 1) == 2.0
    assert score(1, 3) == 1.0


def test_covisitation_recommends_and_excludes_history():
    model = build_covisitation(_covis_frame(), max_session_items=50)
    recs = covisitation_recommend(
        model, context_items=[3], context_weights=[1.0], history={1}, n=5
    )
    # У товара 3 соседи 1 и 2; 1 исключён историей.
    assert [itemid for itemid, _ in recs] == [2]
