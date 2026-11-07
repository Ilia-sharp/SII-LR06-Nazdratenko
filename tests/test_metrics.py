from __future__ import annotations

import numpy as np

from src.metrics import metrics_at_k, ndcg_at_k, precision_at_k, recall_at_k
from src.profiles import build_profiles, context_from_items


def test_precision_recall_known_values():
    recs = [1, 2, 3, 4]
    targets = {1, 4, 9}
    assert precision_at_k(recs, targets, k=10) == 0.2
    assert recall_at_k(recs, targets, k=10) == 2 / 3
    assert ndcg_at_k(recs, targets, k=10) == (
        1.0 + 1.0 / np.log2(5)
    ) / (1.0 + 1.0 / np.log2(3) + 1.0 / np.log2(4))


def test_ndcg_empty_targets_is_zero():
    assert ndcg_at_k([1, 2], set(), k=10) == 0.0


def test_metrics_at_k_keys():
    result = metrics_at_k([1], {1}, k=10)
    assert set(result) == {"precision@10", "recall@10", "ndcg@10"}


def test_profiles_last_events_only():
    import pandas as pd

    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2015-01-01 10:00",
                    "2015-01-01 10:01",
                    "2015-01-01 10:02",
                    "2015-01-01 10:03",
                    "2015-01-01 10:04",
                ],
                utc=True,
            ),
            "visitorid": [7] * 5,
            "event": ["view", "view", "addtocart", "view", "transaction"],
            "itemid": [10, 20, 10, 30, 40],
        }
    )
    profiles = build_profiles(events, context_size=3)
    context, weights = profiles.context(7)
    # Последние 3 события: 10 (addtocart), 30 (view), 40 (transaction).
    # Дубликат 10 схлопывается, порядок — от свежего к старому.
    assert context.tolist() == [40, 30, 10]
    assert weights.tolist() == [5.0, 1.0, 3.0]
    # История для фильтрации — все уникальные товары.
    assert profiles.history(7).tolist() == [10, 20, 30, 40]


def test_context_from_items_dedupes_last():
    context, weights = context_from_items([1, 2, 1, 3])
    assert context == [2, 1, 3]
    assert weights == [1.0, 1.0, 1.0]


def test_unknown_visitor_has_empty_profile():
    import pandas as pd

    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2015-01-01"], utc=True),
            "visitorid": [1],
            "event": ["view"],
            "itemid": [10],
        }
    )
    profiles = build_profiles(events, context_size=5)
    assert profiles.index(999) is None
    assert profiles.history(999).size == 0
    assert profiles.context(999)[0].size == 0
