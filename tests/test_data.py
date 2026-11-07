from __future__ import annotations

import pandas as pd
import pytest

from src.data import build_target_sets, load_events


def test_load_events_validates_and_cleans(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame(
        [
            (1000, 1, "view", 10, None),
            (1000, 1, "view", 10, None),          # полный дубликат
            (2000, 2, "addtocart", 20, None),
            (3000, 3, "unknown", 30, None),       # неизвестное событие
        ],
        columns=["timestamp", "visitorid", "event", "itemid", "transactionid"],
    ).to_csv(path, index=False)

    loaded = load_events(path)
    df = loaded.frame
    assert len(df) == 2
    assert loaded.report["duplicates_removed"] == 1
    assert loaded.report["unknown_events_removed"] == 1
    assert str(df["timestamp"].dtype).startswith("datetime64")
    assert df["timestamp"].is_monotonic_increasing


def test_load_events_requires_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": [1], "visitorid": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="обязательных колонок"):
        load_events(path)


def test_target_sets_only_variant_group():
    test = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2015-01-01"] * 5, utc=True),
            "visitorid": [15, 15, 16, 45, 45],
            "event": ["view", "addtocart", "transaction", "transaction", "view"],
            "itemid": [1, 2, 3, 4, 5],
        }
    )
    targets, report = build_target_sets(
        test, ("addtocart", "transaction"), visitor_mod=30, visitor_remainder=15
    )
    assert set(targets) == {15, 45}
    assert targets[15].tolist() == [2]
    assert targets[45].tolist() == [4]
    assert report["target_visitors"] == 2
