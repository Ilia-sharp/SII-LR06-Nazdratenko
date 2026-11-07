from __future__ import annotations

import pandas as pd
import pytest


def make_events(rows):
    return pd.DataFrame(
        rows, columns=["timestamp", "visitorid", "event", "itemid"]
    )


@pytest.fixture
def small_events() -> pd.DataFrame:
    base = pd.Timestamp("2015-05-01 00:00:00", tz="UTC")
    rows = [
        (base + pd.Timedelta(hours=0), 100, "view", 1),
        (base + pd.Timedelta(hours=0, minutes=10), 100, "view", 3),
        (base + pd.Timedelta(hours=1), 101, "view", 2),
        (base + pd.Timedelta(hours=1, minutes=10), 101, "addtocart", 3),
        (base + pd.Timedelta(hours=5), 15, "view", 1),
        (base + pd.Timedelta(hours=5, minutes=5), 15, "view", 2),
        (base + pd.Timedelta(hours=5, minutes=10), 15, "view", 4),
        (base + pd.Timedelta(hours=5, minutes=15), 15, "addtocart", 1),
        (base + pd.Timedelta(hours=9), 15, "transaction", 3),
        (base + pd.Timedelta(hours=9, minutes=1), 16, "view", 2),
    ]
    frame = make_events(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame
