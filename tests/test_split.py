from __future__ import annotations

import pandas as pd

from src.split import assign_sessions, time_split


def test_time_split_is_chronological(small_events):
    split = time_split(small_events, test_fraction=0.2)
    assert split.train["timestamp"].max() < split.test["timestamp"].min()
    assert split.test["timestamp"].min() >= split.cutoff
    assert set(split.train.index).isdisjoint(set(split.test.index))
    assert len(split.train) + len(split.test) == len(small_events)
    assert split.report["test_rows"] == 2


def test_sessions_split_on_gap_and_visitor(small_events):
    with_sessions = assign_sessions(small_events, gap_minutes=30)
    sessions = with_sessions.groupby("visitorid")["session_id"].nunique()
    # Посетитель 15: события 10:00 и 12:00 — разные сессии (разрыв > 30 мин).
    assert sessions[15] == 2
    # Посетитель 100: 10 минут — одна сессия.
    assert sessions[100] == 1
    # Нумерация сессий уникальна между посетителями.
    counts = with_sessions.groupby("session_id")["visitorid"].nunique()
    assert counts.max() == 1
