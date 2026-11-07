"""Временное разделение и определение сессий."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class TimeSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp
    report: Dict[str, Any]


def time_split(
    events: pd.DataFrame,
    test_fraction: float = 0.20,
) -> TimeSplit:
    """Делит события по времени: последние ``test_fraction`` диапазона — test.

    Перемешивания нет. Граница вычисляется от минимальной и максимальной
    метки времени, поэтому test содержит ровно последние 20% времени.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction должен быть в интервале (0, 1)")

    t_min = events["timestamp"].min()
    t_max = events["timestamp"].max()
    cutoff = t_max - (t_max - t_min) * test_fraction

    train = events[events["timestamp"] < cutoff].copy()
    test = events[events["timestamp"] >= cutoff].copy()

    if train.empty or test.empty:
        raise ValueError("После временного разделения одна из частей пуста.")

    overlap = set(train["timestamp"]).intersection(set(test["timestamp"]))
    if overlap:
        raise AssertionError("Найдены общие метки времени в train и test")

    if train["timestamp"].max() >= test["timestamp"].min():
        raise AssertionError("Граница train/test нарушена")

    report = {
        "cutoff": cutoff.isoformat(),
        "test_fraction": test_fraction,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_share": round(float(len(train) / len(events)), 4),
        "test_share": round(float(len(test) / len(events)), 4),
        "train_timestamp_min": train["timestamp"].min().isoformat(),
        "train_timestamp_max": train["timestamp"].max().isoformat(),
        "test_timestamp_min": test["timestamp"].min().isoformat(),
        "test_timestamp_max": test["timestamp"].max().isoformat(),
    }
    return TimeSplit(train=train, test=test, cutoff=cutoff, report=report)


def assign_sessions(
    events: pd.DataFrame,
    gap_minutes: int = 30,
) -> pd.DataFrame:
    """Назначает session_id: новый посетитель или разрыв > ``gap_minutes``.

    Сессия — последовательность событий одного visitorid, внутри которой
    соседние события отстоят не более чем на ``gap_minutes`` минут.
    """
    if gap_minutes <= 0:
        raise ValueError("gap_minutes должен быть положительным")

    df = events.sort_values(
        ["visitorid", "timestamp"], kind="mergesort"
    ).copy()
    previous_ts = df.groupby("visitorid", sort=False)["timestamp"].shift()
    gap = df["timestamp"] - previous_ts
    is_new = previous_ts.isna() | (gap > pd.Timedelta(minutes=gap_minutes))
    df["session_id"] = is_new.cumsum().astype("int64")

    return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def session_report(events_with_sessions: pd.DataFrame) -> Dict[str, Any]:
    sizes = events_with_sessions.groupby("session_id").size()
    return {
        "sessions": int(sizes.size),
        "mean_events_per_session": round(float(sizes.mean()), 3),
        "median_events_per_session": float(sizes.median()),
        "max_events_per_session": int(sizes.max()),
        "single_event_sessions": int((sizes == 1).sum()),
    }
