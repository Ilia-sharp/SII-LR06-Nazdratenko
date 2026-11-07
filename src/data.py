"""Загрузка, проверка и очистка events.csv Retailrocket."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .config import EVENT_WEIGHTS

REQUIRED_COLUMNS = ("timestamp", "visitorid", "event", "itemid")


@dataclass
class EventsData:
    frame: pd.DataFrame
    report: Dict[str, Any]


def _jsonable(value: Any) -> Any:
    """Приводит numpy/pandas-скаляры к типам, понятным json."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def load_events(path: str | Path) -> EventsData:
    """Читает events.csv, проверяет схему и возвращает очищенный DataFrame.

    Проверки:
      * обязательные колонки;
      * приведение visitorid/itemid к int64, timestamp к datetime UTC;
      * пропуски (transactionid пуст у view/addtocart — это нормально);
      * полные дубликаты строк;
      * неизвестные типы событий;
      * диапазон временных меток и порядок сортировки.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл событий: {path}. "
            "Скачайте Retailrocket events.csv с Kaggle и положите его в data/."
        )

    raw = pd.read_csv(path)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_columns:
        raise ValueError(
            "В events.csv нет обязательных колонок: " + ", ".join(missing_columns)
        )

    n_raw = len(raw)
    missing_before = {c: int(raw[c].isna().sum()) for c in raw.columns}

    df = raw.copy()
    df["visitorid"] = pd.to_numeric(df["visitorid"], errors="coerce")
    df["itemid"] = pd.to_numeric(df["itemid"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["event"] = df["event"].astype(str).str.strip().str.lower()

    invalid_rows = int(
        df[["visitorid", "itemid", "timestamp"]].isna().any(axis=1).sum()
    )
    df = df.dropna(subset=["visitorid", "itemid", "timestamp"]).copy()
    df["visitorid"] = df["visitorid"].astype("int64")
    df["itemid"] = df["itemid"].astype("int64")
    df["timestamp"] = df["timestamp"].astype("int64")

    n_before_dups = len(df)
    df = df.drop_duplicates().copy()
    duplicates_removed = n_before_dups - len(df)

    unknown_events = df.loc[~df["event"].isin(EVENT_WEIGHTS), "event"]
    unknown_count = int(len(unknown_events))
    if unknown_count:
        df = df[df["event"].isin(EVENT_WEIGHTS)].copy()

    if df.empty:
        raise ValueError("После очистки не осталось ни одного события.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values(
        ["timestamp", "visitorid", "itemid"], kind="mergesort"
    ).reset_index(drop=True)

    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    span_days = (ts_max - ts_min).total_seconds() / 86400.0

    report = {
        "source": str(path),
        "rows_raw": n_raw,
        "rows_clean": int(len(df)),
        "missing_values_raw": missing_before,
        "rows_with_broken_ids_or_timestamp": invalid_rows,
        "duplicates_removed": duplicates_removed,
        "unknown_events_removed": unknown_count,
        "event_counts": {
            str(k): int(v) for k, v in df["event"].value_counts().items()
        },
        "visitors": int(df["visitorid"].nunique()),
        "items": int(df["itemid"].nunique()),
        "timestamp_min": ts_min.isoformat(),
        "timestamp_max": ts_max.isoformat(),
        "span_days": round(float(span_days), 3),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    return EventsData(frame=df, report=_jsonable(report))


def event_weight(events: pd.Series) -> pd.Series:
    """Переводит названия событий в учебные веса 1/3/5."""
    return events.map(EVENT_WEIGHTS).astype("float64")


def build_target_sets(
    test: pd.DataFrame,
    target_events: Tuple[str, ...],
    visitor_mod: int,
    visitor_remainder: int,
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """Собирает целевые товары теста для посетителей варианта."""
    group = test[test["visitorid"] % visitor_mod == visitor_remainder]
    target = group[group["event"].isin(target_events)]
    target = target[["visitorid", "itemid"]].drop_duplicates()

    per_visitor: Dict[int, np.ndarray] = {}
    for vid, part in target.groupby("visitorid", sort=True):
        per_visitor[int(vid)] = np.sort(part["itemid"].to_numpy(dtype=np.int64))

    report = {
        "variant_test_events": int(len(group)),
        "variant_test_visitors": int(group["visitorid"].nunique()),
        "target_events_rows": int(len(target)),
        "target_visitors": int(len(per_visitor)),
        "mean_targets_per_visitor": round(
            float(np.mean([len(v) for v in per_visitor.values()]))
            if per_visitor
            else 0.0,
            3,
        ),
        "target_event_counts": {
            str(k): int(v)
            for k, v in group[group["event"].isin(target_events)][
                "event"
            ].value_counts().items()
        },
    }
    return per_visitor, _jsonable(report)


def first_target_times(
    test: pd.DataFrame,
    target_events: Tuple[str, ...],
    visitor_mod: int,
    visitor_remainder: int,
    target_visitors: Dict[int, np.ndarray],
) -> Dict[int, pd.Timestamp]:
    """Время первого целевого события test для каждого целевого посетителя."""
    group = test[
        (test["visitorid"] % visitor_mod == visitor_remainder)
        & test["event"].isin(target_events)
        & test["visitorid"].isin(list(target_visitors))
    ]
    if group.empty:
        return {}
    times = group.groupby("visitorid")["timestamp"].min()
    return {int(vid): ts for vid, ts in times.items()}
