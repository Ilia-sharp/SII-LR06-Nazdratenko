"""Профили посетителей: история для фильтрации и контекст последних событий.

Вариант 16 («Один источник истории»): контекст рекомендаций — только
последние события самого visitorid. Сессии других пользователей в контекст
не подмешиваются; они используются лишь при обучении co-visitation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import EVENT_WEIGHTS


@dataclass
class VisitorProfiles:
    visitors: np.ndarray        # отсортированные visitorid train
    history_indptr: np.ndarray
    history_items: np.ndarray   # уникальные товары всей истории (по возрастанию)
    context_indptr: np.ndarray
    context_items: np.ndarray   # последние ContextSize уникальных товаров
    context_weights: np.ndarray # вес самого свежего события этого товара

    def index(self, visitor_id: int) -> int | None:
        pos = int(np.searchsorted(self.visitors, int(visitor_id)))
        if pos < len(self.visitors) and int(self.visitors[pos]) == int(visitor_id):
            return pos
        return None

    def history(self, visitor_id: int) -> np.ndarray:
        pos = self.index(visitor_id)
        if pos is None:
            return np.empty(0, dtype=np.int64)
        return self.history_items[
            self.history_indptr[pos] : self.history_indptr[pos + 1]
        ]

    def context(self, visitor_id: int) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.index(visitor_id)
        if pos is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        return (
            self.context_items[
                self.context_indptr[pos] : self.context_indptr[pos + 1]
            ],
            self.context_weights[
                self.context_indptr[pos] : self.context_indptr[pos + 1]
            ],
        )


def build_profiles(
    train: pd.DataFrame,
    context_size: int = 5,
) -> VisitorProfiles:
    """Строит компактные CSR-профили для всех посетителей train."""
    if train.empty:
        raise ValueError("Нет train-событий для построения профилей")

    visitors = np.sort(train["visitorid"].unique().astype(np.int64))
    n_visitors = len(visitors)

    # --- Полная история (уникальные товары) для фильтрации выдачи -----------
    history = (
        train[["visitorid", "itemid"]]
        .drop_duplicates()
        .sort_values(["visitorid", "itemid"], kind="mergesort")
    )
    hist_pos = np.searchsorted(
        visitors, history["visitorid"].to_numpy(dtype=np.int64)
    )
    hist_counts = np.bincount(hist_pos, minlength=n_visitors)
    history_indptr = np.concatenate(
        [[0], np.cumsum(hist_counts, dtype=np.int64)]
    )

    # --- Контекст: последние context_size событий пользователя --------------
    tail = (
        train.sort_values(
            ["visitorid", "timestamp"], kind="mergesort"
        )
        .groupby("visitorid", sort=False)
        .tail(context_size)
        .copy()
    )
    tail["weight"] = tail["event"].map(EVENT_WEIGHTS).astype("float64")
    tail = tail.sort_values(
        ["visitorid", "timestamp", "itemid"], kind="mergesort"
    )
    # Если товар повторяется в последних событиях, оставляем самое свежее.
    tail = tail.drop_duplicates(["visitorid", "itemid"], keep="last")
    # Внутри пользователя — от свежего к старому.
    tail = tail.sort_values(
        ["visitorid", "timestamp"], ascending=[True, False], kind="mergesort"
    )

    ctx_pos = np.searchsorted(
        visitors, tail["visitorid"].to_numpy(dtype=np.int64)
    )
    ctx_counts = np.bincount(ctx_pos, minlength=n_visitors)
    context_indptr = np.concatenate(
        [[0], np.cumsum(ctx_counts, dtype=np.int64)]
    )

    return VisitorProfiles(
        visitors=visitors,
        history_indptr=history_indptr,
        history_items=history["itemid"].to_numpy(dtype=np.int64),
        context_indptr=context_indptr,
        context_items=tail["itemid"].to_numpy(dtype=np.int64),
        context_weights=tail["weight"].to_numpy(dtype=np.float64),
    )


def context_from_items(item_ids: list[int]) -> Tuple[list[int], list[float]]:
    """Контекст из текущей сессии: последние уникальные товары, вес 1.0."""
    seen: set[int] = set()
    unique_reversed: list[int] = []
    for itemid in reversed(item_ids):
        itemid = int(itemid)
        if itemid not in seen:
            seen.add(itemid)
            unique_reversed.append(itemid)
    context = list(reversed(unique_reversed))
    return context, [1.0] * len(context)


@dataclass
class EvaluationProfiles:
    """Профили для офлайн-оценки с контекстом строго до целевого события.

    Модель по-прежнему обучена только на train. Для каждого целевого
    пользователя берутся его события (train + более ранние события test)
    до первого целевого события test. Такой контекст доступен в момент
    выдачи и не является утечкой: сами целевые события в него не входят.
    """

    history_map: Dict[int, np.ndarray]
    context_items_map: Dict[int, np.ndarray]
    context_weights_map: Dict[int, np.ndarray]

    def index(self, visitor_id: int) -> int | None:
        return 0 if int(visitor_id) in self.history_map else None

    def history(self, visitor_id: int) -> np.ndarray:
        return self.history_map.get(
            int(visitor_id), np.empty(0, dtype=np.int64)
        )

    def context(self, visitor_id: int) -> Tuple[np.ndarray, np.ndarray]:
        visitor_id = int(visitor_id)
        return (
            self.context_items_map.get(
                visitor_id, np.empty(0, dtype=np.int64)
            ),
            self.context_weights_map.get(
                visitor_id, np.empty(0, dtype=np.float64)
            ),
        )


def build_evaluation_profiles(
    events: pd.DataFrame,
    targets: Dict[int, np.ndarray],
    target_times: Dict[int, pd.Timestamp],
    context_size: int = 5,
) -> EvaluationProfiles:
    """Контекст и история на момент первого целевого события test."""
    if not targets:
        return EvaluationProfiles({}, {}, {})

    wanted = np.fromiter(targets.keys(), dtype=np.int64)
    subset = events[events["visitorid"].isin(wanted)].sort_values(
        ["visitorid", "timestamp"], kind="mergesort"
    )

    history_map: Dict[int, np.ndarray] = {}
    context_items_map: Dict[int, np.ndarray] = {}
    context_weights_map: Dict[int, np.ndarray] = {}

    for visitor_id, part in subset.groupby("visitorid", sort=False):
        visitor_id = int(visitor_id)
        cutoff = target_times.get(visitor_id)
        if cutoff is not None:
            part = part[part["timestamp"] < cutoff]
        history_map[visitor_id] = np.sort(
            part["itemid"].unique().astype(np.int64)
        )

        tail = part.tail(context_size).copy()
        if tail.empty:
            context_items_map[visitor_id] = np.empty(0, dtype=np.int64)
            context_weights_map[visitor_id] = np.empty(0, dtype=np.float64)
            continue
        tail["weight"] = tail["event"].map(EVENT_WEIGHTS).astype("float64")
        tail = tail.sort_values(
            ["timestamp", "itemid"], kind="mergesort"
        ).drop_duplicates("itemid", keep="last")
        tail = tail.sort_values("timestamp", ascending=False, kind="mergesort")
        context_items_map[visitor_id] = tail["itemid"].to_numpy(dtype=np.int64)
        context_weights_map[visitor_id] = tail["weight"].to_numpy(
            dtype=np.float64
        )

    return EvaluationProfiles(
        history_map=history_map,
        context_items_map=context_items_map,
        context_weights_map=context_weights_map,
    )
