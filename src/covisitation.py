"""Item-item / co-visitation модель.

На train для каждой сессии берутся уникальные товары в хронологическом
порядке; все пары внутри сессии увеличивают совместную частоту на 1.
Пара кодируется в неупорядоченном виде ``(min, max)``, поэтому одна и та же
связь, встретившаяся в сессиях с разным порядком товаров, учитывается ровно
один раз. Матрица хранится в симметричном CSR-виде: для каждого товара —
соседи и их совместные частоты; дублей «сосед → сосед» в строке нет.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class CovisitationModel:
    item_ids: np.ndarray          # vocabulary: отсортированные itemid train
    indptr: np.ndarray            # CSR-указатели по item_ids
    neighbors: np.ndarray         # индексы соседей в item_ids
    scores: np.ndarray            # совместные частоты
    stats: Dict[str, Any]

    def _index(self, itemid: int) -> int | None:
        pos = int(np.searchsorted(self.item_ids, itemid))
        if pos < len(self.item_ids) and int(self.item_ids[pos]) == int(itemid):
            return pos
        return None

    def neighbor_slice(self, itemid: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = self._index(itemid)
        if idx is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        start, end = int(self.indptr[idx]), int(self.indptr[idx + 1])
        return self.neighbors[start:end], self.scores[start:end]

    def recommend(
        self,
        context_items: list[int],
        context_weights: list[float] | np.ndarray | None = None,
        history: set[int] | None = None,
        n: int = 10,
    ) -> list[tuple[int, float]]:
        """Агрегирует соседей по последним товарам пользователя.

        score(candidate) = sum(weight(context_item) * co_visit(context, candidate)).
        Товары из истории пользователя исключаются.
        """
        if n <= 0:
            return []
        if context_weights is None:
            context_weights = [1.0] * len(context_items)
        if len(context_weights) != len(context_items):
            raise ValueError("context_items и context_weights разной длины")
        history = history or set()

        aggregated: Dict[int, float] = {}
        for itemid, weight in zip(context_items, context_weights):
            neighbor_idx, neighbor_scores = self.neighbor_slice(int(itemid))
            if len(neighbor_idx) == 0:
                continue
            for idx, score in zip(neighbor_idx, neighbor_scores):
                candidate = int(self.item_ids[int(idx)])
                if candidate in history:
                    continue
                aggregated[candidate] = aggregated.get(candidate, 0.0) + float(
                    weight
                ) * float(score)

        ranked = sorted(
            aggregated.items(), key=lambda kv: (-kv[1], kv[0])
        )
        return [(itemid, float(score)) for itemid, score in ranked[:n]]


def _pair_positions(
    starts: np.ndarray,
    ends: np.ndarray,
    max_session_items: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Все пары позиций ``(i < j)`` внутри сессий — без Python-циклов.

    Сессии длиннее ``max_session_items`` обрезаются до последних товаров
    (защита от квадратичного роста числа пар).

    Возвращает позиции левого и правого элемента пары в плоском массиве
    уникальных товаров сессий и общее количество пар.
    """
    sizes = ends - starts
    limit = max(1, int(max_session_items))
    # Обрезанная сессия начинается с конца: остаются последние `limit` товаров.
    eff_size = np.minimum(sizes, limit)
    eff_start = np.where(sizes > limit, ends - limit, starts)

    counts = eff_size * (eff_size - 1) // 2
    total = int(counts.sum())
    if total == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, 0

    session_of_pair = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    pair_offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)[:-1]]
    )
    local = np.arange(total, dtype=np.int64) - pair_offsets[session_of_pair]

    # Таблица соответствий «номер пары внутри сессии → (i, j)» для каждого
    # размера сессии. Для размера s пар C(s, 2) — всего до 20 тысяч записей.
    max_size = int(eff_size.max())
    table_i: list[np.ndarray] = []
    table_j: list[np.ndarray] = []
    table_offset = np.zeros(max_size + 1, dtype=np.int64)
    position = 0
    for size in range(2, max_size + 1):
        rows, cols = np.triu_indices(size, k=1)
        table_i.append(rows.astype(np.int64))
        table_j.append(cols.astype(np.int64))
        table_offset[size] = position
        position += len(rows)

    lookup = table_offset[eff_size[session_of_pair]] + local
    all_i = np.concatenate(table_i)
    all_j = np.concatenate(table_j)
    left = eff_start[session_of_pair] + all_i[lookup]
    right = eff_start[session_of_pair] + all_j[lookup]
    return left, right, total


def build_covisitation(
    train_sessions: pd.DataFrame,
    max_session_items: int = 50,
) -> CovisitationModel:
    """Строит симметричную co-visitation матрицу по train-сессиям."""
    if train_sessions.empty:
        raise ValueError("Нет train-событий для построения co-visitation")

    item_ids = np.sort(
        train_sessions["itemid"].unique().astype(np.int64)
    )
    n_items = int(len(item_ids))
    item_index = np.searchsorted(
        item_ids, train_sessions["itemid"].to_numpy(dtype=np.int64)
    )

    frame = pd.DataFrame(
        {
            "session_id": train_sessions["session_id"].to_numpy(dtype=np.int64),
            "item_idx": item_index.astype(np.int64),
            # Метка времени как int64 (наносекунды): сортировка по нему в
            # десятки раз быстрее, чем по объектам Timestamp.
            "ts": train_sessions["timestamp"]
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64),
        }
    )
    # Уникальные товары сессии; keep="last" оставляет самое свежее появление.
    unique = (
        frame.sort_values(
            ["session_id", "ts", "item_idx"], kind="mergesort"
        )
        .drop_duplicates(["session_id", "item_idx"], keep="last")
        .reset_index(drop=True)
    )

    session_ids = unique["session_id"].to_numpy()
    item_idx = unique["item_idx"].to_numpy(dtype=np.int64)
    starts = np.flatnonzero(
        np.r_[True, session_ids[1:] != session_ids[:-1]]
    )
    ends = np.r_[starts[1:], len(session_ids)]
    sizes = ends - starts

    left_pos, right_pos, pair_occurrences = _pair_positions(
        starts, ends, max_session_items
    )

    if pair_occurrences:
        left = item_idx[left_pos]
        right = item_idx[right_pos]
        # Неупорядоченная пара: (min, max). Так связь, увиденная в сессиях с
        # разным порядком товаров, попадает в счётчик ровно один раз.
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        # Ключ в int64: n_items^2 должно помещаться (n_items < 3 млрд).
        keys = low * np.int64(n_items) + high
        unique_keys, counts = np.unique(keys, return_counts=True)
        pair_low = (unique_keys // np.int64(n_items)).astype(np.int64)
        pair_high = (unique_keys % np.int64(n_items)).astype(np.int64)
    else:
        pair_low = np.empty(0, dtype=np.int64)
        pair_high = np.empty(0, dtype=np.int64)
        counts = np.empty(0, dtype=np.int64)

    # Симметричная матрица: каждая неупорядоченная пара даёт две дуги.
    src = np.concatenate([pair_low, pair_high])
    dst = np.concatenate([pair_high, pair_low])
    weight = np.concatenate([counts, counts]).astype(np.float32)

    order = np.lexsort((dst, -weight, src))
    src = src[order]
    dst = dst[order]
    weight = weight[order]

    indptr = np.zeros(n_items + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    indptr = np.cumsum(indptr)

    stats = {
        "vocabulary_items": n_items,
        "pair_occurrences": int(pair_occurrences),
        "unique_pairs": int(len(unique_keys)) if pair_occurrences else 0,
        "mean_neighbors_per_item": round(
            float(len(dst) / n_items) if n_items else 0.0, 2
        ),
        "max_session_items": int(max_session_items),
        "sessions_with_pairs": int((sizes >= 2).sum()),
    }
    return CovisitationModel(
        item_ids=item_ids,
        indptr=indptr,
        neighbors=dst.astype(np.int32),
        scores=weight.astype(np.float32),
        stats=stats,
    )
