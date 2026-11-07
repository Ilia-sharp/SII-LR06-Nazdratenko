"""Item-item / co-visitation модель.

На train для каждой сессии берутся уникальные товары в хронологическом
порядке; все пары внутри сессии увеличивают совместную частоту на 1.
Матрица хранится в симметричном CSR-виде: для каждого товара — соседи и
их совместные частоты.
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
            "timestamp": train_sessions["timestamp"].to_numpy(),
        }
    )
    # Уникальные товары сессии; keep="last" оставляет самое свежее появление.
    unique = (
        frame.sort_values(
            ["session_id", "timestamp", "item_idx"], kind="mergesort"
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

    keys_parts: list[np.ndarray] = []

    # Большинство сессий содержат ровно два уникальных товара — это быстрый путь.
    pair_starts = starts[sizes == 2]
    if len(pair_starts):
        left = item_idx[pair_starts]
        right = item_idx[pair_starts + 1]
        keys_parts.append(left * n_items + right)

    # Длинные сессии: все сочетания, но не более max_session_items последних.
    long_mask = sizes > 2
    for start, end in zip(starts[long_mask], ends[long_mask]):
        part = item_idx[start:end]
        if len(part) > max_session_items:
            part = part[-max_session_items:]
        for i in range(len(part) - 1):
            left = int(part[i])
            rest = part[i + 1 :]
            keys_parts.append(left * n_items + rest)

    if keys_parts:
        all_keys = np.concatenate(keys_parts).astype(np.int64, copy=False)
    else:
        all_keys = np.empty(0, dtype=np.int64)

    unique_keys, counts = np.unique(all_keys, return_counts=True)
    left = (unique_keys // n_items).astype(np.int64)
    right = (unique_keys % n_items).astype(np.int64)

    src = np.concatenate([left, right])
    dst = np.concatenate([right, left])
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
        "pair_occurrences": int(len(all_keys)),
        "unique_pairs": int(len(unique_keys)),
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
