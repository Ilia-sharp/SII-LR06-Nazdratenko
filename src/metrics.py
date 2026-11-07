"""Офлайн-метрики Top-N: Precision@k, Recall@k, NDCG@k."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def precision_at_k(
    recommendations: Sequence[int],
    targets: Iterable[int],
    k: int = 10,
) -> float:
    if k <= 0:
        raise ValueError("k должен быть положительным")
    target_set = set(int(x) for x in targets)
    hits = sum(1 for item in recommendations[:k] if int(item) in target_set)
    return hits / k


def recall_at_k(
    recommendations: Sequence[int],
    targets: Iterable[int],
    k: int = 10,
) -> float:
    target_set = set(int(x) for x in targets)
    if not target_set:
        return 0.0
    hits = sum(1 for item in recommendations[:k] if int(item) in target_set)
    return hits / len(target_set)


def ndcg_at_k(
    recommendations: Sequence[int],
    targets: Iterable[int],
    k: int = 10,
) -> float:
    """Бинарная NDCG@k: релевантно, если товар есть в целевых событиях."""
    target_set = set(int(x) for x in targets)
    if not target_set:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recommendations[:k], start=1):
        if int(item) in target_set:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(target_set), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def metrics_at_k(
    recommendations: Sequence[int],
    targets: Iterable[int],
    k: int = 10,
) -> dict[str, float]:
    return {
        f"precision@{k}": precision_at_k(recommendations, targets, k),
        f"recall@{k}": recall_at_k(recommendations, targets, k),
        f"ndcg@{k}": ndcg_at_k(recommendations, targets, k),
    }
