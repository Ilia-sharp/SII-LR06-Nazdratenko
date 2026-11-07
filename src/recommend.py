"""Формирование Top-N выдачи и фильтрация истории."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .covisitation import CovisitationModel
from .popularity import PopularityModel


def popular_recommend(
    model: PopularityModel,
    history: Iterable[int],
    n: int = 10,
) -> list[tuple[int, float]]:
    """Top-N популярностного baseline с исключением истории пользователя."""
    if n <= 0:
        return []
    hidden = set(int(x) for x in history)
    result: list[tuple[int, float]] = []
    for itemid, score in zip(model.item_ids, model.scores):
        itemid = int(itemid)
        if itemid in hidden:
            continue
        result.append((itemid, float(score)))
        if len(result) == n:
            break
    return result


def covisitation_recommend(
    model: CovisitationModel,
    context_items: Sequence[int],
    context_weights: Sequence[float] | None,
    history: Iterable[int],
    n: int = 10,
) -> list[tuple[int, float]]:
    """Top-N co-visitation с исключением всей истории пользователя."""
    return model.recommend(
        context_items=list(int(x) for x in context_items),
        context_weights=(
            None if context_weights is None else list(context_weights)
        ),
        history=set(int(x) for x in history),
        n=n,
    )


def dedupe_keep_recent(item_ids: Sequence[int], limit: int) -> list[int]:
    """Оставляет последние ``limit`` уникальных товаров, порядок — от старых."""
    seen: set[int] = set()
    picked: list[int] = []
    for itemid in reversed(item_ids):
        itemid = int(itemid)
        if itemid in seen:
            continue
        seen.add(itemid)
        picked.append(itemid)
        if len(picked) == limit:
            break
    return list(reversed(picked))


def merge_with_popularity(
    primary: list[tuple[int, float]],
    popularity: PopularityModel,
    history: Iterable[int],
    n: int,
) -> list[tuple[int, float]]:
    """Добивает выдачу популярностью, сохраняя исходные score первых мест.

    Используется только для API/демо при cold start или нехватке кандидатов,
    в офлайн-метриках модели сравниваются без такого смешивания.
    """
    if len(primary) >= n:
        return primary[:n]
    hidden = set(int(x) for x in history)
    hidden.update(int(itemid) for itemid, _ in primary)
    result = list(primary)
    for itemid, score in zip(popularity.item_ids, popularity.scores):
        itemid = int(itemid)
        if itemid in hidden:
            continue
        result.append((itemid, float(score)))
        hidden.add(itemid)
        if len(result) == n:
            break
    return result


def format_recommendations(
    items: Sequence[tuple[int, float]],
) -> list[dict[str, float]]:
    return [
        {"itemid": int(itemid), "score": round(float(score), 6)}
        for itemid, score in items
    ]
