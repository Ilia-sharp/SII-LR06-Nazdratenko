"""Сохранение и загрузка обученной модели, выдача рекомендаций."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from .config import CONTEXT_SIZE
from .covisitation import CovisitationModel
from .popularity import PopularityModel
from .profiles import VisitorProfiles
from .recommend import (
    covisitation_recommend,
    dedupe_keep_recent,
    merge_with_popularity,
    popular_recommend,
)

MODEL_FILE = "model.npz"
METADATA_FILE = "metadata.json"


@dataclass
class ModelBundle:
    popularity: PopularityModel
    covisitation: CovisitationModel
    profiles: VisitorProfiles
    test_visitors: np.ndarray
    test_indptr: np.ndarray
    test_items: np.ndarray
    metadata: Dict[str, Any]


def _test_target_arrays(
    targets: Dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visitors = np.array(sorted(targets), dtype=np.int64)
    counts = np.array([len(targets[int(v)]) for v in visitors], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
    items = (
        np.concatenate([targets[int(v)] for v in visitors]).astype(np.int64)
        if len(visitors)
        else np.empty(0, dtype=np.int64)
    )
    return visitors, indptr, items


def test_targets_dict(bundle: ModelBundle) -> Dict[int, np.ndarray]:
    result: Dict[int, np.ndarray] = {}
    for i, visitor in enumerate(bundle.test_visitors):
        result[int(visitor)] = bundle.test_items[
            bundle.test_indptr[i] : bundle.test_indptr[i + 1]
        ]
    return result


def save_bundle(
    directory: str | Path,
    *,
    popularity: PopularityModel,
    covisitation: CovisitationModel,
    profiles: VisitorProfiles,
    targets: Dict[int, np.ndarray],
    metadata: Dict[str, Any],
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    test_visitors, test_indptr, test_items = _test_target_arrays(targets)
    np.savez_compressed(
        directory / MODEL_FILE,
        popularity_item_ids=popularity.item_ids,
        popularity_scores=popularity.scores,
        covis_item_ids=covisitation.item_ids,
        covis_indptr=covisitation.indptr,
        covis_neighbors=covisitation.neighbors,
        covis_scores=covisitation.scores,
        visitors=profiles.visitors,
        history_indptr=profiles.history_indptr,
        history_items=profiles.history_items,
        context_indptr=profiles.context_indptr,
        context_items=profiles.context_items,
        context_weights=profiles.context_weights,
        test_visitors=test_visitors,
        test_indptr=test_indptr,
        test_items=test_items,
    )
    with (directory / METADATA_FILE).open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    return directory


def load_bundle(directory: str | Path) -> ModelBundle:
    directory = Path(directory)
    model_path = directory / MODEL_FILE
    metadata_path = directory / METADATA_FILE
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"В {directory} нет обученной модели. Сначала выполните "
            "`python -m src.cli train --events data/events.csv`."
        )

    data = np.load(model_path, allow_pickle=False)
    popularity = PopularityModel(
        item_ids=data["popularity_item_ids"],
        scores=data["popularity_scores"],
    )
    covisitation = CovisitationModel(
        item_ids=data["covis_item_ids"],
        indptr=data["covis_indptr"],
        neighbors=data["covis_neighbors"],
        scores=data["covis_scores"],
        stats={},
    )
    profiles = VisitorProfiles(
        visitors=data["visitors"],
        history_indptr=data["history_indptr"],
        history_items=data["history_items"],
        context_indptr=data["context_indptr"],
        context_items=data["context_items"],
        context_weights=data["context_weights"],
    )
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    return ModelBundle(
        popularity=popularity,
        covisitation=covisitation,
        profiles=profiles,
        test_visitors=data["test_visitors"],
        test_indptr=data["test_indptr"],
        test_items=data["test_items"],
        metadata=metadata,
    )


def recommend_for_visitor(
    bundle: ModelBundle,
    visitor_id: int,
    n: int = 10,
    fallback: bool = True,
    model: str = "auto",
) -> Dict[str, Any]:
    """Выдача по visitorid: контекст — только последние события пользователя."""
    history = bundle.profiles.history(visitor_id)
    context, context_weights = bundle.profiles.context(visitor_id)

    if model == "popularity":
        recommendations = popular_recommend(bundle.popularity, history, n)
        model_name = "popularity"
    elif len(context) == 0:
        recommendations = popular_recommend(bundle.popularity, history, n)
        model_name = "popularity_cold_start"
    else:
        recommendations = covisitation_recommend(
            bundle.covisitation,
            context_items=context,
            context_weights=context_weights,
            history=history,
            n=n,
        )
        model_name = "covisitation"
        if fallback and model == "auto" and len(recommendations) < n:
            recommendations = merge_with_popularity(
                recommendations, bundle.popularity, history, n
            )
            if recommendations:
                model_name = "covisitation+popularity_fill"

    return {
        "visitor_id": int(visitor_id),
        "context_items": [int(x) for x in context],
        "context_weights": [round(float(x), 4) for x in context_weights],
        "history_size": int(len(history)),
        "model": model_name,
        "recommendations": [
            {"itemid": int(itemid), "score": round(float(score), 6)}
            for itemid, score in recommendations
        ],
    }


def recommend_for_items(
    bundle: ModelBundle,
    item_ids: Sequence[int],
    n: int = 10,
    fallback: bool = False,
) -> Dict[str, Any]:
    """Выдача по товарам текущей сессии (без visitorid)."""
    items = [int(x) for x in item_ids]
    context = dedupe_keep_recent(items, limit=CONTEXT_SIZE)
    history = set(items)

    recommendations = covisitation_recommend(
        bundle.covisitation,
        context_items=context,
        context_weights=[1.0] * len(context),
        history=history,
        n=n,
    )
    model_name = "covisitation"
    if fallback and len(recommendations) < n:
        recommendations = merge_with_popularity(
            recommendations, bundle.popularity, history, n
        )
        model_name = "covisitation+popularity_fill"

    return {
        "input_items": items,
        "context_items": context,
        "context_weights": [1.0] * len(context),
        "history_size": len(history),
        "model": model_name,
        "recommendations": [
            {"itemid": int(itemid), "score": round(float(score), 6)}
            for itemid, score in recommendations
        ],
    }


def iter_test_targets(
    bundle: ModelBundle,
) -> Iterable[tuple[int, np.ndarray]]:
    for i, visitor in enumerate(bundle.test_visitors):
        yield int(visitor), bundle.test_items[
            bundle.test_indptr[i] : bundle.test_indptr[i + 1]
        ]
