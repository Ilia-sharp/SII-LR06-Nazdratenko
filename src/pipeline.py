"""Полный конвейер: загрузка → split → обучение → оценка → артефакты."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from .artifacts import save_bundle
from .config import CONFIG, ExperimentConfig
from .covisitation import build_covisitation
from .data import build_target_sets, first_target_times, load_events
from .metrics import metrics_at_k
from .popularity import build_popularity
from .profiles import build_evaluation_profiles, build_profiles
from .recommend import covisitation_recommend, popular_recommend
from .split import assign_sessions, session_report, time_split

Progress = Callable[[str], None]


def _noop(message: str) -> None:
    pass


def _average(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def evaluate_models(
    popularity,
    covisitation,
    profiles,
    targets: Dict[int, np.ndarray],
    k: int = 10,
) -> Dict[str, Any]:
    """Считает метрики для обеих моделей на одних и тех же тёплых пользователях."""
    pop_rows: list[dict[str, float]] = []
    cov_rows: list[dict[str, float]] = []
    pop_new_rows: list[dict[str, float]] = []
    cov_new_rows: list[dict[str, float]] = []
    pop_new_hits = 0
    cov_new_hits = 0
    cold_start = 0
    no_context = 0
    pop_catalog: set[int] = set()
    cov_catalog: set[int] = set()
    pop_hits = 0
    cov_hits = 0
    context_lengths: list[int] = []
    evaluated_target_counts: list[int] = []
    new_target_pairs = 0
    all_target_pairs = 0

    for visitor_id, target in targets.items():
        history = profiles.history(visitor_id)
        context, weights = profiles.context(visitor_id)
        if len(context) == 0:
            if len(history) == 0:
                cold_start += 1
            else:
                no_context += 1
            continue

        context_lengths.append(int(len(context)))
        target_set = set(int(x) for x in target)
        evaluated_target_counts.append(len(target_set))
        history_set = set(int(x) for x in history)
        all_target_pairs += len(target_set)

        rec_pop = popular_recommend(popularity, history, n=k)
        rec_cov = covisitation_recommend(
            covisitation, context, weights, history, n=k
        )
        pop_ids = [int(itemid) for itemid, _ in rec_pop]
        cov_ids = [int(itemid) for itemid, _ in rec_cov]

        pop_rows.append(metrics_at_k(pop_ids, target_set, k))
        cov_rows.append(metrics_at_k(cov_ids, target_set, k))
        pop_catalog.update(pop_ids)
        cov_catalog.update(cov_ids)
        pop_hits += int(any(i in target_set for i in pop_ids))
        cov_hits += int(any(i in target_set for i in cov_ids))

        # Вспомогательный срез: цели, которых ещё не было в истории.
        # Именно такие товары система в принципе может рекомендовать.
        new_targets = target_set - history_set
        if new_targets:
            pop_new_rows.append(metrics_at_k(pop_ids, new_targets, k))
            cov_new_rows.append(metrics_at_k(cov_ids, new_targets, k))
            pop_new_hits += int(any(i in new_targets for i in pop_ids))
            cov_new_hits += int(any(i in new_targets for i in cov_ids))
            new_target_pairs += len(new_targets)

    evaluated = len(cov_rows)

    def summarize(
        rows: list[dict[str, float]],
        catalog: set[int],
        hits: int,
        count: int,
    ):
        return {
            f"precision@{k}": round(_average([r[f"precision@{k}"] for r in rows]), 6),
            f"recall@{k}": round(_average([r[f"recall@{k}"] for r in rows]), 6),
            f"ndcg@{k}": round(_average([r[f"ndcg@{k}"] for r in rows]), 6),
            f"hit_rate@{k}": round(hits / count, 6) if count else 0.0,
            "catalog_coverage": len(catalog),
        }

    return {
        "k": k,
        "target_visitors": int(len(targets)),
        "evaluated_visitors": evaluated,
        "cold_start_visitors": cold_start,
        "visitors_with_history_but_no_context": no_context,
        "mean_context_length": round(_average(context_lengths), 3),
        "mean_targets_per_evaluated_visitor": round(
            _average(evaluated_target_counts), 3
        ),
        "popularity": summarize(pop_rows, pop_catalog, pop_hits, evaluated),
        "covisitation": summarize(cov_rows, cov_catalog, cov_hits, evaluated),
        "new_targets": {
            "target_pairs": new_target_pairs,
            "all_target_pairs": all_target_pairs,
            "share_already_in_history": round(
                1.0 - new_target_pairs / all_target_pairs, 6
            )
            if all_target_pairs
            else 0.0,
            "evaluated_visitors": len(cov_new_rows),
            "popularity": summarize(
                pop_new_rows, pop_catalog, pop_new_hits, len(pop_new_rows)
            ),
            "covisitation": summarize(
                cov_new_rows, cov_catalog, cov_new_hits, len(cov_new_rows)
            ),
        },
    }


def build_example(
    popularity,
    covisitation,
    profiles,
    targets: Dict[int, np.ndarray],
    k: int = 10,
) -> Dict[str, Any]:
    """Ищет показательного пользователя с попаданием co-visitation в цель."""
    best: tuple[int, int, int] | None = None
    for visitor_id, target in targets.items():
        context, weights = profiles.context(visitor_id)
        if len(context) == 0:
            continue
        history = profiles.history(visitor_id)
        recs = covisitation_recommend(
            covisitation, context, weights, history, n=k
        )
        rec_ids = [itemid for itemid, _ in recs]
        hits = len(set(rec_ids) & set(int(x) for x in target))
        if hits:
            key = (hits, len(target), -visitor_id)
            if best is None or key > best:
                best = key
                best_visitor = int(visitor_id)

    if best is None:
        for visitor_id, target in targets.items():
            context, weights = profiles.context(visitor_id)
            if len(context):
                best_visitor = int(visitor_id)
                break
        else:
            return {}

    context, weights = profiles.context(best_visitor)
    history = profiles.history(best_visitor)
    rec_cov = covisitation_recommend(
        covisitation, context, weights, history, n=k
    )
    rec_pop = popular_recommend(popularity, history, n=k)
    target = targets[best_visitor]
    target_set = set(int(x) for x in target)

    return {
        "visitor_id": best_visitor,
        "context_items": [int(x) for x in context],
        "context_weights": [round(float(x), 4) for x in weights],
        "history_size": int(len(history)),
        "test_target_items": sorted(target_set),
        "covisitation": [
            {
                "itemid": int(itemid),
                "score": round(float(score), 6),
                "hit": int(itemid) in target_set,
            }
            for itemid, score in rec_cov
        ],
        "popularity": [
            {
                "itemid": int(itemid),
                "score": round(float(score), 6),
                "hit": int(itemid) in target_set,
            }
            for itemid, score in rec_pop
        ],
        "explanation": (
            "Кандидаты — соседи последних товаров пользователя по сессиям "
            "train; score агрегирует совместные частоты с весами событий. "
            "Из выдачи удалены все товары, которые уже были в истории."
        ),
    }


def train_and_evaluate(
    events_path: str | Path,
    artifacts_dir: str | Path,
    config: ExperimentConfig = CONFIG,
    progress: Progress = _noop,
) -> Dict[str, Any]:
    progress("1/7 Загрузка и проверка events.csv")
    loaded = load_events(events_path)
    events = loaded.frame

    progress("2/7 Временное разделение train/test")
    split = time_split(events, config.test_fraction)

    progress("3/7 Определение сессий на train")
    train_sessions = assign_sessions(split.train, config.session_gap_minutes)
    sessions = session_report(train_sessions)

    progress("4/7 Популярностный baseline")
    popularity = build_popularity(split.train)

    progress("5/7 Item-item / co-visitation")
    covisitation = build_covisitation(
        train_sessions, config.max_session_items
    )

    progress("6/7 Профили посетителей и целевые события теста")
    profiles = build_profiles(split.train, config.context_size)
    targets, targets_report = build_target_sets(
        split.test,
        config.target_events,
        config.target_mod,
        config.target_remainder,
    )
    target_times = first_target_times(
        split.test,
        config.target_events,
        config.target_mod,
        config.target_remainder,
        targets,
    )
    eval_profiles = build_evaluation_profiles(
        events, targets, target_times, config.context_size
    )

    progress("7/7 Оценка Precision/Recall/NDCG и сохранение артефактов")
    evaluation = evaluate_models(
        popularity, covisitation, eval_profiles, targets, config.top_n
    )
    example = build_example(
        popularity, covisitation, profiles, targets, config.top_n
    )

    top_popularity = [
        {"itemid": int(itemid), "score": round(float(score), 4)}
        for itemid, score in popularity.top(10)
    ]

    metadata: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lab": "Рекомендательные ИИ-системы",
        "variant": {
            "number": config.variant,
            "condition": f"visitorid % {config.target_mod} = {config.target_remainder}",
            "scenario": "Один источник истории",
            "individual_condition": (
                "Использовать только последние события пользователя; "
                "контекст не смешивается с сессиями других посетителей"
            ),
            "top_n": config.top_n,
        },
        "config": config.as_dict(),
        "data": loaded.report,
        "split": split.report,
        "sessions": sessions,
        "popularity": {
            "weights": config.event_weights,
            "top10": top_popularity,
        },
        "covisitation": covisitation.stats,
        "targets": targets_report,
        "evaluation_protocol": {
            "model_training": "только train (временная граница 80/20)",
            "context": (
                "последние события пользователя строго до первого целевого "
                "события test; более ранние события test допустимы как "
                "контекст, целевые события в контекст не попадают"
            ),
            "history_filter": "все уникальные товары пользователя до момента выдачи",
            "target_events": list(config.target_events),
            "view_role": "контекст, не цель",
        },
        "evaluation": evaluation,
        "example": example,
    }

    save_bundle(
        artifacts_dir,
        popularity=popularity,
        covisitation=covisitation,
        profiles=profiles,
        targets=targets,
        metadata=metadata,
    )
    with (Path(artifacts_dir) / "metrics.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    return metadata


def metrics_table(metadata: Dict[str, Any]) -> str:
    ev = metadata["evaluation"]
    k = ev["k"]
    lines = [
        f"Пользователей в оценке: {ev['evaluated_visitors']} "
        f"(cold start: {ev['cold_start_visitors']})",
        "",
        f"| Модель | Precision@{k} | Recall@{k} | NDCG@{k} | HitRate@{k} |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, title in (
        ("popularity", "Popularity baseline"),
        ("covisitation", "Item-item / co-visitation"),
    ):
        row = ev[name]
        lines.append(
            f"| {title} | {row[f'precision@{k}']:.6f} | "
            f"{row[f'recall@{k}']:.6f} | {row[f'ndcg@{k}']:.6f} | "
            f"{row[f'hit_rate@{k}']:.6f} |"
        )
    return "\n".join(lines)
