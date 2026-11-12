"""Полный конвейер: загрузка → split → обучение → оценка → артефакты."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from .artifacts import save_bundle, write_metadata
from .config import CONFIG, ExperimentConfig
from .covisitation import build_covisitation
from .data import build_target_sets, first_target_times, load_events
from .metrics import metrics_at_k
from .popularity import build_popularity
from .profiles import build_evaluation_profiles, build_profiles
from .recommend import (
    covisitation_recommend,
    merge_with_popularity,
    popular_recommend,
)
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
    """Считает метрики для обеих моделей на одних и тех же тёплых пользователях.

    Дополнительно считается гибрид ``covisitation + popularity fill`` — именно
    такую выдачу сервис отдаёт в режиме ``auto``, когда co-visitation не
    набирает Top-N. Все три модели оцениваются на одном и том же наборе
    пользователей, поэтому строки таблицы сравнимы между собой.
    """
    model_names = ("popularity", "covisitation", "hybrid")
    rows: Dict[str, list[dict[str, float]]] = {name: [] for name in model_names}
    new_rows: Dict[str, list[dict[str, float]]] = {name: [] for name in model_names}
    catalogs: Dict[str, set[int]] = {name: set() for name in model_names}
    new_catalogs: Dict[str, set[int]] = {name: set() for name in model_names}
    hits: Dict[str, int] = {name: 0 for name in model_names}
    new_hits: Dict[str, int] = {name: 0 for name in model_names}

    cold_start = 0
    no_context = 0
    context_lengths: list[int] = []
    evaluated_target_counts: list[int] = []
    new_target_pairs = 0
    all_target_pairs = 0
    filled_users = 0
    # Разбивка по длине контекста — показывает, где модель сильнее.
    by_context: Dict[int, list[dict[str, float]]] = {}

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
        # Гибрид: co-visitation, а при нехватке кандидатов — популярность.
        rec_hybrid = merge_with_popularity(rec_cov, popularity, history, n=k)
        if len(rec_hybrid) > len(rec_cov):
            filled_users += 1
        recs = {
            "popularity": rec_pop,
            "covisitation": rec_cov,
            "hybrid": rec_hybrid,
        }

        for name, rec in recs.items():
            ids = [int(itemid) for itemid, _ in rec]
            rows[name].append(metrics_at_k(ids, target_set, k))
            catalogs[name].update(ids)
            hits[name] += int(any(i in target_set for i in ids))

        cov_metrics = metrics_at_k(
            [int(itemid) for itemid, _ in rec_cov], target_set, k
        )
        by_context.setdefault(int(len(context)), []).append(cov_metrics)

        # Вспомогательный срез: цели, которых ещё не было в истории.
        # Именно такие товары система в принципе может рекомендовать.
        new_targets = target_set - history_set
        if new_targets:
            for name, rec in recs.items():
                ids = [int(itemid) for itemid, _ in rec]
                new_rows[name].append(metrics_at_k(ids, new_targets, k))
                new_catalogs[name].update(ids)
                new_hits[name] += int(any(i in new_targets for i in ids))
            new_target_pairs += len(new_targets)

    evaluated = len(rows["covisitation"])

    def summarize(
        rows_: list[dict[str, float]],
        catalog: set[int],
        hit_count: int,
        count: int,
    ):
        return {
            f"precision@{k}": round(_average([r[f"precision@{k}"] for r in rows_]), 6),
            f"recall@{k}": round(_average([r[f"recall@{k}"] for r in rows_]), 6),
            f"ndcg@{k}": round(_average([r[f"ndcg@{k}"] for r in rows_]), 6),
            f"hit_rate@{k}": round(hit_count / count, 6) if count else 0.0,
            "catalog_coverage": len(catalog),
        }

    context_breakdown = []
    for length in sorted(by_context):
        bucket = by_context[length]
        context_breakdown.append(
            {
                "context_length": length,
                "visitors": len(bucket),
                f"precision@{k}": round(
                    _average([r[f"precision@{k}"] for r in bucket]), 6
                ),
                f"recall@{k}": round(
                    _average([r[f"recall@{k}"] for r in bucket]), 6
                ),
                f"ndcg@{k}": round(
                    _average([r[f"ndcg@{k}"] for r in bucket]), 6
                ),
            }
        )

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
        "popularity": summarize(
            rows["popularity"],
            catalogs["popularity"],
            hits["popularity"],
            evaluated,
        ),
        "covisitation": summarize(
            rows["covisitation"],
            catalogs["covisitation"],
            hits["covisitation"],
            evaluated,
        ),
        "hybrid": summarize(
            rows["hybrid"], catalogs["hybrid"], hits["hybrid"], evaluated
        ),
        "hybrid_filled_users": filled_users,
        "covisitation_by_context_length": context_breakdown,
        "new_targets": {
            "target_pairs": new_target_pairs,
            "all_target_pairs": all_target_pairs,
            "share_already_in_history": round(
                1.0 - new_target_pairs / all_target_pairs, 6
            )
            if all_target_pairs
            else 0.0,
            "evaluated_visitors": len(new_rows["covisitation"]),
            "popularity": summarize(
                new_rows["popularity"],
                new_catalogs["popularity"],
                new_hits["popularity"],
                len(new_rows["popularity"]),
            ),
            "covisitation": summarize(
                new_rows["covisitation"],
                new_catalogs["covisitation"],
                new_hits["covisitation"],
                len(new_rows["covisitation"]),
            ),
            "hybrid": summarize(
                new_rows["hybrid"],
                new_catalogs["hybrid"],
                new_hits["hybrid"],
                len(new_rows["hybrid"]),
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
    timings: Dict[str, float] = {}
    started = time.perf_counter()

    def stage(name: str, message: str, work):
        """Выполняет этап, печатает сообщение и запоминает длительность."""
        progress(message)
        step = time.perf_counter()
        result = work()
        timings[name] = round(time.perf_counter() - step, 3)
        return result

    loaded = stage(
        "load",
        "1/7 Загрузка и проверка events.csv",
        lambda: load_events(events_path),
    )
    events = loaded.frame

    split = stage(
        "split",
        "2/7 Временное разделение train/test",
        lambda: time_split(events, config.test_fraction),
    )

    train_sessions = stage(
        "sessions",
        "3/7 Определение сессий на train",
        lambda: assign_sessions(split.train, config.session_gap_minutes),
    )
    sessions = session_report(train_sessions)

    popularity = stage(
        "popularity",
        "4/7 Популярностный baseline",
        lambda: build_popularity(split.train),
    )

    covisitation = stage(
        "covisitation",
        "5/7 Item-item / co-visitation",
        lambda: build_covisitation(train_sessions, config.max_session_items),
    )

    def build_targets():
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
        return profiles, targets, targets_report, eval_profiles

    profiles, targets, targets_report, eval_profiles = stage(
        "profiles",
        "6/7 Профили посетителей и целевые события теста",
        build_targets,
    )

    def evaluate_and_build_example():
        evaluation = evaluate_models(
            popularity, covisitation, eval_profiles, targets, config.top_n
        )
        example = build_example(
            popularity, covisitation, profiles, targets, config.top_n
        )
        return evaluation, example

    evaluation, example = stage(
        "evaluation",
        "7/7 Оценка Precision/Recall/NDCG и сохранение артефактов",
        evaluate_and_build_example,
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

    save_started = time.perf_counter()
    save_bundle(
        artifacts_dir,
        popularity=popularity,
        covisitation=covisitation,
        profiles=profiles,
        targets=targets,
        metadata=metadata,
    )
    timings["save"] = round(time.perf_counter() - save_started, 3)
    timings["total"] = round(time.perf_counter() - started, 3)
    metadata["timings_seconds"] = timings
    write_metadata(artifacts_dir, metadata)

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
        ("hybrid", "Co-visitation + popularity fill"),
    ):
        row = ev[name]
        lines.append(
            f"| {title} | {row[f'precision@{k}']:.6f} | "
            f"{row[f'recall@{k}']:.6f} | {row[f'ndcg@{k}']:.6f} | "
            f"{row[f'hit_rate@{k}']:.6f} |"
        )
    return "\n".join(lines)
