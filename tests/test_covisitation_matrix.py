"""Тесты на целостность матрицы co-visitation и на оценку моделей."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.artifacts import ModelBundle, recommend_for_visitor
from src.covisitation import CovisitationModel, build_covisitation
from src.popularity import PopularityModel
from src.profiles import VisitorProfiles
from src.split import _assert_disjoint_timestamps, time_split


def _sessions(rows):
    """Собирает DataFrame с готовыми session_id."""
    frame = pd.DataFrame(
        rows,
        columns=["session_id", "timestamp", "visitorid", "event", "itemid"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def test_covisitation_row_has_no_duplicate_neighbours():
    """В строке CSR один сосед не может встречаться дважды."""
    frame = _sessions(
        [
            (0, "2015-01-01 10:00", 1, "view", 10),
            (0, "2015-01-01 10:01", 1, "view", 20),
            (0, "2015-01-01 10:02", 1, "view", 30),
            (1, "2015-01-01 11:00", 2, "view", 10),
            (1, "2015-01-01 11:01", 2, "view", 20),
        ]
    )
    model = build_covisitation(frame, max_session_items=50)

    n_items = len(model.item_ids)
    for index in range(n_items):
        neighbors, _ = model.neighbor_slice(int(model.item_ids[index]))
        assert len(neighbors) == len(set(neighbors.tolist()))

    src = np.repeat(np.arange(n_items), np.diff(model.indptr))
    key = src.astype(np.int64) * n_items + model.neighbors.astype(np.int64)
    assert len(np.unique(key)) == len(model.neighbors)


def test_covisitation_ignores_pair_order():
    """Пара, увиденная в сессиях с разным порядком, считается один раз."""
    frame = _sessions(
        [
            (0, "2015-01-01 10:00", 1, "view", 10),
            (0, "2015-01-01 10:01", 1, "view", 20),
            (1, "2015-01-01 11:00", 2, "view", 20),
            (1, "2015-01-01 11:01", 2, "view", 10),
        ]
    )
    model = build_covisitation(frame, max_session_items=50)

    # Одна неупорядоченная пара, встретившаяся дважды.
    assert model.stats["unique_pairs"] == 1
    assert model.stats["pair_occurrences"] == 2
    assert model.stats["mean_neighbors_per_item"] == 1.0

    neighbors, scores = model.neighbor_slice(10)
    assert neighbors.tolist() == [1]
    assert scores.tolist() == [2.0]


def test_covisitation_stats_match_matrix():
    """Статистика совпадает с реальной структурой матрицы."""
    frame = _sessions(
        [
            (0, "2015-01-01 10:00", 1, "view", 10),
            (0, "2015-01-01 10:01", 1, "view", 20),
            (1, "2015-01-01 11:00", 2, "view", 10),
            (1, "2015-01-01 11:01", 2, "view", 30),
        ]
    )
    model = build_covisitation(frame, max_session_items=50)
    n_items = len(model.item_ids)

    assert model.stats["unique_pairs"] == 2
    assert model.stats["mean_neighbors_per_item"] == round(
        len(model.neighbors) / n_items, 2
    )
    assert len(model.neighbors) == 2 * model.stats["unique_pairs"]


def test_covisitation_truncates_long_sessions():
    """Длинная сессия обрезается до последних max_session_items товаров."""
    rows = [
        (0, f"2015-01-01 10:{minute:02d}", 1, "view", 100 + minute)
        for minute in range(6)
    ]
    frame = _sessions(rows)
    model = build_covisitation(frame, max_session_items=3)

    # Три товара в обрезанной сессии дают три пары.
    assert model.stats["pair_occurrences"] == 3
    assert model.stats["unique_pairs"] == 3


def test_time_split_guard_detects_shared_timestamps():
    """Проверка пересечения меток времени ловит общие значения."""
    stamp = pd.Timestamp("2015-01-01 10:00", tz="UTC")
    train = pd.DataFrame({"timestamp": [stamp, stamp]})
    test = pd.DataFrame({"timestamp": [stamp]})

    try:
        _assert_disjoint_timestamps(train, test)
    except AssertionError:
        return
    raise AssertionError("Общие метки времени не обнаружены")


def test_time_split_parts_do_not_overlap(small_events):
    """После разделения метки времени train и test не пересекаются."""
    split = time_split(small_events, test_fraction=0.2)
    train = set(split.train["timestamp"])
    test = set(split.test["timestamp"])
    assert train.isdisjoint(test)


def _bundle_with_unfillable_gap() -> ModelBundle:
    """Пакет, у которого популярность не может дополнить выдачу."""
    popularity = PopularityModel(
        item_ids=np.array([10, 20, 30], dtype=np.int64),
        scores=np.array([100.0, 50.0, 10.0]),
    )
    covisitation = CovisitationModel(
        item_ids=np.array([100, 200], dtype=np.int64),
        indptr=np.array([0, 1, 2], dtype=np.int64),
        neighbors=np.array([1], dtype=np.int32),
        scores=np.array([5.0], dtype=np.float32),
        stats={},
    )
    profiles = VisitorProfiles(
        visitors=np.array([1], dtype=np.int64),
        history_indptr=np.array([0, 3], dtype=np.int64),
        history_items=np.array([10, 20, 30], dtype=np.int64),
        context_indptr=np.array([0, 1], dtype=np.int64),
        context_items=np.array([100], dtype=np.int64),
        context_weights=np.array([1.0]),
    )
    return ModelBundle(
        popularity=popularity,
        covisitation=covisitation,
        profiles=profiles,
        test_visitors=np.empty(0, dtype=np.int64),
        test_indptr=np.array([0], dtype=np.int64),
        test_items=np.empty(0, dtype=np.int64),
        metadata={},
    )


def test_model_name_stays_covisitation_when_nothing_filled():
    """Если популярность ничего не добавила, название модели не меняется."""
    bundle = _bundle_with_unfillable_gap()
    result = recommend_for_visitor(bundle, 1, n=10)

    assert [item["itemid"] for item in result["recommendations"]] == [200]
    assert result["model"] == "covisitation"
