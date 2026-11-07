"""Популярностный baseline: сумма весов событий по itemid на train."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EVENT_WEIGHTS


@dataclass
class PopularityModel:
    item_ids: np.ndarray
    scores: np.ndarray

    def top(self, n: int) -> list[tuple[int, float]]:
        n = max(0, min(n, len(self.item_ids)))
        return [
            (int(i), float(s))
            for i, s in zip(self.item_ids[:n], self.scores[:n])
        ]


def build_popularity(events: pd.DataFrame) -> PopularityModel:
    """Считает популярность как сумму весов: view=1, addtocart=3, transaction=5.

    Сортировка детерминированная: сначала по убыванию score, при равенстве —
    по возрастанию itemid.
    """
    weights = events["event"].map(EVENT_WEIGHTS)
    if weights.isna().any():
        bad = sorted(events.loc[weights.isna(), "event"].unique().tolist())
        raise ValueError(f"Неизвестные события для весов: {bad}")

    frame = pd.DataFrame(
        {"itemid": events["itemid"].to_numpy(), "weight": weights.to_numpy()}
    )
    grouped = (
        frame.groupby("itemid", sort=False)["weight"]
        .sum()
        .reset_index()
        .sort_values(["weight", "itemid"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    return PopularityModel(
        item_ids=grouped["itemid"].to_numpy(dtype=np.int64),
        scores=grouped["weight"].to_numpy(dtype=np.float64),
    )
