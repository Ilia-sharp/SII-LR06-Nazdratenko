"""Параметры эксперимента.

Лабораторная работа: рекомендательные ИИ-системы.
Вариант 16: ``visitorid % 30 = 15`` — «Один источник истории».
Индивидуальное условие: контекст строится только по последним событиям
самого пользователя (одного источника), без смешивания с чужими сессиями.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple

# --- Идентификация варианта -------------------------------------------------
VARIANT = 16
TARGET_VISITOR_MOD = 30
TARGET_VISITOR_REMAINDER = 15  # 16 - 1

# --- Веса событий основного эксперимента (учебные, не Retailrocket) ---------
EVENT_WEIGHTS: Dict[str, float] = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}

# Целевые события для оценки: просмотр — только контекст.
TARGET_EVENTS: Tuple[str, ...] = ("addtocart", "transaction")
CONTEXT_EVENTS: Tuple[str, ...] = ("view", "addtocart", "transaction")

# --- Подготовка данных ------------------------------------------------------
SESSION_GAP_MINUTES = 30
TEST_FRACTION = 0.20
CONTEXT_SIZE = 5          # последние 5 событий пользователя (вариант 16)
TOP_N = 10
MAX_SESSION_ITEMS = 50    # защита от взрыва пар в очень длинных сессиях
RANDOM_SEED = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """Все воспроизводимые параметры прогона."""

    variant: int = VARIANT
    target_mod: int = TARGET_VISITOR_MOD
    target_remainder: int = TARGET_VISITOR_REMAINDER
    event_weights: Dict[str, float] = field(
        default_factory=lambda: dict(EVENT_WEIGHTS)
    )
    target_events: Tuple[str, ...] = TARGET_EVENTS
    context_events: Tuple[str, ...] = CONTEXT_EVENTS
    session_gap_minutes: int = SESSION_GAP_MINUTES
    test_fraction: float = TEST_FRACTION
    context_size: int = CONTEXT_SIZE
    top_n: int = TOP_N
    max_session_items: int = MAX_SESSION_ITEMS
    random_seed: int = RANDOM_SEED
    context_source: str = "visitor_history"

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


CONFIG = ExperimentConfig()
