"""Конфигурация проекта: вариант задания, модель и параметры генерации.

Все значения зафиксированы по варианту 16 (тема «рабочее место»),
чтобы любой прогон был воспроизводимым.
"""
from __future__ import annotations

# ── Вариант задания ──────────────────────────────────────────────────────────
VARIANT: int = 16
THEME: str = "office"          # тема варианта: рабочее место
PROMPT_TYPE: str = "interior"  # тип промпта: интерьер

# ── Модель ───────────────────────────────────────────────────────────────────
MODEL_ID: str = "stabilityai/sd-turbo"
MODEL_LICENSE: str = "Stability AI Community License"
# sd-turbo дистиллирован из SD 2.1 и рассчитан на малое число шагов
# без classifier-free guidance.
STEPS: int = 4
GUIDANCE_SCALE: float = 0.0
WIDTH: int = 512
HEIGHT: int = 512

# ── Воспроизводимость ────────────────────────────────────────────────────────
SEED: int = 116  # seed по варианту (одинаков для всех промптов прогона)

# ── Файлы ────────────────────────────────────────────────────────────────────
MANIFEST_NAME: str = "manifest.json"


def model_defaults() -> dict:
    """Параметры генерации по умолчанию (вариант 16)."""
    return {
        "model_id": MODEL_ID,
        "steps": STEPS,
        "seed": SEED,
        "guidance_scale": GUIDANCE_SCALE,
        "width": WIDTH,
        "height": HEIGHT,
    }
