"""Чтение промптов из текстового файла.

Формат prompts.txt: один промпт на строку. Пустые строки и строки,
начинающиеся с ``#`` (комментарии), игнорируются.
"""
from __future__ import annotations

from pathlib import Path


def read_prompts(path: str | Path) -> list[str]:
    """Возвращает список непустых промптов из файла ``path``.

    Raises:
        FileNotFoundError: файл не существует.
        ValueError: в файле нет ни одного промпта.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл промптов не найден: {p}")

    prompts: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)

    if not prompts:
        raise ValueError(f"В файле {p} нет ни одного промпта")
    return prompts
