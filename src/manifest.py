"""Манифест прогона: метаданные о модели, seed и каждом изображении.

Формат manifest.json (по заданию):

.. code-block:: json

    {
      "run_id": "20261006-1119",
      "model_id": "stabilityai/sd-turbo",
      "model_version": "<sha ревизии весов>",
      "steps": 4,
      "items": [
        {"file": "img_000.png", "prompt": "...", "seed": 116,
         "elapsed_sec": 17.37}
      ]
    }

Дополнительно к обязательным полям item содержит width, height и
size_bytes — это упрощает автоматические проверки.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    """ID прогона в формате YYYYMMDD-HHMM (локальное время)."""
    return datetime.now().strftime("%Y%m%d-%H%M")


def new_manifest(
    model_id: str,
    model_version: str,
    steps: int,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Создаёт пустой манифест прогона."""
    return {
        "run_id": run_id or new_run_id(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": model_id,
        "model_version": model_version,
        "steps": int(steps),
        "items": [],
    }


def add_item(
    manifest: dict[str, Any],
    *,
    file: str,
    prompt: str,
    seed: int,
    elapsed_sec: float,
    width: int = 512,
    height: int = 512,
    size_bytes: int | None = None,
) -> dict[str, Any]:
    """Добавляет запись об одном сгенерированном изображении."""
    item = {
        "file": file,
        "prompt": prompt,
        "seed": int(seed),
        "elapsed_sec": round(float(elapsed_sec), 2),
        "width": int(width),
        "height": int(height),
        "size_bytes": int(size_bytes) if size_bytes is not None else None,
    }
    manifest["items"].append(item)
    return item


def save_manifest(manifest: dict[str, Any], out_dir: str | Path) -> Path:
    """Атомарно сохраняет манифест в ``out_dir/manifest.json``.

    Сначала запись во временный файл в том же каталоге, затем
    ``os.replace`` — так файл не может оказаться «наполовину записанным»,
    даже если процесс прервут.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final = out / "manifest.json"

    fd, tmp_path = tempfile.mkstemp(dir=out, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, final)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return final


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Читает manifest.json и возвращает словарь."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Манифест не найден: {p}")
    with p.open(encoding="utf-8") as f:
        return json.load(f)
