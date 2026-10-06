"""Проверки результатов генерации (по заданию).

Для каждого файла проверяем:
1. файл существует;
2. размер корректный (не пустой и не подозрительно маленький);
3. это валидный PNG ожидаемого разрешения;
4. изображение не «пустое» (не однотонный шум/чёрный/белый лист).
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat

MIN_SIZE_BYTES = 10_000  # 10 КБ — заведомо больше «сломанной» картинки


def validate_image(
    path: str | Path,
    *,
    min_size_bytes: int = MIN_SIZE_BYTES,
    width: int = 512,
    height: int = 512,
) -> dict:
    """Проверяет один файл и возвращает отчёт с флагами проверок."""
    p = Path(path)
    checks: dict[str, bool] = {}
    brightness: float | None = None

    checks["exists"] = p.exists()
    if not checks["exists"]:
        return {"file": str(p), "ok": False, "checks": checks,
                "error": "файл не существует"}

    size = p.stat().st_size
    checks["not_empty"] = size > 0
    checks["size_ok"] = size > min_size_bytes

    try:
        with Image.open(p) as im:
            im.load()
            checks["is_png"] = im.format == "PNG"
            checks["dimensions_ok"] = im.size == (int(width), int(height))
            gray = im.convert("L")
            brightness = float(ImageStat.Stat(gray).mean[0])
            # «Пустым» считаем почти чёрное или почти белое изображение.
            checks["not_blank"] = 0.5 < brightness < 254.5
    except Exception as exc:  # noqa: BLE001 — любая ошибка = файл невалиден
        return {"file": str(p), "ok": False, "checks": checks,
                "error": f"не удалось открыть как изображение: {exc}"}

    return {
        "file": str(p),
        "ok": all(checks.values()),
        "checks": checks,
        "brightness": round(brightness, 1) if brightness is not None else None,
        "size_bytes": size,
    }


def validate_run(
    out_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict:
    """Проверяет весь прогон: манифест + каждое изображение из items."""
    out = Path(out_dir)
    m_path = Path(manifest_path) if manifest_path else out / "manifest.json"

    if not m_path.exists():
        return {"ok": False, "error": f"{m_path} не найден", "images": []}

    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    results = [
        validate_image(
            out / item["file"],
            width=item.get("width", 512),
            height=item.get("height", 512),
        )
        for item in items
    ]

    return {
        "ok": bool(results) and all(r["ok"] for r in results),
        "manifest": str(m_path),
        "run_id": manifest.get("run_id"),
        "count": len(results),
        "images": results,
    }
