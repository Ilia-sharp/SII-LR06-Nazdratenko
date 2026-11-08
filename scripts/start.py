"""Запуск HTTP API на Render (или локально).

Render передаёт порт через переменную окружения PORT. Слушаем 0.0.0.0,
чтобы health-check гарантированно нашёл сервис.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    artifacts = os.environ.get("ARTIFACTS_DIR", "artifacts")
    print(
        f"SII-LR06 recommender: uvicorn 0.0.0.0:{port}, artifacts={artifacts}",
        flush=True,
    )
    os.environ.setdefault("ARTIFACTS_DIR", artifacts)
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
