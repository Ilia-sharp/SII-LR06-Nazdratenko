"""Предзагрузка весов sd-turbo в локальный кэш Hugging Face.

После этого первый запуск генерации не требует сети:

    python scripts/download_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_ID  # noqa: E402


def main() -> None:
    from huggingface_hub import snapshot_download

    print(f"[i] Скачиваю веса {MODEL_ID} (вариант fp16, ~2.6 ГБ)…")
    path = snapshot_download(
        MODEL_ID,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.md",
            "*fp16.safetensors",
        ],
    )
    print(f"[✓] Веса в кэше: {path}")


if __name__ == "__main__":
    main()
