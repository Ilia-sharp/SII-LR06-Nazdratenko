"""Скачивание events.csv через Kaggle CLI (требуется настроенный kaggle.json)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="каталог для events.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "events.csv"
    if target.exists():
        print(f"Уже существует: {target} ({target.stat().st_size} байт)")
        return 0

    kaggle = shutil.which("kaggle")
    if not kaggle:
        print(
            "Kaggle CLI не найден. Установите `pip install kaggle`, положите "
            "kaggle.json в ~/.kaggle/ и повторите, либо скачайте архив вручную "
            "с https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset",
            file=sys.stderr,
        )
        return 1

    archive = out / "ecommerce-dataset.zip"
    subprocess.run(
        [
            kaggle,
            "datasets",
            "download",
            "-d",
            "retailrocket/ecommerce-dataset",
            "-p",
            str(out),
            "-f",
            "events.csv",
        ],
        check=True,
    )
    # Kaggle может сохранить файл как events.csv.zip или events.csv напрямую.
    downloaded = out / "events.csv.zip"
    if downloaded.exists():
        with zipfile.ZipFile(downloaded) as zf:
            zf.extractall(out)
        downloaded.unlink()
    if archive.exists():
        archive.unlink()

    if not target.exists():
        print("Не удалось найти events.csv после скачивания", file=sys.stderr)
        return 1
    print(f"Готово: {target} ({target.stat().st_size} байт)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
