"""Проверки прогона после генерации:

    python -m src.checks [каталог_с_manifest.json]

Выходной код 0 — все проверки пройдены (или прогона ещё не было:
в CI до первого запуска это нормальная ситуация).
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.validate import validate_run


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs")

    if not (out_dir / "manifest.json").exists():
        print(f"[i] {out_dir}/manifest.json не найден — прогон ещё не "
              f"выполнялся (это не ошибка).")
        return 0

    report = validate_run(out_dir)
    print(f"Манифест  : {report['manifest']}")
    print(f"run_id    : {report.get('run_id')}")
    print(f"Изображений: {report['count']}")

    for r in report["images"]:
        name = Path(r["file"]).name
        if r["ok"]:
            c = ", ".join(k for k, v in r["checks"].items() if v)
            print(f"  ✓ {name}: {c}")
        else:
            print(f"  ✗ {name}: {r.get('error') or r['checks']}")

    if report["ok"]:
        print("[✓] Все проверки пройдены")
        return 0

    print("[✗] Обнаружены проблемы — см. список выше")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
