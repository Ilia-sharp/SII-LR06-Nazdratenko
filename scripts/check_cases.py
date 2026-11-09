"""Проверяет работающий сервис по эталонным ответам из data/check_cases.json.

По умолчанию делает два прохода и убеждается, что результаты совпадают
с эталоном и между собой.

Примеры::

    python scripts/check_cases.py --url http://127.0.0.1:8000
    python scripts/check_cases.py --url https://sii-lr06-nazdratenko.onrender.com --repeat 2

Код возврата 0 — всё совпало, 1 — есть расхождения или сервис недоступен.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "data" / "check_cases.json"
SCORE_TOLERANCE = 1e-6


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def post_recommend(base_url: str, visitor_id: int, timeout: float) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/api/recommend"
    body = json.dumps(
        {"visitor_id": visitor_id, "n": 10, "model": "auto"}
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def compare(case: Dict[str, Any], actual: Dict[str, Any]) -> List[str]:
    problems: List[str] = []

    if actual.get("model") != case["expected_model"]:
        problems.append(
            f"модель: ожидалась {case['expected_model']}, получена {actual.get('model')}"
        )
    if actual.get("history_size") != case["history_size"]:
        problems.append(
            f"история: ожидалось {case['history_size']}, получено {actual.get('history_size')}"
        )
    actual_context = [int(x) for x in actual.get("context_items") or []]
    if actual_context != case["context_items"]:
        problems.append(
            f"контекст: ожидался {case['context_items']}, получен {actual_context}"
        )

    expected_items = case["top10"]
    actual_items = actual.get("recommendations") or []
    actual_ids = [int(item["itemid"]) for item in actual_items]
    expected_ids = [int(item["itemid"]) for item in expected_items]
    if actual_ids != expected_ids:
        problems.append(
            "порядок товаров: ожидался "
            f"{expected_ids}, получен {actual_ids}"
        )
        return problems

    for position, (expected, received) in enumerate(
        zip(expected_items, actual_items), start=1
    ):
        difference = abs(float(expected["score"]) - float(received["score"]))
        if difference > SCORE_TOLERANCE:
            problems.append(
                f"score на {position}-м месте: ожидался "
                f"{expected['score']}, получен {received['score']}"
            )
    return problems


def short_top(items: List[Dict[str, Any]], limit: int = 3) -> str:
    return ", ".join(str(item["itemid"]) for item in items[:limit])


def run_round(
    base_url: str,
    cases: List[Dict[str, Any]],
    timeout: float,
    number: int,
    total: int,
) -> bool:
    print(f"\nПроход {number}/{total}")
    passed = 0

    for case in cases:
        visitor_id = int(case["visitor_id"])
        try:
            actual = post_recommend(base_url, visitor_id, timeout)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            print(f"  ✗ {visitor_id:>10}  HTTP {error.code}: {detail}")
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"  ✗ {visitor_id:>10}  сервис недоступен: {error}")
            continue

        problems = compare(case, actual)
        if problems:
            print(f"  ✗ {visitor_id:>10}  {short_top(actual.get('recommendations') or [])}")
            for problem in problems:
                print(f"      {problem}")
        else:
            passed += 1
            print(
                f"  ✓ {visitor_id:>10}  {actual.get('model'):<26} "
                f"Top-3: {short_top(actual.get('recommendations') or [])}"
            )

    print(f"  Итог прохода: {passed}/{len(cases)} совпало")
    return passed == len(cases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Проверка сервиса рекомендаций по эталонным ответам."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="адрес сервиса (по умолчанию http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES),
        help="файл с эталонными ответами (по умолчанию data/check_cases.json)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="сколько раз повторить проверку (по умолчанию 2)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="таймаут одного запроса в секундах (по умолчанию 90)",
    )
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()

    if args.repeat < 1:
        print("--repeat должен быть не меньше 1", file=sys.stderr)
        return 2

    cases_path = Path(args.cases)
    with cases_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cases = payload.get("cases") or []
    if not cases:
        print(f"В {cases_path} нет кейсов", file=sys.stderr)
        return 2

    print(f"Проверка сервиса: {args.url}")
    print(f"Кейсов: {len(cases)} · проходов: {args.repeat}")
    print("Эталон построен по artifacts/model.npz.")

    all_ok = True
    for number in range(1, args.repeat + 1):
        round_ok = run_round(
            args.url, cases, args.timeout, number, args.repeat
        )
        all_ok = all_ok and round_ok
        if not round_ok and args.repeat > 1:
            print("Дальнейшие проходы остановлены: есть расхождения.")
            break

    print()
    if all_ok:
        if args.repeat > 1:
            print(
                f"Все {args.repeat} прохода дали одинаковый результат. "
                "Сервис работает стабильно."
            )
        else:
            print("Результат совпал с эталоном.")
        return 0
    print("Результат не совпал с эталоном. Проверьте версию модели или деплой.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
