"""Готовит данные для проверки работы сервиса.

Скрипт читает закоммиченные артефакты ``artifacts/`` и сохраняет:

* ``data/check_cases.json`` — эталонные ответы API для набора visitorid;
* ``CHECK.md`` — человекочитаемую инструкцию с теми же ответами.

Запуск из корня репозитория::

    python scripts/make_check_cases.py

Пересобирать файлы нужно только если поменялась модель в ``artifacts/``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifacts import (  # noqa: E402
    load_bundle,
    recommend_for_visitor,
    test_targets_dict,
)

CASES_PATH = ROOT / "data" / "check_cases.json"
CHECK_PATH = ROOT / "CHECK.md"

# visitorid и пояснение, зачем этот кейс нужен.
# Все реальные ID удовлетворяют условию варианта: visitorid % 30 = 15.
CASES: List[Dict[str, Any]] = [
    {
        "visitor_id": 582525,
        "why": "Пример с сайта: два просмотра; покупка из теста попала на 8-е место.",
    },
    {
        "visitor_id": 795975,
        "why": "Два просмотра в контексте; покупка из теста попала на 5-е место.",
    },
    {
        "visitor_id": 1374225,
        "why": "Один товар в истории: co-visitation плюс популярностный fallback, покупка на 1-м месте.",
    },
    {
        "visitor_id": 1181805,
        "why": "Целевой покупатель без истории в train: cold start, популярные товары, покупка на 1-м месте.",
    },
    {
        "visitor_id": 896175,
        "why": "В последнем событии товар из корзины (addtocart, вес 3).",
    },
    {
        "visitor_id": 1258035,
        "why": "В последних событиях две покупки (transaction, вес 5).",
    },
    {
        "visitor_id": 1134975,
        "why": "Длинная история (152 товара) и полный контекст из 5 событий, включая корзину.",
    },
    {
        "visitor_id": 645525,
        "why": "Очень длинная история (144 товара) и полный контекст из 5 просмотров.",
    },
    {
        "visitor_id": 1170165,
        "why": "Полный контекст из 5 последних товаров, чистая co-visitation.",
    },
    {
        "visitor_id": 801975,
        "why": "История из 9 товаров и контекст из 3 событий.",
    },
    {
        "visitor_id": 1125765,
        "why": "История из 11 товаров и контекст из 4 событий.",
    },
    {
        "visitor_id": 16725,
        "why": "Короткая история и контекст из 4 событий.",
    },
]


def build_case(bundle, visitor_id: int, why: str) -> Dict[str, Any]:
    payload = recommend_for_visitor(bundle, visitor_id, n=10)
    recommendations = [
        {"itemid": item["itemid"], "score": item["score"]}
        for item in payload["recommendations"]
    ]
    return {
        "visitor_id": visitor_id,
        "visitorid_mod_30": visitor_id % 30,
        "target_group": visitor_id % 30 == 15,
        "why": why,
        "expected_model": payload["model"],
        "history_size": payload["history_size"],
        "context_items": payload["context_items"],
        "top10": recommendations,
    }


def add_targets(case: Dict[str, Any], targets: Dict[int, Any]) -> None:
    target = targets.get(case["visitor_id"])
    items = [int(x) for x in target] if target is not None else []
    case["target_items"] = items
    hit_rank = None
    recommended = [item["itemid"] for item in case["top10"]]
    for item in items:
        if item in recommended:
            rank = recommended.index(item) + 1
            if hit_rank is None or rank < hit_rank:
                hit_rank = rank
    case["target_hit_rank"] = hit_rank


def dump_json(cases: List[Dict[str, Any]], bundle) -> None:
    evaluation = bundle.metadata.get("evaluation", {})
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variant": bundle.metadata.get("variant", {}).get("number", 16),
        "how_to_check": (
            "POST /api/recommend с телом "
            '{"visitor_id": <id>, "n": 10, "model": "auto"}'
        ),
        "cases": cases,
        "metrics": {
            "popularity": evaluation.get("popularity"),
            "covisitation": evaluation.get("covisitation"),
            "hybrid": evaluation.get("hybrid"),
        },
    }
    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CASES_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def fmt_items(items: List[Dict[str, Any]], limit: int | None = None) -> str:
    shown = items if limit is None else items[:limit]
    text = ", ".join(str(item["itemid"]) for item in shown)
    if limit is not None and len(items) > limit:
        text += ", …"
    return text


def dump_markdown(cases: List[Dict[str, Any]], bundle) -> None:
    evaluation = bundle.metadata.get("evaluation", {})
    cov = evaluation.get("covisitation", {})
    pop = evaluation.get("popularity", {})

    lines: List[str] = [
        "# Проверка работы сервиса",
        "",
        "Эта инструкция и файл `data/check_cases.json` сгенерированы из",
        "закоммиченной модели `artifacts/model.npz`. Поэтому ответы на сайте",
        "должны совпадать с таблицами ниже.",
        "",
        "Все ID в наборе входят в целевую группу варианта 16:",
        "`visitorid % 30 = 15`.",
        "",
        "## Как проверить вручную (2 раза)",
        "",
        "1. Откройте <https://sii-lr06-nazdratenko.onrender.com/>.",
        "2. Введите `visitorid` из таблицы или нажмите его карточку в блоке",
        "   «Покупатели для проверки». Оставьте «10 товаров» и",
        "   «Автоматически», нажмите «Подобрать».",
        "3. Сверьте первые 3 товара и число «История» с таблицей.",
        "   В интерфейсе у товара есть условное название, а под ним —",
        "   настоящий `itemid`; сверяйте именно `itemid`.",
        "4. Повторите для всех ID — это первый проход.",
        "5. Обновите страницу и повторите те же ID — это второй проход.",
        "   Результаты должны быть точно такими же.",
        "",
        "> Важно: важен порядок товаров. Score в полных таблицах ниже тоже",
        "> должен совпадать (в интерфейсе он округлён до двух знаков).",
        "",
        "## Сводная таблица",
        "",
        "| visitorid | Что проверяем | Модель | История | Ожидаемые первые 3 | 1-й проход | 2-й проход |",
        "|---:|---|---|---:|---|---|:---:|",
    ]

    for case in cases:
        lines.append(
            f"| `{case['visitor_id']}` | {case['why']} | "
            f"`{case['expected_model']}` | {case['history_size']} | "
            f"{fmt_items(case['top10'], 3)} | ☐ | ☐ |"
        )

    lines += [
        "",
        f"Метрики модели на тесте: co-visitation Precision@10 = "
        f"{cov.get('precision@10', 0):.6f}, Recall@10 = "
        f"{cov.get('recall@10', 0):.6f}, NDCG@10 = "
        f"{cov.get('ndcg@10', 0):.6f}; popularity Precision@10 = "
        f"{pop.get('precision@10', 0):.6f}.",
        "",
        "## Полная ожидаемая выдача",
        "",
    ]

    for case in cases:
        lines.append(f"### visitorid {case['visitor_id']}")
        lines.append("")
        lines.append(case["why"])
        lines.append("")
        lines.append(
            f"Модель: `{case['expected_model']}` · история: "
            f"{case['history_size']} · контекст: "
            f"{case['context_items'] or 'пусто'}"
        )
        if case["target_items"]:
            hit = case["target_hit_rank"]
            hit_text = f", попадание на {hit}-м месте" if hit else ", попадания нет"
            lines.append(
                "Целевые товары из теста: "
                + ", ".join(str(x) for x in case["target_items"])
                + hit_text
                + "."
            )
        lines += [
            "",
            "| # | itemid | score |",
            "|---:|---:|---:|",
        ]
        for rank, item in enumerate(case["top10"], start=1):
            mark = " ✅" if item["itemid"] in case["target_items"] else ""
            lines.append(f"| {rank} | {item['itemid']}{mark} | {item['score']} |")
        lines.append("")

    lines += [
        "## Автоматическая проверка (сразу 2 прохода)",
        "",
        "```bash",
        "# локальный сервис",
        "python -m src.cli serve --artifacts artifacts --port 8000",
        "python scripts/check_cases.py --url http://127.0.0.1:8000 --repeat 2",
        "",
        "# или сразу опубликованный сервис",
        "python scripts/check_cases.py "
        "--url https://sii-lr06-nazdratenko.onrender.com --repeat 2",
        "```",
        "",
        "Скрипт сравнит модель, историю, контекст, порядок товаров и score.",
        "Он завершится с кодом 1, если хотя бы один ответ отличается.",
        "",
    ]

    CHECK_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    bundle = load_bundle(ROOT / "artifacts")
    targets = test_targets_dict(bundle)

    cases = []
    for definition in CASES:
        visitor_id = int(definition["visitor_id"])
        case = build_case(bundle, visitor_id, definition["why"])
        add_targets(case, targets)
        cases.append(case)

    outside = [case["visitor_id"] for case in cases if not case["target_group"]]
    if outside:
        raise ValueError(
            "Демонстрационные ID должны входить в целевую группу "
            f"visitorid % 30 = 15; нарушители: {outside}"
        )

    dump_json(cases, bundle)
    dump_markdown(cases, bundle)

    print(f"Готово: {CASES_PATH.relative_to(ROOT)}")
    print(f"Готово: {CHECK_PATH.relative_to(ROOT)}")
    for case in cases:
        print(
            f"  {case['visitor_id']:>10}  {case['expected_model']:<26} "
            f"Top-3: {fmt_items(case['top10'], 3)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
