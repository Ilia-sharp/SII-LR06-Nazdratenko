"""CLI: обучение, оценка, выдача рекомендаций и запуск HTTP API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import load_bundle, recommend_for_items, recommend_for_visitor
from .pipeline import metrics_table, train_and_evaluate


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_train(args: argparse.Namespace) -> int:
    metadata = train_and_evaluate(
        events_path=args.events,
        artifacts_dir=args.artifacts,
        progress=lambda message: print(f"[train] {message}", flush=True),
    )
    print()
    print(metrics_table(metadata))
    print()
    print(f"Артефакты: {Path(args.artifacts).resolve()}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.artifacts)
    print(metrics_table(bundle.metadata))
    if args.json:
        _print_json(bundle.metadata["evaluation"])
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.artifacts)
    if args.visitor_id is not None:
        payload = recommend_for_visitor(
            bundle,
            visitor_id=args.visitor_id,
            n=args.n,
            fallback=not args.no_fallback,
            model=args.model,
        )
    elif args.items:
        item_ids = [
            int(part)
            for part in ",".join(args.items).split(",")
            if part.strip()
        ]
        payload = recommend_for_items(
            bundle, item_ids=item_ids, n=args.n, fallback=not args.no_fallback
        )
    else:
        print("Укажите --visitor-id или --items", file=sys.stderr)
        return 2

    if args.json:
        _print_json(payload)
    else:
        print(f"Модель: {payload['model']}")
        print(f"Контекст: {payload['context_items']}")
        print(f"История: {payload['history_size']} товаров")
        for rank, rec in enumerate(payload["recommendations"], start=1):
            print(f"{rank:>2}. itemid={rec['itemid']:<10} score={rec['score']}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    app = create_app(args.artifacts)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Рекомендательный сервис Retailrocket (вариант 16).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="обучить модели и посчитать метрики")
    train.add_argument("--events", default="data/events.csv")
    train.add_argument("--artifacts", default="artifacts")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="показать метрики обученной модели")
    evaluate.add_argument("--artifacts", default="artifacts")
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    recommend = sub.add_parser("recommend", help="Top-N по visitorid или itemid")
    recommend.add_argument("--artifacts", default="artifacts")
    recommend.add_argument("--visitor-id", type=int, default=None)
    recommend.add_argument(
        "--items",
        nargs="+",
        default=None,
        help="itemid текущей сессии через запятую или пробел",
    )
    recommend.add_argument("--n", type=int, default=10)
    recommend.add_argument("--model", choices=["auto", "covisitation", "popularity"],
                           default="auto")
    recommend.add_argument("--no-fallback", action="store_true")
    recommend.add_argument("--json", action="store_true")
    recommend.set_defaults(func=cmd_recommend)

    serve = sub.add_parser("serve", help="запустить HTTP API (FastAPI)")
    serve.add_argument("--artifacts", default="artifacts")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
