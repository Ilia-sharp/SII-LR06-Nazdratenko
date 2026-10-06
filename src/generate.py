"""CLI-генерация изображений — запуск строго по заданию:

    python -m src.generate --prompts prompts.txt --out outputs/

Сохраняет img_000.png, img_001.png, … и manifest.json. Манифест
перезаписывается после каждой генерации, поэтому даже при сбое в
середине прогона уже готовые результаты не потеряются.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config import (
    GUIDANCE_SCALE,
    HEIGHT,
    MANIFEST_NAME,
    MODEL_ID,
    SEED,
    STEPS,
    WIDTH,
)
from src.manifest import add_item, new_manifest, save_manifest
from src.model import encode_prompts, generate_image, load_pipeline, torch_available
from src.prompts import read_prompts
from src.validate import validate_image


def resolve_model_version() -> str:
    """SHA-ревизия весов в Hugging Face Hub (или 'main' без сети)."""
    try:
        from huggingface_hub import model_info
        return model_info(MODEL_ID).sha or "main"
    except Exception:
        return "main"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.generate",
        description="Генерация изображений конвейером диффузии (вариант 16)",
    )
    parser.add_argument("--prompts", default="prompts.txt",
                        help="файл с промптами, по одному на строку")
    parser.add_argument("--out", default="outputs/",
                        help="каталог для изображений и manifest.json")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"seed генератора (по варианту — {SEED})")
    parser.add_argument("--steps", type=int, default=STEPS,
                        help=f"число шагов диффузии (sd-turbo — {STEPS})")
    parser.add_argument("--limit", type=int, default=0,
                        help="взять только первые N промптов (0 — все)")
    args = parser.parse_args(argv)

    if not torch_available():
        print("[ошибка] torch не установлен: pip install -r requirements.txt")
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = read_prompts(args.prompts)
    if args.limit > 0:
        prompts = prompts[: args.limit]

    print(f"[i] Модель : {MODEL_ID}")
    print(f"[i] Seed   : {args.seed} | шагов: {args.steps} | "
          f"guidance: {GUIDANCE_SCALE} | {WIDTH}x{HEIGHT}")
    print(f"[i] Промптов: {len(prompts)} | каталог: {out_dir}/")

    manifest = new_manifest(MODEL_ID, resolve_model_version(), args.steps)
    t_start = time.perf_counter()

    # Фаза 1: кодируем ВСЕ промпты разом — после этого текстовый энкодер
    # выгружается один раз, и шаги диффузии идут без него (экономия RAM).
    pipe = load_pipeline()
    print("[i] Кодирование промптов (CLIP)…")
    embeds_list = encode_prompts(pipe, prompts)

    # Фаза 2: генерация изображений из готовых эмбеддингов.
    for i, prompt in enumerate(prompts):
        file_name = f"img_{i:03d}.png"
        print(f"\n[{i + 1}/{len(prompts)}] seed={args.seed} → {file_name}")
        short = prompt if len(prompt) <= 76 else prompt[:76] + "…"
        print(f"    {short}")

        image, elapsed = generate_image(
            pipe,
            seed=args.seed,
            steps=args.steps,
            width=WIDTH,
            height=HEIGHT,
            prompt=prompt,
            embeds=embeds_list[i] if embeds_list is not None else None,
        )
        path = out_dir / file_name
        image.save(path, format="PNG")

        report = validate_image(path, width=WIDTH, height=HEIGHT)
        add_item(
            manifest,
            file=file_name,
            prompt=prompt,
            seed=args.seed,
            elapsed_sec=elapsed,
            width=WIDTH,
            height=HEIGHT,
            size_bytes=path.stat().st_size,
        )
        save_manifest(manifest, out_dir)

        status = "OK" if report["ok"] else f"ПРОВЕРКИ НЕ ПРОЙДЕНЫ: {report.get('checks')}"
        print(f"    готово за {elapsed:.2f} с | "
              f"{path.stat().st_size // 1024} КБ | проверки: {status}")

    total = time.perf_counter() - t_start
    per_image = total / len(prompts)
    print(f"\n[✓] Готово: {len(prompts)} изображений за {total:.1f} с "
          f"(в среднем {per_image:.1f} с/изображение)")
    print(f"[✓] Манифест: {out_dir / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
