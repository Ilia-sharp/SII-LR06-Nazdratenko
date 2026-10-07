"""Unit-тесты конвейера (без генерации — torch не требуется)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.config import (
    GUIDANCE_SCALE,
    HEIGHT,
    MODEL_ID,
    PROMPT_TYPE,
    SEED,
    STEPS,
    THEME,
    VARIANT,
    WIDTH,
    model_defaults,
)
from src.manifest import add_item, load_manifest, new_manifest, new_run_id, save_manifest
from src.prompts import read_prompts
from src.validate import validate_image, validate_run


# ── config ──────────────────────────────────────────────────────────────────

def test_config_variant():
    """Вариант 16: тема office/рабочее место, тип промпта interior."""
    assert VARIANT == 16
    assert THEME == "office"
    assert PROMPT_TYPE == "interior"


def test_config_seed_and_steps():
    """Seed по варианту — 116; sd-turbo работает за 4 шага."""
    assert SEED == 116
    assert STEPS == 4


def test_config_guidance_zero():
    """sd-turbo требует guidance_scale = 0.0."""
    assert GUIDANCE_SCALE == 0.0


def test_config_model_and_size():
    """Открытая модель — sd-turbo, разрешение 512×512."""
    assert MODEL_ID == "stabilityai/sd-turbo"
    assert (WIDTH, HEIGHT) == (512, 512)


def test_config_model_defaults():
    """model_defaults() возвращает согласованный словарь."""
    d = model_defaults()
    assert d["model_id"] == MODEL_ID
    assert d["seed"] == SEED and d["steps"] == STEPS
    assert d["guidance_scale"] == GUIDANCE_SCALE


# ── prompts ─────────────────────────────────────────────────────────────────

def test_prompts_read_basic(tmp_path: Path):
    """Чтение промптов: непустые строки попадают в список."""
    f = tmp_path / "p.txt"
    f.write_text("first prompt\nsecond prompt\n", encoding="utf-8")
    assert read_prompts(f) == ["first prompt", "second prompt"]


def test_prompts_ignores_comments_and_blank(tmp_path: Path):
    """Строки с # и пустые игнорируются."""
    f = tmp_path / "p.txt"
    f.write_text("# комментарий\n\n  \nprompt one\n   # ещё комментарий\nprompt two\n",
                 encoding="utf-8")
    assert read_prompts(f) == ["prompt one", "prompt two"]


def test_prompts_missing_file_raises(tmp_path: Path):
    """Отсутствующий файл → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_prompts(tmp_path / "nope.txt")


def test_prompts_project_file_has_five():
    """В prompts.txt проекта — 5 промптов (по заданию 3–5)."""
    root = Path(__file__).resolve().parent.parent
    prompts = read_prompts(root / "prompts.txt")
    assert 3 <= len(prompts) <= 5


# ── manifest ────────────────────────────────────────────────────────────────

def test_manifest_new_fields():
    """Новый манифест содержит обязательные поля формата задания."""
    m = new_manifest(MODEL_ID, "main", STEPS, run_id="20260101-1200")
    for key in ("run_id", "model_id", "model_version", "steps", "items"):
        assert key in m
    assert m["items"] == []
    assert m["model_id"] == MODEL_ID


def test_manifest_run_id_format():
    """run_id имеет формат YYYYMMDD-HHMM."""
    rid = new_run_id()
    assert len(rid) == 13 and rid[8] == "-"
    int(rid.replace("-", ""))


def test_manifest_add_item():
    """add_item добавляет запись с промптом, seed и временем."""
    m = new_manifest(MODEL_ID, "main", 4)
    add_item(m, file="img_000.png", prompt="test", seed=116,
             elapsed_sec=1.234, width=512, height=512, size_bytes=123)
    assert len(m["items"]) == 1
    it = m["items"][0]
    assert it["file"] == "img_000.png"
    assert it["seed"] == 116
    assert it["elapsed_sec"] == 1.23


def test_manifest_save_load_roundtrip(tmp_path: Path):
    """Атомарное сохранение: файл читается, данные совпадают."""
    m = new_manifest(MODEL_ID, "main", 4, run_id="r1")
    add_item(m, file="img_000.png", prompt="p", seed=116, elapsed_sec=1.0)
    path = save_manifest(m, tmp_path)
    assert path.name == "manifest.json"
    assert load_manifest(path)["run_id"] == "r1"
    # временных файлов не осталось
    assert [p.name for p in tmp_path.glob("*.tmp")] == []


# ── validate ────────────────────────────────────────────────────────────────

def _make_png(path: Path, size=(512, 512), color=None):
    """Создаёт тестовый PNG: шум (валидный) или заливку (однотонный)."""
    import random
    if color is None:
        img = Image.new("RGB", size)
        px = img.load()
        rnd = random.Random(116)
        for y in range(0, size[1], 4):
            for x in range(0, size[0], 4):
                v = rnd.randint(40, 215)
                for dy in range(4):
                    for dx in range(4):
                        if x + dx < size[0] and y + dy < size[1]:
                            px[x + dx, y + dy] = (v, v, v)
    else:
        img = Image.new("RGB", size, color)
    img.save(path, format="PNG")


def test_validate_good_image(tmp_path: Path):
    """Шумовой PNG 512×512 проходит все проверки."""
    p = tmp_path / "img_000.png"
    _make_png(p)
    report = validate_image(p, width=512, height=512)
    assert report["ok"], report
    assert report["checks"]["is_png"]
    assert report["checks"]["dimensions_ok"]
    assert report["checks"]["not_blank"]


def test_validate_blank_image_fails(tmp_path: Path):
    """Однотонная белая заливка не проходит проверку not_blank."""
    p = tmp_path / "blank.png"
    _make_png(p, color=(255, 255, 255))
    report = validate_image(p, width=512, height=512)
    assert not report["checks"]["not_blank"]
    assert not report["ok"]


def test_validate_wrong_dimensions(tmp_path: Path):
    """Неверное разрешение ловится проверкой dimensions_ok."""
    p = tmp_path / "small.png"
    _make_png(p, size=(256, 256))
    report = validate_image(p, width=512, height=512)
    assert not report["checks"]["dimensions_ok"]
    assert not report["ok"]


def test_validate_missing_file(tmp_path: Path):
    """Отсутствующий файл помечается как невалидный."""
    report = validate_image(tmp_path / "ghost.png")
    assert not report["ok"]
    assert not report["checks"]["exists"]


def test_validate_run_with_manifest(tmp_path: Path):
    """validate_run проверяет все файлы из манифеста."""
    p = tmp_path / "img_000.png"
    _make_png(p)
    m = new_manifest(MODEL_ID, "main", 4, run_id="r2")
    add_item(m, file="img_000.png", prompt="p", seed=116, elapsed_sec=1.0,
             width=512, height=512, size_bytes=p.stat().st_size)
    save_manifest(m, tmp_path)
    report = validate_run(tmp_path)
    assert report["ok"] and report["count"] == 1
