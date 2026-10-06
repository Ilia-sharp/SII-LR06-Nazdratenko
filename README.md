# SII-LR06 · Конвейер диффузии · Вариант 16

**Лабораторная работа №6 «Генерация изображений: конвейер диффузии»**
Дисциплина: Системы искусственного интеллекта.
Автор: **Nazdratenko**.

> Открытая модель **stabilityai/sd-turbo** генерирует изображения **рабочего
> места** (тема варианта `office`, тип промпта `interior`) при фиксированном
> **seed 116**. Всё воспроизводимо, все параметры и время зафиксированы в
> `outputs/manifest.json`. Плюс строго одностраничный сайт с живым прогрессом
> генерации.

## Что по варианту 16

| Параметр | Значение |
|---|---|
| topic_id | `office` |
| Тема | «рабочее место» |
| Тип промпта | `interior` |
| Модель (открытая) | `stabilityai/sd-turbo` (4 шага, CFG 0.0) |
| Лицензия | Stability AI Community License — см. [MODEL_CARD.md](MODEL_CARD.md) |
| Seed | **116** (у всех изображений прогона) |
| Разрешение | 512 × 512 |
| Промптов | 5 (файл `prompts.txt`, 3–5 по заданию) |

## Структура проекта

```
SII-LR06-Nazdratenko/
├── prompts.txt                 # 5 промптов (по одному на строку)
├── src/
│   ├── config.py               # вариант, модель, seed, шаги
│   ├── prompts.py              # чтение prompts.txt
│   ├── model.py                # загрузка SD-Turbo, генерация (ленивый torch)
│   ├── generate.py             # CLI генерации  ← запуск по заданию
│   ├── manifest.py             # manifest.json (атомарная запись)
│   ├── validate.py             # проверки изображений
│   ├── checks.py               # python -m src.checks
│   └── server.py               # FastAPI + SSE (веб-интерфейс)
├── web/
│   └── index.html              # СТРОГО одностраничный сайт (стили и JS внутри)
├── outputs/                    # результаты прогона: img_000…004.png + manifest.json
├── tests/test_pipeline.py      # 18 unit-тестов (без torch)
├── scripts/download_model.py   # предзагрузка весов
├── MODEL_CARD.md               # карточка модели + лицензия
├── ARCHITECTURE.md             # архитектура конвейера (со схемой)
├── render.yaml                 # Render Blueprint (тариф Free)
├── requirements*.txt           # полные / веб / dev-зависимости
└── .github/workflows/ci.yml    # CI: pytest + проверки
```

## Быстрый старт (локально)

```bash
# 1. Окружение
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Зависимости (torch CPU + diffusers)
pip install -r requirements.txt

# 3. Прогон строго как в задании
python -m src.generate --prompts prompts.txt --out outputs/

# 4. Проверки результатов
python -m src.checks
```

После прогона в `outputs/` появятся `img_000.png … img_004.png` и
`manifest.json`. Манифест дозаписывается после каждой генерации, поэтому
прервать прогон безопасно.

### Железо и время генерации (наш фактический прогон)

Железо: виртуальная машина, **CPU 2 ядра Intel Xeon**, без GPU, 4 ГБ RAM.
Точность bf16 + attention slicing + VAE slicing; текстовый энкодер
выгружается из памяти после кодирования промптов (пик RAM ≈ 2.5 ГБ).

Фактический прогон (run_id `20261006-1641`, версия весов
`b261bac6fd2c…`):

| Файл | Промпт (сцена) | Seed | Время | Размер |
|---|---|---|---|---|
| img_000.png | уютное домашнее рабочее место, утренний свет | 116 | 18.28 с | 425 КБ |
| img_001.png | минимализм, белый стол, панорамное окно | 116 | 17.83 с | 394 КБ |
| img_002.png | рабочее место в общежитии, вечерний свет | 116 | 17.96 с | 481 КБ |
| img_003.png | лофт-студия, кирпичная стена, индустриальный свет | 116 | 17.55 с | 525 КБ |
| img_004.png | светлый угол у окна, растения, кофе | 116 | 17.50 с | 390 КБ |

**Итого:** чистый инференс ≈ 89.1 с за 5 изображений (≈ **17.8 с** на
картинку на 2 ядрах CPU); весь прогон wall-clock ≈ 107 с (с загрузкой
модели и кодированием промптов). Обязательно к запуску:
`python -m src.generate --prompts prompts.txt --out outputs/`.

## Одностраничный сайт (веб-интерфейс)

Сайт — **один файл** `web/index.html`: стили и скрипты встроены, все разделы
на одной странице (без вкладок и переходов): параметры прогона → конвейер с
живым прогрессом → галерея → манифест → документация.

```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
# открыть http://localhost:8000
```

Живой прогресс: `POST /api/generate` отдаёт поток Server-Sent Events —
страница показывает стадии (CLIP → диффузия ×4 → VAE → PNG), превью из
промежуточных латентов, таймер и журнал. Три режима:

- **Повтор прогона (replay)** — воспроизведение зафиксированного прогона
  из manifest.json (работает даже без torch);
- **Локально (real)** — настоящий инференс SD-Turbo на сервере;
- **Облако (cloud)** — демо через pollinations.ai.

API: `/api/health`, `/api/info`, `/api/prompts`, `/api/manifest`,
`/api/demo-images`, `/api/live-images`, `/api/docs/{model_card|architecture|readme}`,
`POST /api/generate` (SSE). Swagger: `/docs`.

## Тесты и CI

```bash
pip install -r requirements-dev.txt
pytest -q          # 18 тестов: конфиг, промпты, манифест, валидация
```

CI (`.github/workflows/ci.yml`) ставит только лёгкие зависимости (без torch),
прогоняет тесты, `python -m src.checks` и импорт сервера.

## Деплой на Render (Blueprint, тариф Free)

1. Залить репозиторий на GitHub (см. раздел «Сдача»).
2. На <https://dashboard.render.com> → **New + → Blueprint** → выбрать
   репозиторий → **Apply**. Render прочитает `render.yaml`.
3. Тариф **Free** ($0/мес, 0.1 CPU, 512 MB RAM): сервер стартует на
   `requirements-web.txt` **без torch**; сайт работает в режимах
   replay + cloud, health-check — `/api/health`.
4. Первый запрос после сна может занимать до минуты — сайт сам покажет
   оверлей «Сервер просыпается» и повторит попытки.

## Сдача работы

1. **GitHub.** Создать публичный репозиторий `SII-LR06-Nazdratenko` и
   запушить код (веса не коммитятся — это учтено в `.gitignore`;
   `outputs/` с картинками и манифестом коммитится).
2. **Render.** Развернуть Blueprint (см. выше) — получить живой URL сайта.
3. **GitVerse.** Дублировать репозиторий:
   ```bash
   git remote add gitverse <URL_репозитория_GitVerse>
   git push -u gitverse main
   git tag v1.0 && git push gitverse v1.0
   git push -u origin main && git push origin v1.0
   ```
4. Тег релиза — **v1.0**.

## Лицензии

- Код проекта — MIT (см. `LICENSE`).
- Веса модели — Stability AI Community License (в репозиторий не входят,
  скачиваются с Hugging Face Hub).
