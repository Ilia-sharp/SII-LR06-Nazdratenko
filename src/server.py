"""FastAPI-сервер: одностраничный веб-интерфейс + REST API + SSE-генерация.

Запуск:
    uvicorn src.server:app --host 0.0.0.0 --port 8000

Сайт строго одностраничный: весь интерфейс — один файл web/index.html
(стили и скрипты встроены), все данные подгружаются через /api/*.

Три режима генерации (POST /api/generate, ответ — поток Server-Sent
Events):

* ``replay`` — воспроизведение сохранённого прогона из outputs/manifest.json;
* ``real``   — локальный инференс SD-Turbo (нужны torch + веса), с живым
  прогрессом по шагам и превью из латентов;
* ``cloud``  — демо через публичный генератор pollinations.ai (когда
  локальный инференс невозможен, например на бесплатном Render).

Режим ``auto`` выбирает первый подходящий: replay → real → cloud.
"""
from __future__ import annotations

import base64
import io
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from src.config import (
    GUIDANCE_SCALE,
    HEIGHT,
    MODEL_ID,
    MODEL_LICENSE,
    PROMPT_TYPE,
    SEED,
    STEPS,
    THEME,
    VARIANT,
    WIDTH,
)
from src.manifest import add_item, load_manifest, new_manifest, save_manifest
from src.model import real_available, torch_available
from src.prompts import read_prompts

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
OUT_DIR = Path(os.environ.get("OUTPUTS_DIR", ROOT / "outputs"))
LIVE_DIR = Path(os.environ.get("LIVE_DIR", ROOT / "data" / "live"))
PROMPTS_FILE = Path(os.environ.get("PROMPTS_FILE", ROOT / "prompts.txt"))

ENABLE_REAL_GEN = os.environ.get("ENABLE_REAL_GEN", "1") == "1"
CLOUD_TIMEOUT = int(os.environ.get("CLOUD_TIMEOUT", "90"))
CLOUD_RETRIES = int(os.environ.get("CLOUD_RETRIES", "2"))
CLOUD_MODEL = os.environ.get("CLOUD_MODEL", "flux")
CLOUD_SIZE = int(os.environ.get("CLOUD_SIZE", "768"))
CLOUD_URL = (
    "https://image.pollinations.ai/prompt/{q}"
    "?width={w}&height={h}&seed={s}&model={m}&nologo=true"
)

# Основная живая модель — FLUX.1-schnell на публичном Hugging Face Space.
# Он отдаёт 1024×1024 без водяного знака; Pollinations остаётся резервом.
HF_SPACE = os.environ.get("HF_SPACE", "black-forest-labs-flux-1-schnell")
HF_SIZE = int(os.environ.get("HF_SIZE", "1024"))
HF_STEPS = int(os.environ.get("HF_STEPS", "4"))
HF_START_TIMEOUT = int(os.environ.get("HF_START_TIMEOUT", "45"))
HF_WAIT_TIMEOUT = int(os.environ.get("HF_WAIT_TIMEOUT", "300"))
HF_API = f"https://{HF_SPACE}.hf.space/gradio_api"
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

DOC_FILES: dict[str, Path] = {
    "model_card": ROOT / "MODEL_CARD.md",
    "architecture": ROOT / "ARCHITECTURE.md",
    "readme": ROOT / "README.md",
}

SAFE_FILE = re.compile(r"^[A-Za-z0-9._-]+$")

app = FastAPI(
    title="SII-LR06 · Конвейер диффузии (вариант 16)",
    version="1.0.0",
    description="Учебный проект: генерация изображений моделью sd-turbo, "
                "облачное демо — Flux",
)


# ────────────────────────── Вспомогательные функции ──────────────────────────

def _sse(data: dict) -> str:
    """Кодирует одно SSE-событие."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _safe_join(base: Path, name: str) -> Path:
    """Путь внутри ``base`` без выхода за его пределы."""
    if not SAFE_FILE.match(name):
        raise HTTPException(status_code=400, detail="недопустимое имя файла")
    path = (base / name).resolve()
    if not str(path).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="недопустимый путь")
    return path


def _manifest_or_none() -> dict | None:
    path = OUT_DIR / "manifest.json"
    if not path.exists():
        return None
    try:
        return load_manifest(path)
    except Exception:
        return None


HISTORY_FILE = LIVE_DIR / "history.json"
HISTORY_LIMIT = 40
_history_lock = threading.Lock()


def _load_history() -> list[dict]:
    """Читает историю живых генераций; битый файл не роняет сайт."""
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _record_history(record: dict) -> None:
    """Дописывает результат генерации в data/live/history.json (свежие сверху)."""
    with _history_lock:
        items = [i for i in _load_history() if i.get("file") != record.get("file")]
        items.insert(0, record)
        items = items[:HISTORY_LIMIT]
        try:
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = HISTORY_FILE.with_name("history.json.tmp")
            tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(HISTORY_FILE)
        except Exception:  # noqa: BLE001 — история не должна ломать генерацию
            pass


def _pick_mode(prompt: str, requested: str) -> str | None:
    """Выбирает фактический режим генерации."""
    manifest = _manifest_or_none()
    has_match = bool(manifest and any(
        item.get("prompt") == prompt for item in manifest.get("items", [])
    ))

    real_ok = ENABLE_REAL_GEN and real_available()

    if requested == "replay":
        return "replay" if manifest else None
    if requested == "real":
        return "real" if real_ok else None
    if requested == "cloud":
        return "cloud"
    # auto
    if has_match:
        return "replay"
    if real_ok:
        return "real"
    return "cloud"


# ────────────────────────────── Веб-интерфейс ────────────────────────────────

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Строго одностраничный сайт (все стили/скрипты встроены в файл)."""
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


# ────────────────────────────────── REST API ─────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    """Проверка живости сервера (используется health-check'ом Render)."""
    torch_ok = torch_available()
    real_ok = real_available()
    return {
        "status": "ok",
        "variant": VARIANT,
        "torch": torch_ok,
        "enable_real_gen": ENABLE_REAL_GEN,
        "modes": {
            "replay": _manifest_or_none() is not None,
            "real": ENABLE_REAL_GEN and real_ok,
            "cloud": True,
        },
    }


@app.get("/api/info")
def info() -> dict:
    """Сводка о варианте, модели и последнем прогоне."""
    manifest = _manifest_or_none()
    items = (manifest or {}).get("items", [])
    elapsed = [it.get("elapsed_sec", 0) for it in items]
    total_bytes = sum(it.get("size_bytes") or 0 for it in items)

    return {
        "variant": VARIANT,
        "theme": THEME,
        "prompt_type": PROMPT_TYPE,
        "model_id": MODEL_ID,
        "model_version": (manifest or {}).get("model_version", "main"),
        "license": MODEL_LICENSE,
        "steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
        "torch": torch_available(),
        "enable_real_gen": ENABLE_REAL_GEN,
        "run": {
            "run_id": (manifest or {}).get("run_id"),
            "count": len(items),
            "total_elapsed_sec": round(sum(elapsed), 2) if elapsed else None,
            "avg_elapsed_sec": round(sum(elapsed) / len(elapsed), 2) if elapsed else None,
            "total_bytes": total_bytes or None,
        },
    }


@app.get("/api/prompts")
def prompts() -> dict:
    """Список промптов из prompts.txt."""
    try:
        return {"prompts": read_prompts(PROMPTS_FILE)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/manifest")
def manifest() -> dict:
    """Содержимое outputs/manifest.json (или 404, если прогона не было)."""
    m = _manifest_or_none()
    if m is None:
        raise HTTPException(status_code=404, detail="manifest.json ещё не создан")
    return m


@app.get("/api/demo-images")
def demo_images() -> dict:
    """Изображения последнего прогона из outputs/ (для галереи)."""
    m = _manifest_or_none()
    items = (m or {}).get("items", [])
    result = []
    for item in items:
        if (OUT_DIR / item["file"]).exists():
            result.append({**item, "url": f"/api/images/{item['file']}"})
    return {"count": len(result), "items": result}


@app.get("/api/live-images")
def live_images() -> dict:
    """Изображения, сгенерированные прямо из веб-интерфейса."""
    result = []
    if LIVE_DIR.exists():
        for p in sorted(LIVE_DIR.iterdir(), key=lambda x: x.stat().st_mtime,
                        reverse=True):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            result.append({
                "file": p.name,
                "url": f"/api/live-images/{p.name}",
                "size_bytes": p.stat().st_size,
            })
    return {"count": len(result), "items": result[:12]}


@app.get("/api/history")
def history(limit: int = 24) -> dict:
    """История картинок: живые генерации (свежие сверху), затем архив."""
    limit = max(1, min(int(limit or 24), HISTORY_LIMIT + 20))
    items: list[dict] = []
    seen: set[str] = set()

    for record in _load_history():
        name = record.get("file")
        if not name or name in seen or not (LIVE_DIR / name).is_file():
            continue
        seen.add(name)
        items.append({
            **record,
            "url": f"/api/live-images/{name}",
            "source": "live",
        })

    # Старые картинки без записи в истории — подхватываем по времени файла.
    if LIVE_DIR.exists():
        for p in sorted(LIVE_DIR.iterdir(), key=lambda x: x.stat().st_mtime,
                        reverse=True):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            if p.name in seen:
                continue
            seen.add(p.name)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime))
            items.append({
                "file": p.name,
                "url": f"/api/live-images/{p.name}",
                "prompt": "",
                "seed": None,
                "model": "",
                "created_at": stamp,
                "size_bytes": p.stat().st_size,
                "source": "live",
            })

    manifest = _manifest_or_none()
    for item in (manifest or {}).get("items", []):
        name = item.get("file")
        if not name or not (OUT_DIR / name).is_file():
            continue
        items.append({
            **item,
            "url": f"/api/images/{name}",
            "source": "archive",
        })

    return {"count": len(items[:limit]), "items": items[:limit]}


@app.get("/api/images/{name}")
def image_file(name: str) -> FileResponse:
    """Отдаёт PNG из outputs/."""
    path = _safe_join(OUT_DIR, name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="файл не найден")
    return FileResponse(path, media_type="image/png")


@app.get("/api/live-images/{name}")
def live_image_file(name: str) -> FileResponse:
    """Отдаёт PNG, сгенерированный из веб-интерфейса."""
    path = _safe_join(LIVE_DIR, name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="файл не найден")
    media = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(path, media_type=media)


@app.get("/api/docs/{key}")
def docs(key: str) -> Response:
    """Markdown-документ (model_card | architecture | readme)."""
    path = DOC_FILES.get(key)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="документ не найден")
    return Response(path.read_text(encoding="utf-8"), media_type="text/markdown")


# ─────────────────────────────── SSE-генерация ───────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = ""
    seed: int = SEED
    mode: str = "auto"  # auto | replay | real | cloud


def _replay_stream(req: GenerateRequest):
    """Воспроизведение сохранённого прогона (без вычислений)."""
    manifest = _manifest_or_none()
    items = (manifest or {}).get("items", [])
    if req.prompt:
        items = [it for it in items if it.get("prompt") == req.prompt] or items[:1]

    yield _sse({
        "type": "meta",
        "mode": "replay",
        "run_id": (manifest or {}).get("run_id"),
        "model_id": (manifest or {}).get("model_id", MODEL_ID),
        "steps": (manifest or {}).get("steps", STEPS),
        "note": "Воспроизведение сохранённого прогона "
                f"run_id {(manifest or {}).get('run_id')}",
    })

    for it in items:
        seed_i = it.get("seed", SEED)
        steps = int((manifest or {}).get("steps", STEPS))
        yield _sse({"type": "stage", "stage": "encode",
                    "message": f"Кодирование промпта (CLIP) · seed {seed_i}"})
        time.sleep(0.6)
        for step in range(1, steps + 1):
            yield _sse({"type": "step", "step": step, "steps": steps,
                        "message": f"Шаг диффузии {step}/{steps}"})
            time.sleep(0.5)
        yield _sse({"type": "stage", "stage": "vae",
                    "message": "Декодирование латентов (VAE)"})
        time.sleep(0.5)
        yield _sse({
            "type": "result",
            "mode": "replay",
            "file": it.get("file"),
            "url": f"/api/images/{it.get('file')}",
            "prompt": it.get("prompt"),
            "seed": seed_i,
            "elapsed_sec": it.get("elapsed_sec"),
            "width": it.get("width", WIDTH),
            "height": it.get("height", HEIGHT),
            "size_bytes": it.get("size_bytes"),
            "note": "в оригинальном прогоне",
        })

    yield _sse({"type": "end", "count": len(items),
                "note": "Это запись настоящего прогона — для новой картинки "
                        "выберите режим «Локально» или «Облако»."})


def _real_stream(req: GenerateRequest):
    """Локальный инференс SD-Turbo в отдельном потоке + очередь событий."""
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            from src.model import decode_latents_preview, load_pipeline

            events.put({"type": "meta", "mode": "real", "model_id": MODEL_ID,
                        "steps": STEPS,
                        "note": "Локальный инференс sd-turbo (CPU)"})

            events.put({"type": "stage", "stage": "load",
                        "message": "Загрузка весов модели…"})
            pipe = load_pipeline()

            steps = STEPS
            state = {"last_preview": None}

            def on_step(pipe_, step, timestep, kwargs):
                latents = kwargs["latents"]
                events.put({"type": "step", "step": step + 1, "steps": steps,
                            "message": f"Шаг диффузии {step + 1}/{steps}"})
                try:
                    preview = decode_latents_preview(pipe_, latents)
                    buf = io.BytesIO()
                    preview.save(buf, format="JPEG", quality=72)
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    state["last_preview"] = b64
                    events.put({"type": "preview",
                                "data_url": f"data:image/jpeg;base64,{b64}",
                                "step": step + 1})
                except Exception:  # noqa: BLE001 — превью не критично
                    pass
                return kwargs

            events.put({"type": "stage", "stage": "encode",
                        "message": "Кодирование промпта (CLIP)…"})

            from src.model import generate_one
            image, elapsed = generate_one(
                req.prompt,
                seed=req.seed,
                steps=steps,
                guidance_scale=GUIDANCE_SCALE,
                width=WIDTH,
                height=HEIGHT,
                callback_on_step=on_step,
            )

            events.put({"type": "stage", "stage": "save",
                        "message": "Сохранение PNG…"})
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            fname = time.strftime("live_%Y%m%d-%H%M%S.png")
            path = LIVE_DIR / fname
            image.save(path, format="PNG")

            # Записываем живую генерацию в отдельный манифест data/live.
            live_manifest_path = LIVE_DIR / "manifest.json"
            try:
                lm = load_manifest(live_manifest_path)
            except Exception:
                lm = new_manifest(MODEL_ID, "main", steps,
                                  run_id=time.strftime("%Y%m%d-%H%M"))
            add_item(lm, file=fname, prompt=req.prompt, seed=req.seed,
                     elapsed_sec=elapsed, width=WIDTH, height=HEIGHT,
                     size_bytes=path.stat().st_size)
            save_manifest(lm, LIVE_DIR)

            events.put({
                "type": "result", "mode": "real", "file": fname,
                "url": f"/api/live-images/{fname}",
                "prompt": req.prompt, "seed": req.seed,
                "elapsed_sec": round(elapsed, 2),
                "width": WIDTH, "height": HEIGHT,
                "size_bytes": path.stat().st_size,
            })
            _record_history({
                "file": fname, "prompt": req.prompt, "seed": req.seed,
                "model": MODEL_ID, "mode": "real",
                "elapsed_sec": round(elapsed, 2),
                "width": WIDTH, "height": HEIGHT,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            events.put({"type": "end", "count": 1})
        except Exception as exc:  # noqa: BLE001
            events.put({"type": "error",
                        "message": f"Ошибка локальной генерации: {exc}"})
            events.put({"type": "end", "count": 0})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        try:
            event = events.get(timeout=0.2)
        except queue.Empty:
            yield ": keepalive\n\n"
            continue
        yield _sse(event)
        if event.get("type") == "end":
            break


def _to_jpeg(raw: bytes, upscale_to: int = 0, quality: int = 92,
             sharpen: bool = False, crop_bottom: int = 0) -> bytes:
    """Переводит ответ модели (webp/png) в качественный JPEG.

    ``crop_bottom`` срезает полосу с водяным знаком у резервного канала,
    ``sharpen`` добавляет лёгкий unsharp mask — убирает «мыло».
    ``upscale_to`` — мягкий Lanczos-апскейл, если нужно.
    """
    from PIL import Image, ImageFilter

    with Image.open(io.BytesIO(raw)) as img:
        rgb = img.convert("RGB")
        if crop_bottom and rgb.height > crop_bottom + 32:
            w, h = rgb.size
            rgb = rgb.crop((0, 0, w, h - crop_bottom)).resize((w, h), Image.LANCZOS)
            sharpen = True
        if upscale_to and rgb.width < upscale_to:
            rgb = rgb.resize((upscale_to, upscale_to), Image.LANCZOS)
            sharpen = True
        if sharpen:
            rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=3))
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()


def _hf_inference_image(prompt: str, seed: int, size: int, steps: int) -> bytes:
    """FLUX.1-schnell через Hugging Face Inference Providers (нужен HF_TOKEN).

    Это основной путь, когда токен задан: без ZeroGPU-очереди и с квотой
    аккаунта Hugging Face.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN не задан")
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=HF_TOKEN, timeout=HF_WAIT_TIMEOUT)
    image = client.text_to_image(
        prompt,
        model=HF_SPACE,
        width=size,
        height=size,
        num_inference_steps=steps,
        seed=int(seed) % 2147483647,
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _hf_flux_image(prompt: str, seed: int, size: int, steps: int) -> bytes:
    """FLUX.1-schnell через публичный Hugging Face Space (1024 px, без логотипа).

    Gradio-API: сначала POST создаёт задачу, затем SSE-поток присылает
    готовый файл. Таймауты разделены: короткий на постановку задачи и
    длинный на ожидание GPU-очереди.
    """
    payload = {
        "data": [
            prompt,
            int(seed) % 2147483647,
            True,
            int(size),
            int(size),
            int(steps),
        ]
    }
    request = urllib.request.Request(
        f"{HF_API}/call/infer",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "sii-lr06/1.3 (+https://github.com/Ilia-sharp/SII-LR06-Nazdratenko)",
            **({"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=HF_START_TIMEOUT) as resp:
        started = json.loads(resp.read().decode("utf-8"))

    event_id = started.get("event_id")
    if not event_id:
        raise RuntimeError("FLUX Space не вернул event_id")

    file_url = None
    saw_error = False
    with urllib.request.urlopen(
        f"{HF_API}/call/infer/{event_id}", timeout=HF_WAIT_TIMEOUT
    ) as stream:
        for raw_line in stream:
            line = raw_line.decode("utf-8", "replace").strip()
            if line == "event: error":
                saw_error = True
                continue
            if not line.startswith("data:"):
                continue
            try:
                chunk = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            for item in chunk if isinstance(chunk, list) else [chunk]:
                if isinstance(item, dict) and item.get("url"):
                    file_url = item["url"]

    if not file_url:
        raise RuntimeError(
            "бесплатная квота FLUX исчерпана" if saw_error
            else "FLUX Space не вернул изображение"
        )
    with urllib.request.urlopen(file_url, timeout=HF_START_TIMEOUT) as resp:
        raw = resp.read()
    if len(raw) < 5000:
        raise RuntimeError("FLUX Space вернул пустой файл")
    return raw


def _retry_after_seconds(value: str | None, default: int = 45) -> int:
    """Сколько секунд ждать до следующего бесплатного слота pollinations."""
    try:
        seconds = int(str(value or "").strip())
    except ValueError:
        return default
    return max(1, min(seconds, 90))


def _cloud_stream(req: GenerateRequest):
    """Живая генерация: FLUX.1-schnell на HF Space, резерв — pollinations.ai."""
    yield _sse({"type": "meta", "mode": "cloud",
                "model_id": "FLUX.1-schnell",
                "steps": HF_STEPS, "width": HF_SIZE, "height": HF_SIZE,
                "note": "Живая генерация FLUX.1-schnell"})
    yield _sse({"type": "stage", "stage": "cloud",
                "message": f"FLUX.1-schnell · {HF_SIZE} px…"})

    t0 = time.perf_counter()
    data = b""
    label = "FLUX.1-schnell"
    width = height = HF_SIZE

    if HF_TOKEN:
        try:
            raw = _hf_inference_image(req.prompt, req.seed, HF_SIZE, HF_STEPS)
            data = _to_jpeg(raw)
        except Exception as exc:  # noqa: BLE001 — пробуем Space
            yield _sse({"type": "stage", "stage": "cloud",
                        "message": f"HF Inference недоступен — пробуем FLUX Space ({exc})…"})

    if not data:
        try:
            raw = _hf_flux_image(req.prompt, req.seed, HF_SIZE, HF_STEPS)
            data = _to_jpeg(raw)
        except Exception as exc:  # noqa: BLE001 — переходим на резервный канал
            yield _sse({"type": "stage", "stage": "cloud",
                        "message": f"FLUX Space занят — резервный канал ({exc})…"})

    if not data:
        q = urllib.parse.quote(req.prompt, safe="")
        url = CLOUD_URL.format(q=q, w=CLOUD_SIZE, h=CLOUD_SIZE,
                               s=req.seed, m=CLOUD_MODEL)
        data, ctype, last_error = b"", "", "нет ответа"
        for attempt in range(1, CLOUD_RETRIES + 2):
            wait = 0
            try:
                request = urllib.request.Request(url, headers={
                    "User-Agent": "sii-lr06/1.3 (+https://github.com/Ilia-sharp/SII-LR06-Nazdratenko)",
                    "Accept": "image/*",
                })
                with urllib.request.urlopen(request, timeout=CLOUD_TIMEOUT) as resp:
                    ctype = (resp.headers.get("Content-Type") or "").lower()
                    data = resp.read()
                if ctype.startswith("image/") and len(data) > 5000:
                    break
                last_error = f"неожиданный ответ ({ctype or 'без типа'}, {len(data)} байт)"
                data = b""
            except urllib.error.HTTPError as exc:
                data = b""
                if exc.code == 402:
                    wait = _retry_after_seconds(exc.headers.get("Retry-After"))
                    last_error = "бесплатный лимит API (одна картинка в 45 секунд)"
                elif exc.code == 429:
                    wait = _retry_after_seconds(exc.headers.get("Retry-After"), 30)
                    last_error = "слишком много запросов к API"
                else:
                    last_error = f"HTTP {exc.code} {exc.reason}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                data = b""

            if attempt <= CLOUD_RETRIES:
                if wait:
                    yield _sse({"type": "stage", "stage": "cloud",
                                "message": f"Свободный слот API через {wait} с — ждём…"})
                    time.sleep(wait + 1)
                else:
                    yield _sse({"type": "stage", "stage": "cloud",
                                "message": f"Повтор запроса {attempt} из {CLOUD_RETRIES}…"})
                    time.sleep(2)

        if not data:
            yield _sse({"type": "error",
                        "message": f"Облачный генератор не ответил: {last_error}"})
            yield _sse({"type": "end", "count": 0})
            return

        label = "Flux · резерв"
        width = height = CLOUD_SIZE
        ext = "jpg"
        data = _to_jpeg(data, sharpen=True, crop_bottom=34)
    else:
        ext = "jpg"

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"cloud_{time.strftime('%Y%m%d-%H%M%S')}.{ext}"
    (LIVE_DIR / fname).write_bytes(data)

    elapsed = round(time.perf_counter() - t0, 2)
    _record_history({
        "file": fname, "prompt": req.prompt, "seed": req.seed,
        "model": label, "mode": "cloud",
        "elapsed_sec": elapsed,
        "width": width, "height": height,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    yield _sse({
        "type": "result", "mode": "cloud", "model": label, "file": fname,
        "url": f"/api/live-images/{fname}",
        "prompt": req.prompt, "seed": req.seed,
        "elapsed_sec": elapsed,
        "width": width, "height": height,
        "size_bytes": len(data),
    })
    yield _sse({"type": "end", "count": 1})


@app.post("/api/generate")
def generate(req: GenerateRequest) -> StreamingResponse:
    """SSE-поток генерации: события meta/stage/step/preview/result/error/end."""
    req.prompt = (req.prompt or "").strip()
    if not req.prompt:
        raise HTTPException(status_code=400, detail="пустой промпт")
    if len(req.prompt) > 1500:
        raise HTTPException(status_code=400, detail="промпт слишком длинный")

    mode = _pick_mode(req.prompt, req.mode)
    if mode is None:
        raise HTTPException(
            status_code=409,
            detail="Режим недоступен: нет сохранённого прогона для replay, "
                   "либо torch/веса не установлены для real",
        )

    streams = {"replay": _replay_stream, "real": _real_stream, "cloud": _cloud_stream}
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(streams[mode](req),
                             media_type="text/event-stream",
                             headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
