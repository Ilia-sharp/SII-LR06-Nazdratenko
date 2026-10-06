"""Загрузка диффузионного конвейера (SD-Turbo) и генерация изображения.

torch импортируется лениво и только внутри функций — благодаря этому
сервер (src/server.py) может работать вообще без torch, например на
бесплатном тарифе Render, где доступно всего 512 МБ RAM.

Экономия RAM на CPU (важно для машин с 4 ГБ памяти):
  * dtype bfloat16 + attention slicing + VAE slicing;
  * все промпты кодируются текстовым энкодером ЗАРАНЕЕ, после чего
    энкодер выгружается один раз (~0.6 ГБ), а шаги диффузии идут без него;
  * после каждого изображения куча процессов возвращается системе
    (malloc_trim), чтобы память не «нарастала» от прогона к прогону.
"""
from __future__ import annotations

import ctypes
import gc
import threading
from typing import Any, Callable

from src.config import GUIDANCE_SCALE, MODEL_ID, STEPS

_pipe = None
_pipe_lock = threading.Lock()
_pipe_meta: dict[str, Any] = {}


def torch_available() -> bool:
    """True, если torch установлен в окружении."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def get_device() -> str:
    """Выбирает устройство инференса: cuda → mps → cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline():
    """Загружает SD-Turbo (результат кэшируется в глобальной переменной)."""
    global _pipe, _pipe_meta

    with _pipe_lock:
        if _pipe is not None:
            return _pipe

        import torch
        from diffusers import AutoPipelineForText2Image

        device = get_device()
        if device in ("cuda", "mps"):
            dtype = torch.float16
        else:
            dtype = torch.bfloat16

        # Пытаемся загрузить компактные fp16-веса (вдвое меньше трафик);
        # если в репозитории их нет — берём стандартные.
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                MODEL_ID, dtype=dtype, variant="fp16")
        except Exception:
            pipe = AutoPipelineForText2Image.from_pretrained(
                MODEL_ID, dtype=dtype)

        if device == "cpu":
            # Экономия RAM: вычисление внимания и VAE по частям.
            # (в diffusers 0.41+ методы VAE перенесены в объект vae)
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.vae.enable_slicing()
            except Exception:
                try:
                    pipe.enable_vae_slicing()
                except Exception:
                    pass

        _pipe = pipe.to(device)
        _pipe_meta = {"device": device, "dtype": str(dtype)}
        return _pipe


def pipe_meta() -> dict[str, Any]:
    """Информация о загруженном конвейере (может быть пустым)."""
    return dict(_pipe_meta)


def is_cpu() -> bool:
    return _pipe_meta.get("device", "cpu") == "cpu"


# ─────────────────────── Кодирование промптов ───────────────────────────────

def encode_prompts(pipe, prompts: list[str]):
    """Кодирует промпты текстовым энкодером.

    На CPU после кодирования ВСЕХ промптов энкодер выгружается один раз —
    так шаги диффузии идут с запасом RAM ≈ 0.6 ГБ и без повторных
    догрузок (которые фрагментируют память).

    Returns:
        список эмбеддингов; None, если устройство не CPU (энкодер не
        выгружаем, промпт передаётся текстом).
    """
    if not is_cpu():
        return None

    import torch

    if getattr(pipe, "text_encoder", None) is None:
        _reload_text_encoder(pipe)

    embeds = []
    for p in prompts:
        with torch.no_grad():
            e, _ = pipe.encode_prompt(
                p, pipe.device, num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        embeds.append(e)

    _free_text_encoder(pipe)
    return embeds


def _free_text_encoder(pipe) -> None:
    """Выгружает текстовый энкодер и токенизатор из конвейера."""
    pipe.text_encoder = None
    pipe.tokenizer = None
    try:
        pipe._internal_dict.pop("text_encoder", None)
        pipe._internal_dict.pop("tokenizer", None)
    except Exception:
        pass
    gc.collect()
    _trim_heap()


def _reload_text_encoder(pipe) -> None:
    """Лениво возвращает текстовый энкодер после выгрузки (~2–4 с).

    Нужен для повторных генераций в одном процессе (например, в сервере):
    энкодер догружается с диска только при кодировании нового промпта.
    """
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    if getattr(pipe, "tokenizer", None) is None:
        pipe.tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    encoder = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", variant="fp16", dtype=torch.bfloat16,
    )
    pipe.text_encoder = encoder.to(pipe.device)


# ─────────────────────────── Генерация ──────────────────────────────────────

def generate_image(
    pipe,
    *,
    seed: int,
    steps: int = STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    width: int = 512,
    height: int = 512,
    prompt: str | None = None,
    embeds=None,
    callback_on_step: Callable | None = None,
):
    """Генерирует одно изображение из эмбеддингов (или текста на GPU).

    Returns:
        (PIL.Image, elapsed_sec) — изображение и время инференса в секундах.
    """
    import time

    import torch

    device = _pipe_meta.get("device", "cpu")
    generator = torch.Generator(device=device).manual_seed(int(seed))

    kwargs: dict[str, Any] = {
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "generator": generator,
        "width": int(width),
        "height": int(height),
    }
    if embeds is not None:
        kwargs["prompt_embeds"] = embeds
    else:
        kwargs["prompt"] = prompt
    if callback_on_step is not None:
        kwargs["callback_on_step_end"] = callback_on_step
        kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    t0 = time.perf_counter()
    result = pipe(**kwargs)
    elapsed = time.perf_counter() - t0

    image = result.images[0]
    del result
    gc.collect()
    _trim_heap()
    return image, elapsed


def generate_one(
    prompt: str,
    *,
    seed: int,
    steps: int = STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    width: int = 512,
    height: int = 512,
    callback_on_step: Callable | None = None,
):
    """Одноразовая генерация по тексту (для сервера): кодирует и освобождает."""
    pipe = load_pipeline()
    if is_cpu():
        embeds = encode_prompts(pipe, [prompt])[0]
    else:
        embeds = None
    return generate_image(
        pipe,
        seed=seed,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        prompt=prompt,
        embeds=embeds,
        callback_on_step=callback_on_step,
    )


def _trim_heap() -> None:
    """Возвращает освободившуюся память ОС (glibc malloc_trim)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def decode_latents_preview(pipe, latents):
    """Декодирует промежуточные латенты в PIL-превью (для живого прогресса)."""
    import torch

    with torch.no_grad():
        lat = latents / pipe.vae.config.scaling_factor
        sample = pipe.vae.decode(lat).sample
    return pipe.image_processor.postprocess(sample, output_type="pil")[0]
