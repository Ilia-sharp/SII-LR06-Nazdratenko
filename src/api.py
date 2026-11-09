"""Простой HTTP API на FastAPI.

Запуск:
    python -m src.cli serve --artifacts artifacts --port 8000
Документация Swagger: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .artifacts import ModelBundle, load_bundle, recommend_for_items, recommend_for_visitor

WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


class RecommendRequest(BaseModel):
    visitor_id: Optional[int] = Field(
        default=None, description="ID посетителя из events.csv"
    )
    item_ids: Optional[List[int]] = Field(
        default=None, description="Товары текущей сессии"
    )
    n: int = Field(default=10, ge=1, le=50)
    model: str = Field(default="auto", pattern="^(auto|covisitation|popularity)$")
    fallback: bool = Field(
        default=True, description="Добивать ли выдачу популярностью (только auto)"
    )


def create_app(artifacts_dir: str | Path | None = None) -> FastAPI:
    directory = Path(
        artifacts_dir or os.environ.get("ARTIFACTS_DIR", "artifacts")
    )

    app = FastAPI(
        title="Retailrocket Recommender · вариант 16",
        version="1.0",
        description=(
            "Top-N рекомендации товаров по visitorid или товарам текущей "
            "сессии. Модели: popularity baseline и item-item/co-visitation."
        ),
    )

    cache: dict[str, ModelBundle] = {}

    def bundle() -> ModelBundle:
        if "value" not in cache:
            try:
                cache["value"] = load_bundle(directory)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return cache["value"]

    @app.get("/", include_in_schema=False)
    def root():
        if WEB_INDEX.exists():
            return FileResponse(WEB_INDEX, media_type="text/html")
        return {
            "service": "Retailrocket recommender",
            "variant": 16,
            "docs": "/docs",
            "health": "/health",
            "recommend": "/recommend?visitor_id=123&n=10",
        }

    @app.get("/api", include_in_schema=False)
    def api_root():
        return {
            "service": "Retailrocket recommender",
            "variant": 16,
            "docs": "/docs",
            "health": "/health",
            "recommend": "/recommend?visitor_id=123&n=10",
        }

    @app.get("/health")
    @app.get("/api/health", include_in_schema=False)
    def health():
        try:
            loaded = bundle()
        except HTTPException:
            return {"status": "no_model", "artifacts": str(directory)}
        return {
            "status": "ok",
            "artifacts": str(directory),
            "visitors": int(len(loaded.profiles.visitors)),
            "items": int(len(loaded.covisitation.item_ids)),
        }

    @app.get("/info")
    @app.get("/api/info", include_in_schema=False)
    def info():
        loaded = bundle()
        meta = loaded.metadata
        return {
            "lab": meta.get("lab"),
            "variant": meta.get("variant"),
            "config": meta.get("config"),
            "data": meta.get("data"),
            "split": meta.get("split"),
            "sessions": meta.get("sessions"),
            "popularity": meta.get("popularity"),
            "covisitation": meta.get("covisitation"),
            "targets": meta.get("targets"),
            "evaluation": meta.get("evaluation"),
        }

    @app.get("/metrics")
    @app.get("/api/metrics", include_in_schema=False)
    def metrics():
        return bundle().metadata.get("evaluation", {})

    @app.get("/recommend")
    @app.get("/api/recommend", include_in_schema=False)
    def recommend_get(
        visitor_id: Optional[int] = Query(default=None),
        items: Optional[str] = Query(
            default=None, description="itemid через запятую: 1,2,3"
        ),
        n: int = Query(default=10, ge=1, le=50),
        model: str = Query(default="auto", pattern="^(auto|covisitation|popularity)$"),
        fallback: bool = Query(default=True),
    ):
        if (visitor_id is None) == (items is None):
            raise HTTPException(
                status_code=422,
                detail="Укажите ровно одно: visitor_id или items",
            )
        loaded = bundle()
        if visitor_id is not None:
            return recommend_for_visitor(
                loaded, visitor_id, n=n, fallback=fallback, model=model
            )
        try:
            item_ids = [int(x.strip()) for x in items.split(",") if x.strip()]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="items: ожидаются int") from exc
        return recommend_for_items(
            loaded, item_ids=item_ids, n=n, fallback=fallback
        )

    @app.post("/recommend")
    @app.post("/api/recommend", include_in_schema=False)
    def recommend_post(request: RecommendRequest):
        if (request.visitor_id is None) == (request.item_ids is None):
            raise HTTPException(
                status_code=422,
                detail="Укажите ровно одно поле: visitor_id или item_ids",
            )
        loaded = bundle()
        if request.visitor_id is not None:
            return recommend_for_visitor(
                loaded,
                request.visitor_id,
                n=request.n,
                fallback=request.fallback,
                model=request.model,
            )
        return recommend_for_items(
            loaded,
            item_ids=request.item_ids or [],
            n=request.n,
            fallback=request.fallback,
        )

    return app


app = create_app()
