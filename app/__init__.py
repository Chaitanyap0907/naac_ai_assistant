"""
app/__init__.py
NAAC Accreditation AI Assistant – FastAPI application factory.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.core import get_settings, logger
from app.routers import chat_router, ingest_router

_settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(
        title=_settings.app_title,
        version=_settings.app_version,
        description=(
            "AI-powered NAAC Accreditation Assistant using IBM Granite (watsonx.ai) "
            "and Retrieval-Augmented Generation (RAG) backed by ChromaDB."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ────────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static files & templates ────────────────────────────────────────────────
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # ── Routers ─────────────────────────────────────────────────────────────────
    app.include_router(chat_router)
    app.include_router(ingest_router)

    # ── Health & root ────────────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "service": _settings.app_title, "version": _settings.app_version}

    @app.get("/", tags=["UI"], include_in_schema=False)
    async def root():
        from fastapi.responses import FileResponse
        return FileResponse("templates/index.html")

    logger.info(f"{_settings.app_title} v{_settings.app_version} ready")
    return app
