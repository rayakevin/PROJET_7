"""Point d'entrée applicatif FastAPI."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings


def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI."""

    app = FastAPI(
        title=settings.app_name,
        description="API REST du POC RAG culturel Puls-Events.",
        version="0.1.0",
    )
    app.include_router(router)
    return app


app = create_app()
