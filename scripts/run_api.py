"""Script de lancement local de l'API FastAPI."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402


def main() -> None:
    """Demarre Uvicorn en local."""

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "local",
    )


if __name__ == "__main__":
    main()
