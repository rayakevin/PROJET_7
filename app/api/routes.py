"""Routes FastAPI exposant le systeme RAG."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.api.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    RebuildRequest,
    RebuildResponse,
)
from app.config import settings
from app.ingestion.build_dataset import build_dataset
from app.ingestion.fetch_events import fetch_events
from app.services.qa_service import QAParameters, QAService
from app.services.rebuild_service import rebuild_vector_index


router = APIRouter()
_qa_service_cache: QAService | None = None


def get_qa_service() -> QAService:
    """Retourne un service QA reutilise entre les appels."""

    global _qa_service_cache
    if _qa_service_cache is None:
        _qa_service_cache = QAService()
    return _qa_service_cache


def reset_qa_service_cache() -> None:
    """Force le rechargement du service QA apres rebuild."""

    global _qa_service_cache
    _qa_service_cache = None


def vector_store_ready() -> bool:
    """Indique si les artefacts d'index sont presents localement."""

    vector_store_dir = Path(settings.vector_store_dir)
    return all(
        (vector_store_dir / filename).exists()
        for filename in ["index.faiss", "index.pkl", "chunks.json"]
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Verifier la disponibilite de l'API",
)
def health() -> HealthResponse:
    """Retourne l'etat minimal de l'API et de l'index local."""

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.app_env,
        vector_store_ready=vector_store_ready(),
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Poser une question au chatbot RAG",
)
def ask(
    request: AskRequest,
    service: QAService = Depends(get_qa_service),
) -> AskResponse:
    """Genere une reponse augmentee a partir de l'index FAISS."""

    try:
        response = service.ask(
            request.question,
            parameters=QAParameters(
                top_k=request.top_k,
                retrieval_max_score=request.retrieval_max_score,
                retrieval_candidate_multiplier=request.retrieval_candidate_multiplier,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index vectoriel absent. Lancez /rebuild ou scripts/rebuild_index.py --index.",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return AskResponse(**response.to_dict())


@router.post(
    "/rebuild",
    response_model=RebuildResponse,
    responses={403: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Reconstruire le dataset et l'index FAISS",
)
def rebuild(
    request: RebuildRequest,
    x_rebuild_token: str | None = Header(default=None),
) -> RebuildResponse:
    """Reconstruit l'index vectoriel a la demande.

    Si `API_REBUILD_TOKEN` est defini, l'appel doit fournir le meme token dans
    l'en-tete `X-Rebuild-Token`.
    """

    if settings.api_rebuild_token and x_rebuild_token != settings.api_rebuild_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token de rebuild invalide.",
        )

    try:
        raw_events_path = None
        if request.fetch:
            raw_events_path = fetch_events(
                city=request.city,
                search=request.search,
                keywords=request.keywords,
            )

        dataset_path = build_dataset(raw_events_path=raw_events_path)
        result = rebuild_vector_index(
            dataset_path=dataset_path,
            max_events=request.max_events,
        )
        reset_qa_service_cache()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return RebuildResponse(
        status="ok",
        fetched=request.fetch,
        dataset_path=result.dataset_path,
        vector_store_dir=result.vector_store_dir,
        events_count=result.events_count,
        chunks_count=result.chunks_count,
    )
