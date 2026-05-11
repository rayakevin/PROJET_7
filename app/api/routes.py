"""Routes FastAPI exposant le système RAG."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.api.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetadataResponse,
    RebuildRequest,
    RebuildResponse,
)
from app.config import settings
from app.ingestion.build_dataset import build_dataset
from app.ingestion.fetch_events import fetch_events
from app.services.interaction_log import log_interaction, save_feedback
from app.services.qa_service import QAParameters, QAService
from app.services.rebuild_service import rebuild_vector_index


router = APIRouter()
_qa_service_cache: QAService | None = None
logger = logging.getLogger(__name__)


def get_qa_service() -> QAService:
    """Retourne un service QA réutilisé entre les appels."""

    global _qa_service_cache
    if _qa_service_cache is None:
        try:
            _qa_service_cache = QAService()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Index vectoriel absent. Lancez /rebuild ou "
                    "scripts/rebuild_index.py --index."
                ),
            ) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
    return _qa_service_cache


def reset_qa_service_cache() -> None:
    """Force le rechargement du service QA après rebuild."""

    global _qa_service_cache
    _qa_service_cache = None


def vector_store_ready() -> bool:
    """Indique si les artefacts d'index sont présents localement."""

    vector_store_dir = Path(settings.vector_store_dir)
    required_files = ["index.faiss", "index.pkl", "chunks.json"]
    return all(
        (vector_store_dir / bucket / filename).exists()
        for bucket in ["future", "past"]
        for filename in required_files
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérifier la disponibilité de l'API",
)
def health() -> HealthResponse:
    """Retourne l'état minimal de l'API et de l'index local."""

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.app_env,
        vector_store_ready=vector_store_ready(),
    )


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Consulter la configuration publique du POC",
)
def metadata() -> MetadataResponse:
    """Expose les informations utiles aux équipes métier et produit.

    La clé API Mistral et le token de rebuild ne sont jamais retournés.
    """

    return MetadataResponse(
        app_name=settings.app_name,
        environment=settings.app_env,
        source_dataset_url=settings.opendatasoft_records_url,
        events_location=settings.events_location,
        events_lookback_days=settings.events_lookback_days,
        events_lookahead_days=settings.events_lookahead_days,
        embedding_model=settings.mistral_embedding_model,
        embedding_provider=settings.embedding_provider,
        chat_model=settings.mistral_chat_model,
        llm_provider=settings.llm_provider,
        ollama_base_url=settings.ollama_base_url,
        ollama_chat_model=settings.ollama_chat_model,
        ollama_embedding_model=settings.ollama_embedding_model,
        ollama_min_tokens=settings.ollama_min_tokens,
        ollama_num_ctx=settings.ollama_num_ctx,
        top_k=settings.top_k,
        retrieval_max_score=settings.retrieval_max_score,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
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
    """Génère une réponse augmentée à partir de l'index FAISS."""

    try:
        response = service.ask(
            request.question,
            parameters=QAParameters(
                top_k=request.top_k,
                retrieval_max_score=request.retrieval_max_score,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                llm_provider=request.llm_provider,
                llm_model=request.llm_model,
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

    response_payload = response.to_dict()
    try:
        interaction_id = log_interaction(
            question=response_payload["question"],
            answer=response_payload["answer"],
            sources=response_payload["sources"],
            parameters=response_payload["parameters"],
        )
    except Exception as exc:
        logger.warning("Interaction non journalisée : %s", exc)
        interaction_id = None
    return AskResponse(interaction_id=interaction_id, **response_payload)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Enregistrer un feedback utilisateur",
)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Enregistre un retour simple sur une réponse déjà produite."""

    saved = save_feedback(
        interaction_id=request.interaction_id,
        score=request.score,
        comment=request.comment,
    )
    if not saved:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interaction introuvable.",
        )
    return FeedbackResponse(status="ok", interaction_id=request.interaction_id)


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
    """Reconstruit l'index vectoriel à la demande.

    Si `API_REBUILD_TOKEN` est défini, l'appel doit fournir le même token dans
    l'en-tête `X-Rebuild-Token`.
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
