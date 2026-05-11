"""Service de reconstruction de l'index vectoriel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import settings
from app.ingestion.build_dataset import DEFAULT_PROCESSED_EVENTS_FILENAME
from app.rag.chunking import chunk_events
from app.rag.embeddings import EmbeddingModel, build_embedding_model
from app.rag.temporal import event_bucket
from app.rag.vector_store import build_vector_store
from app.utils.io import read_json


@dataclass(frozen=True, slots=True)
class RebuildIndexResult:
    """Résumé de reconstruction de l'index."""

    dataset_path: str
    vector_store_dir: str
    events_count: int
    chunks_count: int
    future_events_count: int = 0
    past_events_count: int = 0

    def to_dict(self) -> dict[str, str | int]:
        """Convertit le résultat en dictionnaire sérialisable."""

        return asdict(self)


def rebuild_vector_index(
    dataset_path: str | Path | None = None,
    vector_store_dir: str | Path | None = None,
    embedding_model: EmbeddingModel | None = None,
    max_events: int | None = None,
) -> RebuildIndexResult:
    """Construit deux index FAISS : événements futurs et événements passés."""

    source_path = Path(
        dataset_path
        or settings.processed_data_dir / DEFAULT_PROCESSED_EVENTS_FILENAME
    )
    target_dir = Path(vector_store_dir or settings.vector_store_dir)
    events = read_json(source_path)
    if max_events is not None:
        events = events[:max_events]

    future_events = [
        event for event in events if event_bucket(event) == "future"
    ]
    past_events = [
        event for event in events if event_bucket(event) == "past"
    ]
    model = embedding_model or build_embedding_model()
    future_chunks = chunk_events(future_events)
    past_chunks = chunk_events(past_events)

    build_vector_store(
        chunks=future_chunks,
        embedding_model=model,
        output_dir=target_dir / "future",
    )
    build_vector_store(
        chunks=past_chunks,
        embedding_model=model,
        output_dir=target_dir / "past",
    )

    return RebuildIndexResult(
        dataset_path=str(source_path),
        vector_store_dir=str(target_dir),
        events_count=len(events),
        chunks_count=len(future_chunks) + len(past_chunks),
        future_events_count=len(future_events),
        past_events_count=len(past_events),
    )
