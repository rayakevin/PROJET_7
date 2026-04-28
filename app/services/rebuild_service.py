"""Service de reconstruction de l'index vectoriel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import settings
from app.ingestion.build_dataset import DEFAULT_PROCESSED_EVENTS_FILENAME
from app.rag.chunking import chunk_events
from app.rag.embeddings import EmbeddingModel, MistralEmbeddingModel
from app.rag.vector_store import build_vector_store
from app.utils.io import read_json


@dataclass(frozen=True, slots=True)
class RebuildIndexResult:
    """Resume de reconstruction de l'index."""

    dataset_path: str
    vector_store_dir: str
    events_count: int
    chunks_count: int

    def to_dict(self) -> dict[str, str | int]:
        """Convertit le resultat en dictionnaire serialisable."""

        return asdict(self)


def rebuild_vector_index(
    dataset_path: str | Path | None = None,
    vector_store_dir: str | Path | None = None,
    embedding_model: EmbeddingModel | None = None,
    max_events: int | None = None,
) -> RebuildIndexResult:
    """Construit l'index FAISS depuis le dataset normalise."""

    source_path = Path(
        dataset_path
        or settings.processed_data_dir / DEFAULT_PROCESSED_EVENTS_FILENAME
    )
    target_dir = Path(vector_store_dir or settings.vector_store_dir)
    events = read_json(source_path)
    if max_events is not None:
        events = events[:max_events]

    chunks = chunk_events(events)
    model = embedding_model or MistralEmbeddingModel()
    build_vector_store(chunks=chunks, embedding_model=model, output_dir=target_dir)

    return RebuildIndexResult(
        dataset_path=str(source_path),
        vector_store_dir=str(target_dir),
        events_count=len(events),
        chunks_count=len(chunks),
    )
