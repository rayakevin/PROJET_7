"""Tests d'integration du vector store avec vrais embeddings Mistral."""

from pathlib import Path

import pytest

from app.config import settings
from app.rag.embeddings import MistralEmbeddingModel
from app.rag.vector_store import FaissVectorStore


def test_real_vector_store_search_with_mistral_embeddings() -> None:
    """Verifie une recherche sur l'index FAISS reel."""

    index_path = Path(settings.vector_store_dir) / "index.faiss"
    chunks_path = Path(settings.vector_store_dir) / "chunks.json"

    if not settings.mistral_api_key:
        pytest.skip("MISTRAL_API_KEY non renseignee.")
    if not index_path.exists() or not chunks_path.exists():
        pytest.skip("Index FAISS reel absent. Lancer scripts/rebuild_index.py --index.")

    vector_store = FaissVectorStore.load(settings.vector_store_dir, MistralEmbeddingModel())
    results = vector_store.search(
        "Quels concerts de jazz sont disponibles a Paris ?",
        top_k=3,
    )

    assert len(vector_store.chunks) > 1000
    assert results
    assert all(result.score >= 0 for result in results)
