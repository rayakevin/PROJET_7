"""Tests d'intégration du vector store avec vrais embeddings locaux."""

from pathlib import Path

import pytest
import requests

from app.config import settings
from app.rag.embeddings import build_embedding_model
from app.rag.vector_store import FaissVectorStore


def test_real_vector_store_search_with_local_embeddings() -> None:
    """Vérifie une recherche sur l'index FAISS réel."""

    future_dir = Path(settings.vector_store_dir) / "future"
    index_path = future_dir / "index.faiss"
    chunks_path = future_dir / "chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        pytest.skip("Index FAISS réel absent. Lancer scripts/rebuild_index.py --index.")
    try:
        requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2).raise_for_status()
    except requests.RequestException:
        pytest.skip("Ollama non disponible localement.")

    vector_store = FaissVectorStore.load(build_embedding_model(), future_dir)
    results = vector_store.search(
        "Quels concerts de jazz sont disponibles à Paris ?",
        top_k=3,
    )

    assert len(vector_store.chunks) > 1000
    assert results
    assert all(result.score >= 0 for result in results)
