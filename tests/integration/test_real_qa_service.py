"""Test d'integration du chatbot RAG avec Mistral."""

from pathlib import Path

import pytest

from app.config import settings
from app.services.qa_service import QAService


def test_real_qa_service_answers_with_sources() -> None:
    """Verifie une reponse RAG complete sur l'index local."""

    index_path = Path(settings.vector_store_dir) / "index.faiss"
    chunks_path = Path(settings.vector_store_dir) / "chunks.json"
    pkl_path = Path(settings.vector_store_dir) / "index.pkl"

    if not settings.mistral_api_key:
        pytest.skip("MISTRAL_API_KEY non renseignee.")
    if not index_path.exists() or not chunks_path.exists() or not pkl_path.exists():
        pytest.skip("Index FAISS reel absent. Lancer scripts/rebuild_index.py --index.")

    response = QAService().ask("Quels concerts de jazz sont disponibles a Paris ?")

    assert response.answer
    assert response.sources
    assert any("jazz" in source.title.lower() for source in response.sources)
