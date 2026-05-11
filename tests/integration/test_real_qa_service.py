"""Test d'intégration du chatbot RAG avec Ollama local."""

from pathlib import Path

import pytest
import requests

from app.config import settings
from app.services.qa_service import QAService


def test_real_qa_service_answers_with_sources() -> None:
    """Vérifie une réponse RAG complète sur l'index local."""

    future_dir = Path(settings.vector_store_dir) / "future"
    index_path = future_dir / "index.faiss"
    chunks_path = future_dir / "chunks.json"
    pkl_path = future_dir / "index.pkl"

    if not index_path.exists() or not chunks_path.exists() or not pkl_path.exists():
        pytest.skip("Index FAISS réel absent. Lancer scripts/rebuild_index.py --index.")
    try:
        requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2).raise_for_status()
    except requests.RequestException:
        pytest.skip("Ollama non disponible localement.")

    response = QAService().ask("Je cherche un concert de jazz à Paris à venir")

    assert response.answer
    assert response.sources
    assert "jazz" in response.answer.lower()
