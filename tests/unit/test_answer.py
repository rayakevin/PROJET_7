"""Tests unitaires de generation de prompt RAG."""

from app.rag.answer import format_context, to_mistral_role
from app.rag.chunking import TextChunk
from app.rag.vector_store import SearchResult


def test_format_context_includes_event_metadata() -> None:
    """Verifie que le contexte injecte au LLM contient les informations metier."""

    result = SearchResult(
        chunk=TextChunk(
            id="evt-001::chunk-0",
            text="Titre : Concert jazz\nDescription : Concert gratuit.",
            metadata={
                "title": "Concert jazz",
                "location_name": "Parc central",
                "city": "Paris",
                "start": "2026-05-02T19:30:00Z",
                "end": "2026-05-02T22:00:00Z",
            },
        ),
        score=0.42,
    )

    context = format_context([result])

    assert "Concert jazz" in context
    assert "Parc central" in context
    assert "2026-05-02T19:30:00Z" in context
    assert "Score : 0.4200" in context


def test_to_mistral_role_converts_human_to_user() -> None:
    """Verifie la compatibilite des roles LangChain avec Mistral."""

    assert to_mistral_role("human") == "user"
    assert to_mistral_role("system") == "system"
