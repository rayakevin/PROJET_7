"""Tests unitaires du service de question-reponse."""

import pytest

from app.rag.chunking import TextChunk
from app.rag.vector_store import SearchResult
from app.services.qa_service import QAService


class FakeRetriever:
    """Retriever de test sans index FAISS."""

    def retrieve(self, question: str) -> list[SearchResult]:
        return [
            SearchResult(
                chunk=TextChunk(
                    id="evt-001::chunk-0",
                    text=f"Contexte pour {question}",
                    metadata={
                        "event_uid": "evt-001",
                        "title": "Concert jazz",
                        "city": "Paris",
                        "location_name": "Parc central",
                        "start": "2026-05-02T19:30:00Z",
                        "end": "2026-05-02T22:00:00Z",
                    },
                ),
                score=0.12,
            )
        ]


class FakeAnswerGenerator:
    """Generateur de reponse de test sans appel Mistral."""

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        return f"Reponse a '{question}' avec {len(contexts)} source."


def test_qa_service_returns_answer_and_sources() -> None:
    """Verifie l'orchestration retrieval puis generation."""

    service = QAService(
        retriever=FakeRetriever(),
        answer_generator=FakeAnswerGenerator(),
    )

    response = service.ask("Quels concerts jazz a Paris ?")

    assert response.answer == "Reponse a 'Quels concerts jazz a Paris ?' avec 1 source."
    assert response.sources[0].event_uid == "evt-001"
    assert response.sources[0].title == "Concert jazz"
    assert response.to_dict()["sources"][0]["city"] == "Paris"


def test_qa_service_rejects_empty_question() -> None:
    """Verifie la gestion des questions vides."""

    service = QAService(
        retriever=FakeRetriever(),
        answer_generator=FakeAnswerGenerator(),
    )

    with pytest.raises(ValueError):
        service.ask("   ")
