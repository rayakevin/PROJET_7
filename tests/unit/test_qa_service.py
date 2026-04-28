"""Tests unitaires du service de question-reponse."""

import pytest

from app.rag.chunking import TextChunk
from app.rag.vector_store import SearchResult
from app.services.qa_service import QAParameters, QAService


class FakeRetriever:
    """Retriever de test sans index FAISS."""

    def __init__(self) -> None:
        self.last_options = {}

    def retrieve(self, question: str) -> list[SearchResult]:
        return self.search(question)

    def search(
        self,
        question: str,
        top_k: int | None = None,
        max_score: float | None = None,
        candidate_multiplier: int | None = None,
    ) -> list[SearchResult]:
        self.last_options = {
            "top_k": top_k,
            "max_score": max_score,
            "candidate_multiplier": candidate_multiplier,
        }
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

    def __init__(self) -> None:
        self.last_options = {}

    def generate(
        self,
        question: str,
        contexts: list[SearchResult],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.last_options = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
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
    assert response.to_dict()["parameters"]["top_k"] is None


def test_qa_service_applies_runtime_parameters() -> None:
    """Verifie la transmission des hyperparametres a la chaine RAG."""

    retriever = FakeRetriever()
    answer_generator = FakeAnswerGenerator()
    service = QAService(
        retriever=retriever,
        answer_generator=answer_generator,
    )

    response = service.ask(
        "Quels concerts jazz a Paris ?",
        parameters=QAParameters(
            top_k=4,
            retrieval_max_score=0.5,
            retrieval_candidate_multiplier=6,
            temperature=0.4,
            max_tokens=500,
        ),
    )

    assert retriever.last_options == {
        "top_k": 4,
        "max_score": 0.5,
        "candidate_multiplier": 6,
    }
    assert answer_generator.last_options == {
        "temperature": 0.4,
        "max_tokens": 500,
    }
    assert response.parameters.top_k == 4


def test_qa_service_rejects_empty_question() -> None:
    """Verifie la gestion des questions vides."""

    service = QAService(
        retriever=FakeRetriever(),
        answer_generator=FakeAnswerGenerator(),
    )

    with pytest.raises(ValueError):
        service.ask("   ")
