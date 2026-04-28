"""Service de question-réponse RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.rag.answer import AnswerGenerator, build_answer_generator
from app.rag.retriever import EventRetriever
from app.rag.vector_store import SearchResult


@dataclass(frozen=True, slots=True)
class QAParameters:
    """Paramètres configurables pour une question RAG.

    Chaque champ est optionnel : quand une valeur vaut `None`, le service garde
    le paramètre par défaut défini dans `app.config.settings`.
    """

    top_k: int | None = None
    retrieval_max_score: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    llm_provider: str | None = None
    llm_model: str | None = None


@dataclass(frozen=True, slots=True)
class AnswerSource:
    """Source utilisée pour générer une réponse.

    `score` est une distance FAISS : plus elle est basse, plus la source est
    proche de la question.
    """

    chunk_id: str
    event_uid: str
    title: str
    city: str
    location_name: str
    start: str
    end: str
    score: float


@dataclass(frozen=True, slots=True)
class QAResponse:
    """Réponse complète du chatbot."""

    question: str
    answer: str
    sources: list[AnswerSource]
    parameters: QAParameters = field(default_factory=QAParameters)

    def to_dict(self) -> dict[str, Any]:
        """Convertit la réponse en dictionnaire sérialisable."""

        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [asdict(source) for source in self.sources],
            "parameters": asdict(self.parameters),
        }


class QAService:
    """Orchestration explicite du retrieval et de la génération."""

    def __init__(
        self,
        retriever: EventRetriever | None = None,
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        """Charge les dépendances RAG, ou utilise celles injectées par les tests."""

        self.retriever = retriever or EventRetriever.from_local()
        self.answer_generator = answer_generator

    def ask(
        self,
        question: str,
        parameters: QAParameters | None = None,
    ) -> QAResponse:
        """Pose une question au chatbot RAG."""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("La question ne peut pas être vide.")

        runtime_parameters = parameters or QAParameters()
        contexts = self.retriever.search(
            cleaned_question,
            top_k=runtime_parameters.top_k,
            max_score=runtime_parameters.retrieval_max_score,
        )
        answer_generator = self.answer_generator or build_answer_generator(
            provider=runtime_parameters.llm_provider,
            model=runtime_parameters.llm_model,
        )
        answer = answer_generator.generate(
            cleaned_question,
            contexts,
            temperature=runtime_parameters.temperature,
            max_tokens=runtime_parameters.max_tokens,
        )

        return QAResponse(
            question=cleaned_question,
            answer=answer,
            sources=build_sources(contexts),
            parameters=runtime_parameters,
        )


def build_sources(contexts: list[SearchResult]) -> list[AnswerSource]:
    """Transforme les résultats de retrieval en sources exposées."""

    sources: list[AnswerSource] = []
    for result in contexts:
        metadata = result.chunk.metadata
        sources.append(
            AnswerSource(
                chunk_id=result.chunk.id,
                event_uid=str(metadata.get("event_uid", "")),
                title=str(metadata.get("title", "")),
                city=str(metadata.get("city", "")),
                location_name=str(metadata.get("location_name", "")),
                start=str(metadata.get("start", "")),
                end=str(metadata.get("end", "")),
                score=float(result.score),
            )
        )
    return sources
