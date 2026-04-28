"""Service de question-reponse RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.runnables import RunnableLambda

from app.rag.answer import AnswerGenerator, MistralAnswerGenerator
from app.rag.retriever import EventRetriever
from app.rag.vector_store import SearchResult


@dataclass(frozen=True, slots=True)
class QAParameters:
    """Parametres configurables pour une question RAG."""

    top_k: int | None = None
    retrieval_max_score: float | None = None
    retrieval_candidate_multiplier: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class AnswerSource:
    """Source utilisee pour generer une reponse."""

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
    """Reponse complete du chatbot."""

    question: str
    answer: str
    sources: list[AnswerSource]
    parameters: QAParameters = field(default_factory=QAParameters)

    def to_dict(self) -> dict[str, Any]:
        """Convertit la reponse en dictionnaire serialisable."""

        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [asdict(source) for source in self.sources],
            "parameters": asdict(self.parameters),
        }


class QAService:
    """Orchestration LangChain du retrieval et de la generation."""

    def __init__(
        self,
        retriever: EventRetriever | None = None,
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self.retriever = retriever or EventRetriever.from_local()
        self.answer_generator = answer_generator or MistralAnswerGenerator()
        self.chain = RunnableLambda(self._retrieve_step) | RunnableLambda(
            self._generation_step
        )

    def ask(
        self,
        question: str,
        parameters: QAParameters | None = None,
    ) -> QAResponse:
        """Pose une question au chatbot RAG."""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("La question ne peut pas etre vide.")

        return self.chain.invoke(
            {
                "question": cleaned_question,
                "parameters": parameters or QAParameters(),
            }
        )

    def _retrieve_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload["question"])
        parameters = payload["parameters"]
        contexts = self.retriever.search(
            question,
            top_k=parameters.top_k,
            max_score=parameters.retrieval_max_score,
            candidate_multiplier=parameters.retrieval_candidate_multiplier,
        )
        return {
            "question": question,
            "contexts": contexts,
            "parameters": parameters,
        }

    def _generation_step(self, payload: dict[str, Any]) -> QAResponse:
        question = str(payload["question"])
        contexts = payload["contexts"]
        parameters = payload["parameters"]
        answer = self.answer_generator.generate(
            question,
            contexts,
            temperature=parameters.temperature,
            max_tokens=parameters.max_tokens,
        )

        return QAResponse(
            question=question,
            answer=answer,
            sources=build_sources(contexts),
            parameters=parameters,
        )


def build_sources(contexts: list[SearchResult]) -> list[AnswerSource]:
    """Transforme les resultats de retrieval en sources exposees."""

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
