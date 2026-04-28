"""Composant de retrieval sur l'index FAISS."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.rag.embeddings import EmbeddingModel, MistralEmbeddingModel
from app.rag.vector_store import FaissVectorStore, SearchResult


class EventRetriever:
    """Recherche les chunks d'evenements les plus utiles pour une question."""

    def __init__(
        self,
        vector_store: FaissVectorStore,
        top_k: int = settings.top_k,
        max_score: float | None = settings.retrieval_max_score,
    ) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.max_score = max_score

    @classmethod
    def from_local(
        cls,
        vector_store_dir: str | Path = settings.vector_store_dir,
        embedding_model: EmbeddingModel | None = None,
        top_k: int = settings.top_k,
        max_score: float | None = settings.retrieval_max_score,
    ) -> "EventRetriever":
        """Recharge le retriever depuis l'index local."""

        model = embedding_model or MistralEmbeddingModel()
        vector_store = FaissVectorStore.load(vector_store_dir, model)
        return cls(vector_store=vector_store, top_k=top_k, max_score=max_score)

    def retrieve(self, question: str) -> list[SearchResult]:
        """Retourne les meilleurs chunks pour une question."""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("La question ne peut pas etre vide.")

        return self.search(
            cleaned_question,
            top_k=self.top_k,
            max_score=self.max_score,
        )

    def search(
        self,
        question: str,
        top_k: int | None = None,
        max_score: float | None = None,
        candidate_multiplier: int | None = None,
    ) -> list[SearchResult]:
        """Recherche avec des parametres optionnels de retrieval."""

        return self.vector_store.search(
            question,
            top_k=top_k or self.top_k,
            max_score=self.max_score if max_score is None else max_score,
            candidate_multiplier=(
                settings.retrieval_candidate_multiplier
                if candidate_multiplier is None
                else candidate_multiplier
            ),
        )
