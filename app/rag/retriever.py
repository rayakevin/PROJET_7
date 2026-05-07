"""Composant de retrieval sur l'index FAISS."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.rag.embeddings import EmbeddingModel, build_embedding_model
from app.rag.temporal import TemporalBucket, classify_question_bucket
from app.rag.vector_store import FaissVectorStore, SearchResult


class EventRetriever:
    """Recherche les chunks d'événements les plus utiles pour une question."""

    def __init__(
        self,
        future_store: FaissVectorStore,
        past_store: FaissVectorStore,
        top_k: int = settings.top_k,
        max_score: float | None = settings.retrieval_max_score,
    ) -> None:
        """Conserve les deux index FAISS et les paramètres de recherche."""

        self.future_store = future_store
        self.past_store = past_store
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
        """Recharge les index localement.

        Les nouveaux artefacts sont stockés dans deux sous-dossiers :
        `future/` pour les événements à venir et `past/` pour les événements
        terminés. Si ces sous-dossiers n'existent pas encore, on garde une
        compatibilité avec l'ancien index unique.
        """

        model = embedding_model or build_embedding_model()
        base_dir = Path(vector_store_dir)
        future_dir = base_dir / "future"
        past_dir = base_dir / "past"

        if future_dir.exists() and past_dir.exists():
            future_store = FaissVectorStore.load(model, future_dir)
            past_store = FaissVectorStore.load(model, past_dir)
        else:
            single_store = FaissVectorStore.load(model, base_dir)
            future_store = single_store
            past_store = single_store

        return cls(
            future_store=future_store,
            past_store=past_store,
            top_k=top_k,
            max_score=max_score,
        )

    def retrieve(self, question: str) -> list[SearchResult]:
        """Retourne les meilleurs chunks pour une question."""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("La question ne peut pas être vide.")

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
    ) -> list[SearchResult]:
        """Recherche avec des paramètres optionnels de retrieval."""

        bucket = classify_question_bucket(question)
        vector_store = self._select_store(bucket)
        return vector_store.search(
            question,
            top_k=top_k or self.top_k,
            max_score=self.max_score if max_score is None else max_score,
        )

    def _select_store(self, bucket: TemporalBucket) -> FaissVectorStore:
        """Sélectionne l'index FAISS adapté à l'intention temporelle."""

        if bucket == "past":
            return self.past_store
        return self.future_store
