"""Gestion des embeddings Mistral."""

from __future__ import annotations

import time
from typing import Protocol

from mistralai import Mistral

from app.config import settings


class EmbeddingModel(Protocol):
    """Interface minimale attendue par le vector store."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectorise une liste de documents."""

    def embed_query(self, text: str) -> list[float]:
        """Vectorise une requete utilisateur."""


class MistralEmbeddingModel:
    """Client d'embeddings Mistral."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        batch_delay_seconds: float | None = None,
        max_retries: int | None = None,
        retry_sleep_seconds: float | None = None,
    ) -> None:
        self.api_key = api_key or settings.mistral_api_key
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY doit etre renseignee pour les embeddings.")

        self.model = model or settings.mistral_embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.batch_delay_seconds = (
            settings.embedding_batch_delay_seconds
            if batch_delay_seconds is None
            else batch_delay_seconds
        )
        self.max_retries = max_retries or settings.embedding_max_retries
        self.retry_sleep_seconds = (
            settings.embedding_retry_sleep_seconds
            if retry_sleep_seconds is None
            else retry_sleep_seconds
        )
        self.client = Mistral(api_key=self.api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectorise une liste de textes par lots."""

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(self._embed_batch(batch))
            if self.batch_delay_seconds > 0 and start + self.batch_size < len(texts):
                time.sleep(self.batch_delay_seconds)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Vectorise une question utilisateur."""

        return self.embed_documents([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    inputs=texts,
                )
                return [list(item.embedding) for item in response.data]
            except Exception as exc:
                if not self._is_rate_limit_error(exc) or attempt >= self.max_retries:
                    raise RuntimeError(
                        "Echec de generation des embeddings Mistral. "
                        "Verifier MISTRAL_API_KEY, MISTRAL_EMBEDDING_MODEL "
                        "et les limites de debit du compte."
                    ) from exc

                sleep_seconds = self.retry_sleep_seconds * (attempt + 1)
                time.sleep(sleep_seconds)

        raise RuntimeError("Echec inattendu de generation des embeddings Mistral.")

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "429" in message or "rate limit" in message or "rate_limited" in message
