"""Gestion des embeddings Mistral ou Ollama."""

from __future__ import annotations

import time
from typing import Protocol

import requests
from mistralai import Mistral

from app.config import settings


class EmbeddingModel(Protocol):
    """Interface minimale attendue par le vector store."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectorise une liste de documents."""

    def embed_query(self, text: str) -> list[float]:
        """Vectorise une requête utilisateur."""


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
        """Prépare le client Mistral et les paramètres d'appel par lots."""

        self.api_key = api_key or settings.mistral_api_key
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY doit être renseignée pour les embeddings.")

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
        """Vectorise un lot de textes avec retries simples en cas de rate limit."""

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
                        "Échec de génération des embeddings Mistral. "
                        "Vérifier MISTRAL_API_KEY, MISTRAL_EMBEDDING_MODEL "
                        "et les limites de débit du compte."
                    ) from exc

                sleep_seconds = self.retry_sleep_seconds * (attempt + 1)
                # Attente linéaire simple : suffisante pour un POC et lisible.
                time.sleep(sleep_seconds)

        raise RuntimeError("Échec inattendu de génération des embeddings Mistral.")

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Indique si l'erreur ressemble à une limite de débit Mistral."""

        message = str(exc).lower()
        return "429" in message or "rate limit" in message or "rate_limited" in message


class OllamaEmbeddingModel:
    """Client d'embeddings local via Ollama."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        """Prépare l'accès au serveur Ollama local."""

        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.timeout_seconds = timeout_seconds or settings.ollama_timeout_seconds

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectorise une liste de textes avec Ollama."""

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Vectorise une question utilisateur avec Ollama."""

        return self.embed_documents([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Appelle l'endpoint Ollama d'embeddings par lots."""

        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            return [list(embedding) for embedding in payload["embeddings"]]
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise RuntimeError(
                    "Échec de génération des embeddings Ollama. "
                    "Vérifier OLLAMA_BASE_URL et OLLAMA_EMBEDDING_MODEL."
                ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                "Échec de génération des embeddings Ollama. "
                "Vérifier OLLAMA_BASE_URL et OLLAMA_EMBEDDING_MODEL."
            ) from exc

        # Compatibilité avec les anciennes versions d'Ollama qui n'exposent que
        # /api/embeddings, limité à un texte par appel.
        embeddings: list[list[float]] = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                embeddings.append(list(response.json()["embedding"]))
            except requests.RequestException as exc:
                raise RuntimeError(
                    "Échec de génération des embeddings Ollama. "
                    "Vérifier OLLAMA_BASE_URL et OLLAMA_EMBEDDING_MODEL."
                ) from exc
        return embeddings


def build_embedding_model(provider: str | None = None) -> EmbeddingModel:
    """Construit le modèle d'embeddings configuré."""

    selected_provider = (provider or settings.embedding_provider).lower()
    if selected_provider == "mistral":
        return MistralEmbeddingModel()
    if selected_provider == "ollama":
        return OllamaEmbeddingModel()
    raise ValueError(
        "EMBEDDING_PROVIDER doit valoir 'mistral' ou 'ollama'. "
        f"Valeur reçue : {selected_provider}."
    )
