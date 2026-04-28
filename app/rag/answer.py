"""Génération de réponse augmentée avec Mistral."""

from __future__ import annotations

import re
from typing import Protocol

from langchain_core.prompts import ChatPromptTemplate
from mistralai import Mistral
import requests

from app.config import settings
from app.rag.vector_store import SearchResult


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es l'assistant culturel de Puls-Events. "
            "Réponds uniquement à partir du contexte fourni. "
            "Si le contexte ne suffit pas, dis-le clairement. "
            "Propose des événements concrets avec titre, lieu et date quand ils sont disponibles. "
            "Reste concis, utile et naturel.",
        ),
        (
            "human",
            "Question utilisateur :\n{question}\n\n"
            "Contexte récupéré depuis la base vectorielle :\n{context}\n\n"
            "Réponse :",
        ),
    ]
)


class AnswerGenerator(Protocol):
    """Interface minimale du générateur de réponse."""

    def generate(
        self,
        question: str,
        contexts: list[SearchResult],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Génère une réponse finale."""


class MistralAnswerGenerator:
    """Génère une réponse naturelle avec le modèle de chat Mistral."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Prépare le client Mistral et les paramètres par défaut du LLM."""

        self.api_key = api_key or settings.mistral_api_key
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY doit être renseignée pour le chatbot.")

        self.model = model or settings.mistral_chat_model
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.client = Mistral(api_key=self.api_key)

    def generate(
        self,
        question: str,
        contexts: list[SearchResult],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Génère une réponse augmentée par le contexte récupéré."""

        messages = ANSWER_PROMPT.format_messages(
            question=question,
            context=format_context(contexts),
        )
        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": to_mistral_role(message.type), "content": str(message.content)}
                for message in messages
            ],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        content = response.choices[0].message.content
        return str(content).strip()


class OllamaAnswerGenerator:
    """Génère une réponse naturelle avec un modèle local Ollama."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        """Prépare l'appel au serveur Ollama local."""

        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_chat_model
        self.timeout_seconds = timeout_seconds or settings.ollama_timeout_seconds

    def generate(
        self,
        question: str,
        contexts: list[SearchResult],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Génère une réponse augmentée avec Ollama."""

        messages = ANSWER_PROMPT.format_messages(
            question=question,
            context=format_context(contexts),
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": to_ollama_role(message.type), "content": str(message.content)}
                for message in messages
            ],
            "stream": False,
            "options": {
                "temperature": settings.llm_temperature
                if temperature is None
                else temperature,
                "num_predict": max(
                    settings.llm_max_tokens if max_tokens is None else max_tokens,
                    settings.ollama_min_tokens,
                ),
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Échec de génération avec Ollama. "
                "Vérifier que le serveur Ollama est lancé et que le modèle est installé."
            ) from exc

        content = str(response.json()["message"].get("content", "")).strip()
        if not content:
            raise RuntimeError(
                "Ollama n'a pas produit de réponse finale. "
                "Augmenter LLM_MAX_TOKENS ou choisir un modèle non reasoning."
            )
        return remove_thinking_block(content)


class FallbackAnswerGenerator:
    """Générateur qui bascule vers Ollama si Mistral échoue."""

    def __init__(
        self,
        primary: AnswerGenerator,
        fallback: AnswerGenerator,
    ) -> None:
        """Stocke le générateur principal et le générateur de secours."""

        self.primary = primary
        self.fallback = fallback

    def generate(
        self,
        question: str,
        contexts: list[SearchResult],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Essaie le générateur principal, puis le fallback local."""

        try:
            return self.primary.generate(
                question,
                contexts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            return self.fallback.generate(
                question,
                contexts,
                temperature=temperature,
                max_tokens=max_tokens,
            )


def build_answer_generator(
    provider: str | None = None,
    model: str | None = None,
) -> AnswerGenerator:
    """Construit le générateur de réponse configuré."""

    selected_provider = (provider or settings.llm_provider).lower()
    if selected_provider == "mistral":
        return MistralAnswerGenerator(model=model)
    if selected_provider == "ollama":
        return OllamaAnswerGenerator(model=model)
    if selected_provider == "auto":
        fallback = OllamaAnswerGenerator()
        if settings.mistral_api_key:
            return FallbackAnswerGenerator(
                primary=MistralAnswerGenerator(model=model),
                fallback=fallback,
            )
        return fallback
    raise ValueError(
        "LLM_PROVIDER doit valoir 'mistral', 'ollama' ou 'auto'. "
        f"Valeur reçue : {selected_provider}."
    )


def format_context(contexts: list[SearchResult]) -> str:
    """Formate les chunks récupérés pour le prompt."""

    if not contexts:
        return "Aucun contexte pertinent retrouvé."

    parts: list[str] = []
    for index, result in enumerate(contexts, start=1):
        metadata = result.chunk.metadata
        parts.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"Titre : {metadata.get('title', '')}",
                    f"Lieu : {metadata.get('location_name', '')}",
                    f"Ville : {metadata.get('city', '')}",
                    f"Début : {metadata.get('start', '')}",
                    f"Fin : {metadata.get('end', '')}",
                    f"Distance FAISS : {result.score:.4f}",
                    f"Texte : {result.chunk.text}",
                ]
            )
        )

    return "\n\n".join(parts)


def to_mistral_role(langchain_role: str) -> str:
    """Convertit les rôles LangChain vers les rôles attendus par Mistral."""

    return "user" if langchain_role == "human" else langchain_role


def to_ollama_role(langchain_role: str) -> str:
    """Convertit les rôles LangChain vers les rôles attendus par Ollama."""

    return "user" if langchain_role == "human" else langchain_role


def remove_thinking_block(content: str) -> str:
    """Retire un bloc de raisonnement éventuel produit par certains modèles."""

    cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return cleaned_content.strip()
