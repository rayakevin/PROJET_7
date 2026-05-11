"""Adaptateur LangChain pour le modèle de chat Mistral."""

from __future__ import annotations

import time
from typing import Any

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from mistralai import Mistral

from app.config import settings


class MistralChatModel(BaseChatModel):
    """Expose Mistral comme un ChatModel LangChain reutilisable."""

    api_key: str
    model: str = settings.mistral_chat_model
    temperature: float = settings.llm_temperature
    max_tokens: int = settings.llm_max_tokens
    max_retries: int = settings.embedding_max_retries
    retry_sleep_seconds: float = settings.embedding_retry_sleep_seconds

    @property
    def _llm_type(self) -> str:
        """Identifiant technique du modèle."""

        return "mistral-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Paramètres utiles pour le tracing LangChain."""

        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Appelle l'API chat Mistral et retourne un résultat LangChain."""

        del stop, run_manager, kwargs
        client = Mistral(api_key=self.api_key)
        mistral_messages = [
            {"role": to_mistral_role(message), "content": str(message.content)}
            for message in messages
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.complete(
                    model=self.model,
                    messages=mistral_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = str(response.choices[0].message.content).strip()
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )
            except Exception:
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_sleep_seconds)

        raise RuntimeError("Échec inattendu lors de l'appel au modèle Mistral.")


class OllamaChatModel(BaseChatModel):
    """Expose Ollama comme un ChatModel LangChain réutilisable."""

    base_url: str = settings.ollama_base_url
    model: str = settings.ollama_chat_model
    temperature: float = settings.llm_temperature
    max_tokens: int = settings.llm_max_tokens
    timeout_seconds: int = settings.ollama_timeout_seconds
    num_ctx: int = settings.ollama_num_ctx

    @property
    def _llm_type(self) -> str:
        """Identifiant technique du modèle."""

        return "ollama-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Paramètres utiles pour le tracing LangChain."""

        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Appelle l'API chat Ollama et retourne un résultat LangChain."""

        del stop, run_manager, kwargs
        payload = {
            "model": self.model,
            "messages": [
                {"role": to_ollama_role(message), "content": str(message.content)}
                for message in messages
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        response = requests.post(
            f"{self.base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        content = remove_thinking_block(
            str(response.json()["message"].get("content", "")).strip()
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )


def to_mistral_role(message: BaseMessage) -> str:
    """Convertit un message LangChain vers un role Mistral."""

    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    return "user"


def to_ollama_role(message: BaseMessage) -> str:
    """Convertit un message LangChain vers un rôle Ollama."""

    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    return "user"


def remove_thinking_block(content: str) -> str:
    """Retire un éventuel bloc de raisonnement produit par certains modèles."""

    if "</think>" not in content:
        return content.strip()
    return content.split("</think>", maxsplit=1)[1].strip()
