"""Adaptateur LangChain pour le modele de chat Mistral."""

from __future__ import annotations

import time
from typing import Any

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
        """Identifiant technique du modele."""

        return "mistral-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Parametres utiles pour le tracing LangChain."""

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
        """Appelle l'API chat Mistral et retourne un resultat LangChain."""

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

        raise RuntimeError("Echec inattendu lors de l'appel au modele Mistral.")


def to_mistral_role(message: BaseMessage) -> str:
    """Convertit un message LangChain vers un role Mistral."""

    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    return "user"
