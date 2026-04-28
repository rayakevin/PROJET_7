"""Generation de reponse augmentee avec Mistral."""

from __future__ import annotations

from typing import Protocol

from langchain_core.prompts import ChatPromptTemplate
from mistralai import Mistral

from app.config import settings
from app.rag.vector_store import SearchResult


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es l'assistant culturel de Puls-Events. "
            "Reponds uniquement a partir du contexte fourni. "
            "Si le contexte ne suffit pas, dis-le clairement. "
            "Propose des evenements concrets avec titre, lieu et date quand ils sont disponibles. "
            "Reste concis, utile et naturel.",
        ),
        (
            "human",
            "Question utilisateur :\n{question}\n\n"
            "Contexte recupere depuis la base vectorielle :\n{context}\n\n"
            "Reponse :",
        ),
    ]
)


class AnswerGenerator(Protocol):
    """Interface minimale du generateur de reponse."""

    def generate(
        self,
        question: str,
        contexts: list[SearchResult],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Genere une reponse finale."""


class MistralAnswerGenerator:
    """Genere une reponse naturelle avec le modele de chat Mistral."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key or settings.mistral_api_key
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY doit etre renseignee pour le chatbot.")

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
        """Genere une reponse augmentee par le contexte retrieve."""

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


def format_context(contexts: list[SearchResult]) -> str:
    """Formate les chunks recuperes pour le prompt."""

    if not contexts:
        return "Aucun contexte pertinent retrouve."

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
                    f"Debut : {metadata.get('start', '')}",
                    f"Fin : {metadata.get('end', '')}",
                    f"Score : {result.score:.4f}",
                    f"Texte : {result.chunk.text}",
                ]
            )
        )

    return "\n\n".join(parts)


def to_mistral_role(langchain_role: str) -> str:
    """Convertit les roles LangChain vers les roles attendus par Mistral."""

    return "user" if langchain_role == "human" else langchain_role
