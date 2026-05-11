"""Schémas Pydantic de l'API REST."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    """État de disponibilité de l'API."""

    status: str
    app_name: str
    environment: str
    vector_store_ready: bool


class MetadataResponse(BaseModel):
    """Métadonnées publiques sur la configuration du POC."""

    app_name: str
    environment: str
    source_dataset_url: str
    events_location: str
    events_lookback_days: int
    events_lookahead_days: int
    embedding_model: str
    embedding_provider: str
    chat_model: str
    llm_provider: str
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embedding_model: str
    ollama_min_tokens: int
    ollama_num_ctx: int
    top_k: int
    retrieval_max_score: float | None
    chunk_size: int
    chunk_overlap: int
    vector_store_ready: bool


class AskRequest(BaseModel):
    """Requête utilisateur pour le chatbot."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "Quels concerts jazz à Paris ?",
                    "top_k": 3,
                    "retrieval_max_score": 0.45,
                    "temperature": 0.2,
                    "max_tokens": 600,
                    "llm_provider": "mistral",
                }
            ]
        }
    )

    question: str = Field(..., min_length=1, examples=["Quels concerts jazz à Paris ?"])
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Nombre maximum de sources retournées.",
    )
    retrieval_max_score: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Distance FAISS maximale conservée. Plus bas = plus strict.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.5,
        description="Température du fournisseur LLM.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=100,
        le=2000,
        description="Nombre maximum de tokens générés.",
    )
    llm_provider: Literal["mistral", "ollama", "auto"] | None = Field(
        default=None,
        description=(
            "Fournisseur de génération. 'auto' essaie Ollama puis Mistral."
        ),
    )
    llm_model: str | None = Field(
        default=None,
        description="Modèle de génération optionnel pour le fournisseur choisi.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Refuse les questions vides ou uniquement composées d'espaces."""

        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("La question ne peut pas être vide.")
        return cleaned_value

    @field_validator("llm_model")
    @classmethod
    def validate_llm_model(cls, value: str | None) -> str | None:
        """Nettoie le modèle LLM optionnel envoyé par Swagger ou l'UI."""

        if value is None:
            return None

        cleaned_value = value.strip()
        if not cleaned_value or cleaned_value.lower() == "string":
            return None

        return cleaned_value


class AnswerSourceResponse(BaseModel):
    """Source documentaire utilisée dans une réponse."""

    chunk_id: str
    event_uid: str
    title: str
    city: str
    location_name: str
    start: str
    end: str
    score: float = Field(
        ...,
        description="Distance FAISS du chunk source. Plus bas = plus proche.",
    )


class AskResponse(BaseModel):
    """Réponse augmentée du chatbot."""

    interaction_id: int | None = Field(
        default=None,
        description="Identifiant local de l'interaction, utile pour envoyer un feedback.",
    )
    question: str
    answer: str
    sources: list[AnswerSourceResponse]
    parameters: dict[str, int | float | str | None]


class FeedbackRequest(BaseModel):
    """Feedback utilisateur sur une interaction enregistrée."""

    interaction_id: int = Field(..., ge=1)
    score: Literal["positive", "negative"]
    comment: str | None = Field(default=None, max_length=1000)

    @field_validator("comment")
    @classmethod
    def validate_comment(cls, value: str | None) -> str | None:
        """Nettoie le commentaire optionnel."""

        if value is None:
            return None
        cleaned_value = value.strip()
        return cleaned_value or None


class FeedbackResponse(BaseModel):
    """Réponse après enregistrement d'un feedback."""

    status: str
    interaction_id: int


class RebuildRequest(BaseModel):
    """Paramètres de reconstruction du dataset et de l'index."""

    fetch: bool = Field(
        default=False,
        description="Récupère à nouveau les événements OpenDataSoft avant indexation.",
    )
    city: str | None = Field(default=None, description="Ville cible optionnelle.")
    search: str | None = Field(default=None, description="Recherche texte optionnelle.")
    keywords: list[str] | None = Field(
        default=None,
        description="Mots-clés optionnels. Tous doivent être présents.",
    )
    max_events: int | None = Field(
        default=None,
        ge=1,
        description="Limite optionnelle pour un rebuild de test.",
    )


class RebuildResponse(BaseModel):
    """Résultat de reconstruction de l'index."""

    status: str
    fetched: bool
    dataset_path: str
    vector_store_dir: str
    events_count: int
    chunks_count: int
    future_events_count: int = 0
    past_events_count: int = 0


class ErrorResponse(BaseModel):
    """Format d'erreur standard."""

    detail: str
