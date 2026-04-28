"""Schemas Pydantic de l'API REST."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Etat de disponibilite de l'API."""

    status: str
    app_name: str
    environment: str
    vector_store_ready: bool


class AskRequest(BaseModel):
    """Requete utilisateur pour le chatbot."""

    question: str = Field(..., min_length=1, examples=["Quels concerts jazz a Paris ?"])
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Nombre maximum de sources retournees.",
    )
    retrieval_max_score: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Distance FAISS maximale conservee. Plus bas = plus strict.",
    )
    retrieval_candidate_multiplier: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Nombre de candidats FAISS avant reranking.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.5,
        description="Temperature du LLM Mistral.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=100,
        le=2000,
        description="Nombre maximum de tokens generes.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Refuse les questions vides ou uniquement composees d'espaces."""

        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("La question ne peut pas etre vide.")
        return cleaned_value


class AnswerSourceResponse(BaseModel):
    """Source documentaire utilisee dans une reponse."""

    chunk_id: str
    event_uid: str
    title: str
    city: str
    location_name: str
    start: str
    end: str
    score: float


class AskResponse(BaseModel):
    """Reponse augmentee du chatbot."""

    question: str
    answer: str
    sources: list[AnswerSourceResponse]
    parameters: dict[str, int | float | None]


class RebuildRequest(BaseModel):
    """Parametres de reconstruction du dataset et de l'index."""

    fetch: bool = Field(
        default=False,
        description="Recupere a nouveau les evenements OpenDataSoft avant indexation.",
    )
    city: str | None = Field(default=None, description="Ville cible optionnelle.")
    search: str | None = Field(default=None, description="Recherche texte optionnelle.")
    keywords: list[str] | None = Field(
        default=None,
        description="Mots-cles optionnels. Tous doivent etre presents.",
    )
    max_events: int | None = Field(
        default=None,
        ge=1,
        description="Limite optionnelle pour un rebuild de test.",
    )


class RebuildResponse(BaseModel):
    """Resultat de reconstruction de l'index."""

    status: str
    fetched: bool
    dataset_path: str
    vector_store_dir: str
    events_count: int
    chunks_count: int


class ErrorResponse(BaseModel):
    """Format d'erreur standard."""

    detail: str
