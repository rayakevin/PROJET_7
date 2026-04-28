"""Configuration centrale du projet.

Ce module centralise le chargement des variables d'environnement et la
définition des paramètres globaux utilisés dans l'application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _int_env(name: str, default: int) -> int:
    """Convertit une variable d'environnement en entier.

    Parameters
    ----------
    name : str
        Nom de la variable d'environnement à lire.
    default : int
        Valeur de repli si la variable n'est pas définie.

    Returns
    -------
    int
        Valeur convertie en entier si la variable existe,
        sinon la valeur par défaut.
    """
    value = os.getenv(name)
    return int(value) if value is not None else default


def _float_env(name: str, default: float) -> float:
    """Convertit une variable d'environnement en nombre décimal."""

    value = os.getenv(name)
    return float(value) if value is not None else default


def _optional_float_env(name: str, default: float | None = None) -> float | None:
    """Convertit une variable d'environnement optionnelle en nombre décimal."""

    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


@dataclass(slots=True)
class Settings:
    """Regroupe les paramètres globaux du projet.

    Cette classe rassemble les paramètres nécessaires au fonctionnement
    de l'application :
    - configuration générale de l'API ;
    - accès à OpenAgenda ;
    - chemins des répertoires de données ;
    - paramètres techniques du pipeline.

    Les valeurs sont lues depuis les variables d'environnement avec
    des valeurs par défaut adaptées au développement local.
    """

    app_name: str = os.getenv("APP_NAME", "Projet 7 - POC RAG")
    app_env: str = os.getenv("APP_ENV", "local")
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = _int_env("APP_PORT", 8000)
    api_rebuild_token: str = os.getenv("API_REBUILD_TOKEN", "")

    opendatasoft_records_url: str = os.getenv(
        "OPENDATASOFT_RECORDS_URL",
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "evenements-publics-openagenda/records",
    )
    events_location: str = os.getenv("EVENTS_LOCATION", "Paris")
    events_lookback_days: int = _int_env("EVENTS_LOOKBACK_DAYS", 365)
    events_lookahead_days: int = _int_env("EVENTS_LOOKAHEAD_DAYS", 90)
    events_page_size: int = _int_env("EVENTS_PAGE_SIZE", 100)
    request_timeout_seconds: int = _int_env("REQUEST_TIMEOUT_SECONDS", 30)

    data_dir: Path = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    raw_data_dir: Path = Path(os.getenv("RAW_DATA_DIR", BASE_DIR / "data" / "raw"))
    processed_data_dir: Path = Path(
        os.getenv("PROCESSED_DATA_DIR", BASE_DIR / "data" / "processed")
    )
    vector_store_dir: Path = Path(
        os.getenv("VECTOR_STORE_DIR", BASE_DIR / "data" / "vector_store")
    )
    evaluation_data_dir: Path = Path(
        os.getenv("EVALUATION_DATA_DIR", BASE_DIR / "data" / "evaluation")
    )

    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    mistral_embedding_model: str = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
    mistral_chat_model: str = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")
    llm_provider: str = os.getenv("LLM_PROVIDER", "mistral")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "mistral")
    llm_temperature: float = _float_env("LLM_TEMPERATURE", 0.2)
    llm_max_tokens: int = _int_env("LLM_MAX_TOKENS", 600)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:30b")
    ollama_embedding_model: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        "nomic-embed-text",
    )
    ollama_timeout_seconds: int = _int_env("OLLAMA_TIMEOUT_SECONDS", 180)
    ollama_min_tokens: int = _int_env("OLLAMA_MIN_TOKENS", 1200)
    embedding_batch_size: int = _int_env("EMBEDDING_BATCH_SIZE", 64)
    embedding_batch_delay_seconds: float = _float_env(
        "EMBEDDING_BATCH_DELAY_SECONDS", 1.0
    )
    embedding_max_retries: int = _int_env("EMBEDDING_MAX_RETRIES", 5)
    embedding_retry_sleep_seconds: float = _float_env(
        "EMBEDDING_RETRY_SLEEP_SECONDS", 10.0
    )
    top_k: int = _int_env("TOP_K", 3)
    retrieval_max_score: float | None = _optional_float_env(
        "RETRIEVAL_MAX_SCORE", 0.45
    )
    chunk_size: int = _int_env("CHUNK_SIZE", 800)
    chunk_overlap: int = _int_env("CHUNK_OVERLAP", 100)


settings = Settings()
