"""Configuration centrale du projet.

Ce module centralise le chargement des variables d'environnement
et la définition des paramètres globaux utilisés dans l'application.
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
    interim_data_dir: Path = Path(
        os.getenv("INTERIM_DATA_DIR", BASE_DIR / "data" / "interim")
    )
    processed_data_dir: Path = Path(
        os.getenv("PROCESSED_DATA_DIR", BASE_DIR / "data" / "processed")
    )
    vector_store_dir: Path = Path(
        os.getenv("VECTOR_STORE_DIR", BASE_DIR / "data" / "vector_store")
    )
    evaluation_data_dir: Path = Path(
        os.getenv("EVALUATION_DATA_DIR", BASE_DIR / "data" / "evaluation")
    )


settings = Settings()
