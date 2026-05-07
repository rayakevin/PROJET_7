"""Outils temporels simples pour séparer passé et futur."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, date, datetime
from typing import Any, Literal


TemporalBucket = Literal["future", "past"]


PAST_TERMS = {
    "a eu lieu",
    "ont eu lieu",
    "avait lieu",
    "passe",
    "passes",
    "precedent",
    "precedente",
    "historique",
}

FUTURE_TERMS = {
    "a venir",
    "avenir",
    "futur",
    "future",
    "prochain",
    "prochaine",
    "prochains",
    "prochaines",
    "ce week end",
    "ce weekend",
    "cette semaine",
    "prochainement",
}


def classify_question_bucket(question: str) -> TemporalBucket:
    """Classe une question dans l'index futur ou passé.

    La règle est volontairement simple pour le POC :
    - une formulation passée explicite va vers l'index `past` ;
    - une formulation future explicite va vers l'index `future` ;
    - sinon on privilégie `future`, car l'assistant recommande surtout des
      événements à venir.
    """

    normalized_question = normalize_temporal_text(question)
    if any(term in normalized_question for term in PAST_TERMS):
        return "past"
    if any(term in normalized_question for term in FUTURE_TERMS):
        return "future"
    return "future"


def event_bucket(metadata: dict[str, Any], today: date | None = None) -> TemporalBucket:
    """Classe un événement selon sa date de fin."""

    reference_day = today or datetime.now(UTC).date()
    end = parse_event_date(str(metadata.get("end", "")))
    start = parse_event_date(str(metadata.get("start", "")))
    event_last_day = end or start

    if event_last_day is not None and event_last_day < reference_day:
        return "past"
    return "future"


def parse_event_date(value: str) -> date | None:
    """Convertit une date ISO issue d'OpenAgenda en objet `date`."""

    cleaned_value = value.strip()
    if not cleaned_value:
        return None

    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned_value):
            return date.fromisoformat(cleaned_value)

        iso_value = cleaned_value.replace("Z", "+00:00")
        return datetime.fromisoformat(iso_value).date()
    except ValueError:
        return None


def normalize_temporal_text(value: str) -> str:
    """Normalise une question pour repérer des expressions simples."""

    normalized = unicodedata.normalize("NFKD", value.lower())
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()
