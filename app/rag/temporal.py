"""Détection et application de contraintes temporelles simples."""

from __future__ import annotations

import calendar
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any


@dataclass(frozen=True, slots=True)
class DateFilter:
    """Fenêtre temporelle déduite d'une question utilisateur."""

    start: date
    end: date | None
    label: str


def detect_temporal_filter(question: str, today: date | None = None) -> DateFilter | None:
    """Détecte quelques expressions temporelles françaises courantes."""

    normalized_question = normalize_temporal_text(question)
    reference_day = today or datetime.now(UTC).date()

    if "aujourd hui" in normalized_question or "aujourdhui" in normalized_question:
        return DateFilter(reference_day, reference_day, "aujourd'hui")

    if "demain" in normalized_question:
        tomorrow = reference_day + timedelta(days=1)
        return DateFilter(tomorrow, tomorrow, "demain")

    if (
        "ce week end" in normalized_question
        or "ce weekend" in normalized_question
        or "week end" in normalized_question
        or "weekend" in normalized_question
    ):
        return current_weekend(reference_day)

    if "semaine prochaine" in normalized_question:
        return next_week(reference_day)

    if (
        "prochaines semaines" in normalized_question
        or "prochains semaines" in normalized_question
        or "semaines a venir" in normalized_question
    ):
        return DateFilter(
            reference_day,
            reference_day + timedelta(days=28),
            "prochaines semaines",
        )

    if "prochains jours" in normalized_question:
        return DateFilter(
            reference_day,
            reference_day + timedelta(days=7),
            "prochains jours",
        )

    if "cette semaine" in normalized_question:
        days_until_sunday = 6 - reference_day.weekday()
        return DateFilter(
            reference_day,
            reference_day + timedelta(days=days_until_sunday),
            "cette semaine",
        )

    if "ce mois" in normalized_question:
        last_day = calendar.monthrange(reference_day.year, reference_day.month)[1]
        return DateFilter(
            reference_day,
            date(reference_day.year, reference_day.month, last_day),
            "ce mois",
        )

    future_terms = {
        "a venir",
        "avenir",
        "futur",
        "futurs",
        "future",
        "futures",
        "prochain",
        "prochaine",
        "prochains",
        "prochaines",
        "prochainement",
    }
    if any(term in normalized_question for term in future_terms):
        return DateFilter(reference_day, None, "à venir")

    return None


def event_matches_date_filter(
    metadata: dict[str, Any],
    date_filter: DateFilter,
) -> bool:
    """Indique si un événement chevauche la fenêtre temporelle demandée."""

    start = parse_event_date(str(metadata.get("start", "")))
    end = parse_event_date(str(metadata.get("end", "")))

    if start is None and end is None:
        return False

    start = start or end
    end = end or start
    if start is None or end is None:
        return False

    filter_end = date_filter.end or date.max
    return start <= filter_end and end >= date_filter.start


def event_start_date(metadata: dict[str, Any]) -> date | None:
    """Retourne la date de début utilisée pour trier les événements."""

    return parse_event_date(str(metadata.get("start", "")))


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


def current_weekend(reference_day: date) -> DateFilter:
    """Retourne le week-end courant ou le prochain samedi-dimanche."""

    weekday = reference_day.weekday()
    if weekday <= 4:
        start = reference_day + timedelta(days=5 - weekday)
    else:
        start = reference_day

    days_until_sunday = 6 - start.weekday()
    end = start + timedelta(days=days_until_sunday)
    return DateFilter(start, end, "ce week-end")


def next_week(reference_day: date) -> DateFilter:
    """Retourne la semaine civile suivant la date de référence."""

    days_until_next_monday = 7 - reference_day.weekday()
    start = reference_day + timedelta(days=days_until_next_monday)
    return DateFilter(start, start + timedelta(days=6), "semaine prochaine")


def normalize_temporal_text(value: str) -> str:
    """Normalise un texte pour repérer des expressions temporelles."""

    normalized = unicodedata.normalize("NFKD", value.lower())
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()
