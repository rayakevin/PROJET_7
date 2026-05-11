"""Tests unitaires de la séparation passé/futur."""

from datetime import date

from app.rag.temporal import (
    classify_question_bucket,
    event_bucket,
    parse_event_date,
)


def test_classify_question_bucket_detects_past_question() -> None:
    """Vérifie qu'une formulation passée utilise l'index passé."""

    assert classify_question_bucket("Quels événements ont eu lieu en mai ?") == "past"


def test_classify_question_bucket_detects_future_question() -> None:
    """Vérifie qu'une formulation future utilise l'index futur."""

    assert classify_question_bucket("Quels événements sont à venir ?") == "future"


def test_classify_question_bucket_defaults_to_future() -> None:
    """Vérifie le choix par défaut orienté recommandation."""

    assert classify_question_bucket("Je cherche un concert de jazz") == "future"


def test_event_bucket_uses_end_date() -> None:
    """Vérifie qu'un événement terminé est classé dans le passé."""

    metadata = {
        "start": "2026-04-01T10:00:00+00:00",
        "end": "2026-04-01T12:00:00+00:00",
    }

    assert event_bucket(metadata, today=date(2026, 5, 6)) == "past"


def test_event_bucket_keeps_ongoing_event_in_future() -> None:
    """Vérifie qu'un événement encore ouvert reste dans l'index futur."""

    metadata = {
        "start": "2025-10-01T10:00:00+00:00",
        "end": "2026-06-01T12:00:00+00:00",
    }

    assert event_bucket(metadata, today=date(2026, 5, 6)) == "future"


def test_parse_event_date_handles_iso_datetime() -> None:
    """Vérifie la lecture des dates ISO OpenAgenda."""

    assert parse_event_date("2026-05-06T10:00:00+00:00") == date(2026, 5, 6)
