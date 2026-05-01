"""Tests unitaires des contraintes temporelles du retrieval."""

from datetime import date

from app.rag.temporal import (
    DateFilter,
    current_weekend,
    detect_temporal_filter,
    event_matches_date_filter,
    next_week,
)


def test_detect_temporal_filter_finds_current_weekend() -> None:
    """Vérifie la détection de l'expression 'ce week-end'."""

    date_filter = detect_temporal_filter(
        "Quels événements à Paris ce week-end ?",
        today=date(2026, 5, 1),
    )

    assert date_filter == DateFilter(
        start=date(2026, 5, 2),
        end=date(2026, 5, 3),
        label="ce week-end",
    )


def test_detect_temporal_filter_finds_next_weeks() -> None:
    """Vérifie la détection des événements des prochaines semaines."""

    date_filter = detect_temporal_filter(
        "Donne-moi les événements des prochaines semaines",
        today=date(2026, 5, 1),
    )

    assert date_filter == DateFilter(
        start=date(2026, 5, 1),
        end=date(2026, 5, 29),
        label="prochaines semaines",
    )


def test_event_matches_date_filter_rejects_past_event() -> None:
    """Vérifie qu'un événement passé ne passe pas un filtre futur."""

    date_filter = DateFilter(
        start=date(2026, 5, 2),
        end=date(2026, 5, 3),
        label="ce week-end",
    )

    assert not event_matches_date_filter(
        {"start": "2026-04-20T10:00:00Z", "end": "2026-04-20T12:00:00Z"},
        date_filter,
    )
    assert event_matches_date_filter(
        {"start": "2026-05-02T10:00:00Z", "end": "2026-05-02T12:00:00Z"},
        date_filter,
    )


def test_current_weekend_keeps_sunday_when_already_weekend() -> None:
    """Vérifie le calcul du week-end quand on est déjà dimanche."""

    assert current_weekend(date(2026, 5, 3)) == DateFilter(
        start=date(2026, 5, 3),
        end=date(2026, 5, 3),
        label="ce week-end",
    )


def test_next_week_starts_on_next_monday() -> None:
    """Vérifie que 'semaine prochaine' démarre au lundi suivant."""

    assert next_week(date(2026, 5, 1)) == DateFilter(
        start=date(2026, 5, 4),
        end=date(2026, 5, 10),
        label="semaine prochaine",
    )
