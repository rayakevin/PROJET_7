"""Tests unitaires de normalisation des evenements."""

import json
from pathlib import Path

from app.ingestion.normalize_events import (
    clean_text,
    normalize_event,
    normalize_events,
    normalize_keywords,
)


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_clean_text_removes_extra_spaces() -> None:
    """Verifie que le nettoyage compacte les espaces inutiles."""

    assert clean_text("  Festival\n jazz\t gratuit  ") == "Festival jazz gratuit"


def test_normalize_keywords_returns_unique_lowercase_values() -> None:
    """Verifie le nettoyage, la casse et la deduplication des mots-cles."""

    keywords = normalize_keywords([" Jazz ", "concert", "JAZZ", ""])

    assert keywords == ["jazz", "concert"]


def test_normalize_event_builds_full_text() -> None:
    """Verifie qu'un evenement brut devient un document indexable."""

    raw_event = {
        "uid": "evt-001",
        "title": "Festival jazz en plein air",
        "description": "Concert gratuit.",
        "location_name": "Parc central",
        "city": "Paris",
        "start": "2026-05-02T19:30:00Z",
        "end": "2026-05-02T22:00:00Z",
        "keywords": ["jazz", "concert"],
    }

    normalized_event = normalize_event(raw_event)

    assert normalized_event["uid"] == "evt-001"
    assert normalized_event["keywords"] == ["jazz", "concert"]
    assert "Festival jazz en plein air" in normalized_event["full_text"]
    assert "Paris" in normalized_event["full_text"]


def test_normalize_event_supports_opendatasoft_shape() -> None:
    """Verifie la normalisation du dataset OpenDataSoft OpenAgenda."""

    raw_event = {
        "uid": "81452007",
        "title_fr": "Cosplaymania",
        "description_fr": "Un atelier avec une cosplayeuse.",
        "longdescription_fr": "<p>Venez decouvrir la mousse EVA.</p>",
        "keywords_fr": ["cosplay", "creation"],
        "location_name": "Cite des sciences",
        "location_city": "Paris",
        "firstdate_begin": "2025-04-27T09:00:00+00:00",
        "firstdate_end": "2025-04-27T11:00:00+00:00",
    }

    normalized_event = normalize_event(raw_event)

    assert normalized_event["uid"] == "81452007"
    assert normalized_event["title"] == "Cosplaymania"
    assert normalized_event["location_name"] == "Cite des sciences"
    assert normalized_event["city"] == "Paris"
    assert normalized_event["start"] == "2025-04-27T09:00:00+00:00"
    assert normalized_event["end"] == "2025-04-27T11:00:00+00:00"
    assert normalized_event["keywords"] == ["cosplay", "creation"]
    assert "Venez decouvrir la mousse EVA" in normalized_event["full_text"]
    assert "<p>" not in normalized_event["full_text"]


def test_normalize_events_works_with_fixture() -> None:
    """Verifie la normalisation de la fixture d'evenements."""

    raw_events = json.loads((FIXTURES_DIR / "sample_events.json").read_text())

    normalized_events = normalize_events(raw_events)

    assert len(normalized_events) == 2
    assert all(event["full_text"] for event in normalized_events)
