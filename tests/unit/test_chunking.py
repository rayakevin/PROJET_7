"""Tests unitaires du chunking."""

import pytest

from app.rag.chunking import chunk_event, chunk_events, split_text


def test_split_text_respects_size_and_overlap() -> None:
    """Vérifie le découpage avec chevauchement."""

    text = " ".join(f"mot{i}" for i in range(40))

    chunks = split_text(text, chunk_size=60, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert chunks[0][-10:].strip() in chunks[1]


def test_split_text_rejects_invalid_parameters() -> None:
    """Vérifie la validation des paramètres."""

    with pytest.raises(ValueError):
        split_text("texte", chunk_size=100, chunk_overlap=100)


def test_chunk_event_keeps_event_metadata() -> None:
    """Vérifie que les métadonnées métier suivent chaque chunk."""

    event = {
        "uid": "evt-001",
        "title": "Concert jazz",
        "city": "Paris",
        "location_name": "Parc central",
        "start": "2026-05-02T19:30:00Z",
        "end": "2026-05-02T22:00:00Z",
        "keywords": ["jazz", "concert"],
        "full_text": "Titre : Concert jazz. " * 20,
    }

    chunks = chunk_event(event, chunk_size=80, chunk_overlap=10)

    assert chunks
    assert chunks[0].id == "evt-001::chunk-0"
    assert chunks[0].metadata["event_uid"] == "evt-001"
    assert chunks[0].metadata["title"] == "Concert jazz"
    assert chunks[0].metadata["city"] == "Paris"


def test_chunk_events_flattens_all_chunks() -> None:
    """Vérifie le découpage d'une liste d'événements."""

    events = [
        {"uid": "evt-001", "full_text": "A" * 120},
        {"uid": "evt-002", "full_text": "B" * 120},
    ]

    chunks = chunk_events(events, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 4
    assert {chunk.metadata["event_uid"] for chunk in chunks} == {"evt-001", "evt-002"}
