"""Tests unitaires du contrôle qualité."""

from app.ingestion.quality import assess_events_quality


def test_assess_events_quality_reports_full_text_quality() -> None:
    """Vérifie les indicateurs principaux du rapport qualité."""

    events = [
        {
            "uid": "evt-001",
            "title": "Concert jazz",
            "description": "Un concert de jazz en plein air à Paris.",
            "location_name": "Parc central",
            "city": "Paris",
            "start": "2026-05-02T19:30:00Z",
            "end": "2026-05-02T22:00:00Z",
            "full_text": "x" * 160,
        },
        {
            "uid": "evt-001",
            "title": "",
            "description": "Description trop courte.",
            "location_name": "Salle test",
            "city": "Paris",
            "start": "2026-05-03T19:30:00Z",
            "end": "2026-05-03T22:00:00Z",
            "full_text": "court",
        },
    ]

    report = assess_events_quality(events, min_full_text_chars=120)

    assert report["total_events"] == 2
    assert report["missing_counts"]["title"] == 1
    assert report["coverage_percent"]["title"] == 50.0
    assert report["full_text"]["too_short_count"] == 1
    assert report["duplicate_uid_count"] == 1
    assert report["indexable_events"] == 1
    assert report["indexable_percent"] == 50.0
