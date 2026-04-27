"""Controle qualite du dataset normalise."""

from __future__ import annotations

from typing import Any


REQUIRED_FIELDS = [
    "uid",
    "title",
    "description",
    "location_name",
    "city",
    "start",
    "end",
    "full_text",
]


def is_present(value: Any) -> bool:
    """Indique si une valeur normalisee contient une information exploitable."""

    return value is not None and str(value).strip() != ""


def assess_events_quality(
    events: list[dict[str, Any]],
    min_full_text_chars: int = 120,
) -> dict[str, Any]:
    """Calcule des indicateurs qualite pour le dataset normalise."""

    total_events = len(events)
    missing_counts = {
        field: sum(1 for event in events if not is_present(event.get(field)))
        for field in REQUIRED_FIELDS
    }
    coverage_percent = {
        field: round(
            100 * (total_events - missing_count) / total_events,
            2,
        )
        if total_events
        else 0.0
        for field, missing_count in missing_counts.items()
    }

    full_text_lengths = [len(str(event.get("full_text") or "")) for event in events]
    too_short_count = sum(
        1 for length in full_text_lengths if length < min_full_text_chars
    )

    seen_uids: set[str] = set()
    duplicate_uid_count = 0
    for event in events:
        uid = str(event.get("uid") or "")
        if uid in seen_uids:
            duplicate_uid_count += 1
        elif uid:
            seen_uids.add(uid)

    indexable_events = [
        event
        for event in events
        if all(is_present(event.get(field)) for field in REQUIRED_FIELDS)
        and len(str(event.get("full_text") or "")) >= min_full_text_chars
    ]

    return {
        "total_events": total_events,
        "required_fields": REQUIRED_FIELDS,
        "missing_counts": missing_counts,
        "coverage_percent": coverage_percent,
        "full_text": {
            "min_chars": min(full_text_lengths) if full_text_lengths else 0,
            "max_chars": max(full_text_lengths) if full_text_lengths else 0,
            "avg_chars": round(sum(full_text_lengths) / total_events, 2)
            if total_events
            else 0.0,
            "too_short_threshold_chars": min_full_text_chars,
            "too_short_count": too_short_count,
            "too_short_percent": round(100 * too_short_count / total_events, 2)
            if total_events
            else 0.0,
        },
        "duplicate_uid_count": duplicate_uid_count,
        "indexable_events": len(indexable_events),
        "indexable_percent": round(100 * len(indexable_events) / total_events, 2)
        if total_events
        else 0.0,
    }
