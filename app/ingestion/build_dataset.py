"""Construction du dataset final."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.ingestion.normalize_events import normalize_events
from app.ingestion.quality import assess_events_quality
from app.utils.io import read_json, write_json


DEFAULT_RAW_EVENTS_FILENAME = "events_raw.json"
DEFAULT_PROCESSED_EVENTS_FILENAME = "events_processed.json"
DEFAULT_QUALITY_REPORT_FILENAME = "events_quality_report.json"


def build_dataset(
    raw_events_path: str | Path | None = None,
    output_path: str | Path | None = None,
    quality_report_path: str | Path | None = None,
) -> Path:
    """Construit le dataset normalisé et son rapport qualité."""

    source_path = Path(raw_events_path or settings.raw_data_dir / DEFAULT_RAW_EVENTS_FILENAME)
    target_path = Path(output_path or settings.processed_data_dir / DEFAULT_PROCESSED_EVENTS_FILENAME)
    report_path = Path(
        quality_report_path
        or settings.processed_data_dir / DEFAULT_QUALITY_REPORT_FILENAME
    )

    raw_events = read_json(source_path)
    normalized_events = normalize_events(raw_events)
    quality_report = assess_events_quality(normalized_events)

    write_json(quality_report, report_path)
    return write_json(normalized_events, target_path)
