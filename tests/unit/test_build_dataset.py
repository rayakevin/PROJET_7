"""Tests unitaires de construction du dataset normalise."""

import json
from pathlib import Path

from app.ingestion.build_dataset import build_dataset
from app.utils.io import read_json


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_build_dataset_writes_normalized_events(tmp_path) -> None:
    """Verifie qu'un fichier brut devient un fichier normalise."""

    raw_events = json.loads((FIXTURES_DIR / "sample_events.json").read_text())
    raw_path = tmp_path / "raw" / "events_raw.json"
    output_path = tmp_path / "processed" / "events_processed.json"
    quality_report_path = tmp_path / "processed" / "events_quality_report.json"

    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(json.dumps(raw_events), encoding="utf-8")

    written_path = build_dataset(
        raw_events_path=raw_path,
        output_path=output_path,
        quality_report_path=quality_report_path,
    )
    processed_events = read_json(written_path)
    quality_report = read_json(quality_report_path)

    assert written_path == output_path
    assert len(processed_events) == 2
    assert processed_events[0]["uid"] == "evt-001"
    assert "Festival jazz en plein air" in processed_events[0]["full_text"]
    assert "Paris" in processed_events[0]["full_text"]
    assert quality_report["total_events"] == 2
    assert quality_report["coverage_percent"]["full_text"] == 100.0
    assert quality_report["duplicate_uid_count"] == 0
