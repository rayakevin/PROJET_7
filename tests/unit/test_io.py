"""Tests unitaires des helpers d'entrée / sortie."""

from app.utils.io import read_json, write_json


def test_write_json_creates_parent_directory_and_file(tmp_path) -> None:
    """Vérifie que l'écriture JSON crée le dossier parent si besoin."""

    output_path = tmp_path / "nested" / "events.json"
    data = [{"uid": "evt-001"}]

    written_path = write_json(data, output_path)

    assert written_path == output_path
    assert read_json(output_path) == data
