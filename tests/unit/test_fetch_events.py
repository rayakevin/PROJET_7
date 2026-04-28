"""Tests unitaires de récupération des événements bruts."""

from typing import Any

from app.clients.opendatasoft_client import EventsQuery
from app.ingestion.fetch_events import fetch_events
from app.utils.io import read_json


class FakeEventsClient:
    """Faux client utilisé pour tester sans réseau."""

    def __init__(self) -> None:
        """Prépare la capture de la dernière requête reçue."""

        self.last_query: EventsQuery | None = None

    def build_default_query(
        self,
        city: str | None = None,
        search: str | None = None,
        keywords: list[str] | None = None,
    ) -> EventsQuery:
        """Construit une requête de test et la mémorise."""

        query = EventsQuery(
            city=city,
            search=search,
            keywords=keywords,
        )
        self.last_query = query
        return query

    def list_events(self, query: EventsQuery) -> list[dict[str, Any]]:
        """Retourne un événement minimal sans appel réseau."""

        return [
            {
                "uid": "evt-001",
                "title_fr": "Festival jazz",
                "location_city": query.city,
            }
        ]


def test_fetch_events_writes_raw_events(tmp_path) -> None:
    """Vérifie que les événements récupérés sont écrits en JSON brut."""

    output_path = tmp_path / "raw" / "events_raw.json"
    client = FakeEventsClient()

    written_path = fetch_events(
        output_path=output_path,
        city="Paris",
        search="jazz",
        keywords=["concert"],
        client=client,
    )

    events = read_json(written_path)

    assert written_path == output_path
    assert events == [
        {"uid": "evt-001", "title_fr": "Festival jazz", "location_city": "Paris"}
    ]
    assert client.last_query is not None
    assert client.last_query.search == "jazz"
    assert client.last_query.keywords == ["concert"]
