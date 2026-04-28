"""Récupération des événements bruts depuis OpenDataSoft."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.clients.opendatasoft_client import EventsQuery, OpenDataSoftEventsClient
from app.config import settings
from app.utils.io import write_json


DEFAULT_RAW_EVENTS_FILENAME = "events_raw.json"


class EventsClient(Protocol):
    """Contrat minimal attendu pour récupérer des événements."""

    def build_default_query(
        self,
        city: str | None = None,
        search: str | None = None,
        keywords: list[str] | None = None,
    ) -> EventsQuery:
        """Construit une requête par défaut."""

    def list_events(self, query: EventsQuery) -> list[dict]:
        """Retourne les événements correspondant à la requête."""


def fetch_events(
    output_path: str | Path | None = None,
    city: str | None = None,
    search: str | None = None,
    keywords: list[str] | None = None,
    client: EventsClient | None = None,
) -> Path:
    """Récupère les événements publics et les sauvegarde en JSON brut."""

    events_client = client or OpenDataSoftEventsClient()
    query = events_client.build_default_query(
        city=city,
        search=search,
        keywords=keywords,
    )
    events = events_client.list_events(query)

    target_path = Path(output_path or settings.raw_data_dir / DEFAULT_RAW_EVENTS_FILENAME)
    return write_json(events, target_path)
