"""Client OpenDataSoft pour le dataset public OpenAgenda."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from app.config import settings


@dataclass(slots=True)
class EventsQuery:
    """Filtres utilisés pour lire le dataset public d'événements."""

    city: str | None = None
    search: str | None = None
    keywords: list[str] | None = None
    date_gte: str | None = None
    date_lte: str | None = None
    size: int = settings.events_page_size


class OpenDataSoftEventsClient:
    """Lit les événements publics OpenAgenda via le portail OpenDataSoft."""

    def __init__(
        self,
        records_url: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        """Initialise le client HTTP avec l'URL et le timeout configurés."""

        self.records_url = records_url or settings.opendatasoft_records_url
        self.timeout_seconds = timeout_seconds or settings.request_timeout_seconds
        self.session = requests.Session()

    def build_default_query(
        self,
        city: str | None = None,
        search: str | None = None,
        keywords: list[str] | None = None,
    ) -> EventsQuery:
        """Construit une requête compatible avec le protocole d'ingestion."""

        now = datetime.now(UTC)
        date_gte = (now - timedelta(days=settings.events_lookback_days)).date()
        date_lte = (now + timedelta(days=settings.events_lookahead_days)).date()

        return EventsQuery(
            city=city or settings.events_location,
            search=search,
            keywords=keywords,
            date_gte=date_gte.isoformat(),
            date_lte=date_lte.isoformat(),
            size=settings.events_page_size,
        )

    def list_events(self, query: EventsQuery) -> list[dict[str, Any]]:
        """Retourne tous les événements correspondant aux filtres principaux."""

        offset = 0
        limit = min(query.size, 100)
        events: list[dict[str, Any]] = []

        while True:
            # OpenDataSoft renvoie les résultats par page : on avance avec
            # `offset` jusqu'à avoir lu toutes les pages disponibles.
            payload = self._request_records(query=query, limit=limit, offset=offset)
            records = payload.get("results", [])
            events.extend(self._filter_keywords(records, query.keywords))

            offset += limit
            if offset >= int(payload.get("total_count", 0)) or not records:
                break

        return events

    def _request_records(
        self,
        query: EventsQuery,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        """Exécute un appel paginé vers l'endpoint OpenDataSoft."""

        params: dict[str, str | int] = {
            "limit": limit,
            "offset": offset,
            "order_by": "firstdate_begin asc",
            "where": self._build_where(query),
        }

        if query.search:
            params["q"] = query.search

        response = self.session.get(
            self.records_url,
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _build_where(query: EventsQuery) -> str:
        """Construit la clause `where` utilisée par l'API OpenDataSoft."""

        clauses = ["location_city is not null", "firstdate_begin is not null"]

        if query.city:
            clauses.append(f'location_city="{query.city}"')
        if query.date_gte:
            clauses.append(f"firstdate_begin >= date'{query.date_gte[:10]}'")
        if query.date_lte:
            clauses.append(f"firstdate_begin <= date'{query.date_lte[:10]}'")

        return " and ".join(clauses)

    @staticmethod
    def _filter_keywords(
        records: list[dict[str, Any]],
        keywords: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Filtre localement les événements qui contiennent tous les mots-clés."""

        if not keywords:
            return records

        expected = {keyword.lower() for keyword in keywords}
        filtered_records = []
        for record in records:
            record_keywords = {
                str(keyword).lower() for keyword in record.get("keywords_fr") or []
            }
            if expected.issubset(record_keywords):
                filtered_records.append(record)

        return filtered_records
