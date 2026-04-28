"""Tests unitaires du client OpenDataSoft."""

from typing import Any

from app.clients.opendatasoft_client import OpenDataSoftEventsClient


class FakeResponse:
    """Réponse HTTP minimale pour les tests."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Stocke le JSON qui sera retourné par la fausse réponse."""

        self._payload = payload

    def json(self) -> dict[str, Any]:
        """Retourne le payload de test."""

        return self._payload

    def raise_for_status(self) -> None:
        """Simule une réponse HTTP sans erreur."""

        return None


class FakeSession:
    """Session HTTP minimale pour capturer les paramètres envoyés."""

    def __init__(self) -> None:
        """Prépare la liste des appels HTTP capturés."""

        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        params: dict[str, str | int],
        timeout: int,
    ) -> FakeResponse:
        """Capture l'appel GET et retourne une page OpenDataSoft factice."""

        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(
            {
                "total_count": 1,
                "results": [
                    {
                        "uid": "evt-ods-001",
                        "title_fr": "Concert jazz",
                        "location_city": "Paris",
                        "keywords_fr": ["jazz"],
                    }
                ],
            }
        )


def test_opendatasoft_client_builds_filtered_request() -> None:
    """Vérifie la construction d'un appel filtré vers OpenDataSoft."""

    client = OpenDataSoftEventsClient(
        records_url="https://example.test/records",
        timeout_seconds=10,
    )
    fake_session = FakeSession()
    client.session = fake_session

    query = client.build_default_query(
        city="Paris",
        search="jazz",
        keywords=["jazz"],
    )
    events = client.list_events(query)

    assert events == [
        {
            "uid": "evt-ods-001",
            "title_fr": "Concert jazz",
            "location_city": "Paris",
            "keywords_fr": ["jazz"],
        }
    ]
    params = fake_session.calls[0]["params"]
    assert params["q"] == "jazz"
    assert 'location_city="Paris"' in params["where"]
    assert "firstdate_begin >=" in params["where"]
