"""Tests unitaires du client OpenDataSoft."""

from typing import Any

from app.clients.opendatasoft_client import OpenDataSoftEventsClient


class FakeResponse:
    """Reponse HTTP minimale pour les tests."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    """Session HTTP minimale pour capturer les parametres envoyes."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        params: dict[str, str | int],
        timeout: int,
    ) -> FakeResponse:
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
    """Verifie la construction d'un appel filtre vers OpenDataSoft."""

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
