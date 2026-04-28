"""Tests d'integration de l'API FastAPI."""

from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.services.qa_service import AnswerSource, QAResponse
from app.services.rebuild_service import RebuildIndexResult


class FakeQAService:
    """Service QA de test sans appel Mistral."""

    def ask(self, question: str) -> QAResponse:
        return QAResponse(
            question=question,
            answer="Voici une reponse de test.",
            sources=[
                AnswerSource(
                    chunk_id="evt-001::chunk-0",
                    event_uid="evt-001",
                    title="Concert jazz",
                    city="Paris",
                    location_name="Parc central",
                    start="2026-05-02T19:30:00Z",
                    end="2026-05-02T22:00:00Z",
                    score=0.42,
                )
            ],
        )


def test_health_endpoint_returns_status() -> None:
    """Verifie le endpoint de disponibilite."""

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "vector_store_ready" in payload


def test_ask_endpoint_returns_rag_answer() -> None:
    """Verifie le endpoint /ask avec un service QA remplace."""

    app.dependency_overrides[routes.get_qa_service] = lambda: FakeQAService()
    client = TestClient(app)

    response = client.post("/ask", json={"question": "Quels concerts jazz ?"})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Voici une reponse de test."
    assert payload["sources"][0]["title"] == "Concert jazz"


def test_ask_endpoint_rejects_empty_question() -> None:
    """Verifie la validation des questions vides."""

    client = TestClient(app)

    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_rebuild_endpoint_runs_pipeline(monkeypatch, tmp_path) -> None:
    """Verifie le endpoint /rebuild sans appel externe."""

    monkeypatch.setattr(routes.settings, "api_rebuild_token", "")
    dataset_path = tmp_path / "events_processed.json"

    def fake_build_dataset(raw_events_path=None):
        return dataset_path

    def fake_rebuild_vector_index(dataset_path=None, max_events=None):
        return RebuildIndexResult(
            dataset_path=str(dataset_path),
            vector_store_dir=str(tmp_path / "vector_store"),
            events_count=2,
            chunks_count=3,
        )

    monkeypatch.setattr(routes, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(routes, "rebuild_vector_index", fake_rebuild_vector_index)
    client = TestClient(app)

    response = client.post("/rebuild", json={"fetch": False, "max_events": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["events_count"] == 2
    assert payload["chunks_count"] == 3


def test_rebuild_endpoint_can_require_token(monkeypatch) -> None:
    """Verifie la protection optionnelle du rebuild."""

    monkeypatch.setattr(routes.settings, "api_rebuild_token", "secret")
    client = TestClient(app)

    response = client.post("/rebuild", json={"fetch": False})

    assert response.status_code == 403
