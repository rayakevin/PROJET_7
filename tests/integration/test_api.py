"""Tests d'integration de l'API FastAPI."""

from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.services.qa_service import AnswerSource, QAParameters, QAResponse
from app.services.rebuild_service import RebuildIndexResult


class FakeQAService:
    """Service QA de test sans appel Mistral."""

    def __init__(self) -> None:
        """Mémorise les paramètres reçus par le faux service."""

        self.last_parameters: QAParameters | None = None

    def ask(
        self,
        question: str,
        parameters: QAParameters | None = None,
    ) -> QAResponse:
        """Retourne une réponse stable sans appeler le vrai RAG."""

        self.last_parameters = parameters
        return QAResponse(
            question=question,
            answer="Voici une réponse de test.",
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
            parameters=parameters or QAParameters(),
        )


class RuntimeErrorQAService:
    """Service QA qui simule une erreur de génération ou de retrieval."""

    def ask(
        self,
        question: str,
        parameters: QAParameters | None = None,
    ) -> QAResponse:
        """Déclenche une erreur applicative volontaire."""

        del question, parameters
        raise RuntimeError("Erreur LLM simulée.")


def test_health_endpoint_returns_status() -> None:
    """Vérifie le endpoint de disponibilité."""

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "vector_store_ready" in payload


def test_metadata_endpoint_returns_public_configuration() -> None:
    """Vérifie que /metadata expose la configuration sans secret."""

    client = TestClient(app)

    response = client.get("/metadata")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_dataset_url"].startswith("https://")
    assert payload["embedding_model"]
    assert "mistral_api_key" not in payload
    assert "api_rebuild_token" not in payload


def test_ask_endpoint_returns_rag_answer() -> None:
    """Vérifie le endpoint /ask avec un service QA remplacé."""

    app.dependency_overrides[routes.get_qa_service] = lambda: FakeQAService()
    client = TestClient(app)

    response = client.post("/ask", json={"question": "Quels concerts jazz ?"})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Voici une réponse de test."
    assert payload["sources"][0]["title"] == "Concert jazz"
    assert payload["parameters"]["top_k"] is None


def test_ask_endpoint_accepts_runtime_parameters() -> None:
    """Vérifie que /ask accepte les hyperparamètres exposés à l'UI."""

    fake_service = FakeQAService()
    app.dependency_overrides[routes.get_qa_service] = lambda: fake_service
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": "Quels concerts jazz ?",
            "top_k": 4,
            "retrieval_max_score": 0.5,
            "temperature": 0.4,
            "max_tokens": 500,
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["parameters"]["top_k"] == 4
    assert fake_service.last_parameters == QAParameters(
        top_k=4,
        retrieval_max_score=0.5,
        temperature=0.4,
        max_tokens=500,
    )


def test_ask_endpoint_rejects_empty_question() -> None:
    """Vérifie la validation des questions vides."""

    client = TestClient(app)

    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_endpoint_rejects_invalid_llm_provider() -> None:
    """Vérifie la validation Pydantic du fournisseur LLM."""

    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": "Quels concerts jazz ?",
            "llm_provider": "local-inconnu",
        },
    )

    assert response.status_code == 422


def test_ask_endpoint_rejects_out_of_range_parameters() -> None:
    """Vérifie la validation des bornes des hyperparamètres."""

    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": "Quels concerts jazz ?",
            "top_k": 0,
            "temperature": 2.0,
            "max_tokens": 50,
        },
    )

    assert response.status_code == 422


def test_ask_endpoint_returns_503_when_service_initialization_fails(
    monkeypatch,
) -> None:
    """Vérifie le 503 propre si le service QA ne peut pas se charger."""

    class MissingIndexQAService:
        """Service QA qui simule un index absent au chargement."""

        def __init__(self) -> None:
            """Déclenche une erreur d'index absent."""

            raise FileNotFoundError("index.faiss")

    routes.reset_qa_service_cache()
    monkeypatch.setattr(routes, "QAService", MissingIndexQAService)
    client = TestClient(app)

    response = client.post("/ask", json={"question": "Quels concerts jazz ?"})

    routes.reset_qa_service_cache()
    assert response.status_code == 503
    assert "Index vectoriel absent" in response.json()["detail"]


def test_ask_endpoint_returns_503_when_service_runtime_fails() -> None:
    """Vérifie le 503 propre si la chaîne RAG échoue pendant /ask."""

    app.dependency_overrides[routes.get_qa_service] = lambda: RuntimeErrorQAService()
    client = TestClient(app)

    response = client.post("/ask", json={"question": "Quels concerts jazz ?"})

    app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["detail"] == "Erreur LLM simulée."


def test_rebuild_endpoint_runs_pipeline(monkeypatch, tmp_path) -> None:
    """Vérifie le endpoint /rebuild sans appel externe."""

    monkeypatch.setattr(routes.settings, "api_rebuild_token", "")
    dataset_path = tmp_path / "events_processed.json"

    def fake_build_dataset(raw_events_path=None):
        """Remplace la construction du dataset pendant le test API."""

        return dataset_path

    def fake_rebuild_vector_index(dataset_path=None, max_events=None):
        """Remplace la reconstruction FAISS pendant le test API."""

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
    """Vérifie la protection optionnelle du rebuild."""

    monkeypatch.setattr(routes.settings, "api_rebuild_token", "secret")
    client = TestClient(app)

    response = client.post("/rebuild", json={"fetch": False})

    assert response.status_code == 403
