"""Tests unitaires du script d'evaluation RAG."""

from app.rag.chunking import TextChunk
from app.rag.vector_store import SearchResult
from scripts.evaluate_rag import (
    EvaluationExample,
    build_ragas_judge_model,
    build_required_metrics_summary,
    build_report,
    compute_local_metrics,
    load_examples,
    run_predictions,
)


class FakeRetriever:
    """Retriever de test sans index."""

    def retrieve(self, question: str) -> list[SearchResult]:
        """Retourne un contexte stable pour la question de test."""

        return [
            SearchResult(
                chunk=TextChunk(
                    id="evt-001::chunk-0",
                    text=f"Contexte pertinent pour {question}",
                    metadata={
                        "event_uid": "evt-001",
                        "title": "Concert jazz",
                        "city": "Paris",
                        "location_name": "JASS CLUB",
                        "start": "2026-01-01T10:00:00Z",
                        "end": "2026-01-01T12:00:00Z",
                    },
                ),
                score=0.25,
            )
        ]


class FakeAnswerGenerator:
    """Générateur de test sans appel LLM."""

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        """Retourne une réponse fixe sans appeler Mistral."""

        return "Concert jazz au JASS CLUB à Paris."


class FakeQAService:
    """Service QA minimal pour tester l'evaluation."""

    retriever = FakeRetriever()
    answer_generator = FakeAnswerGenerator()


def test_compute_local_metrics_includes_source_count() -> None:
    """Vérifie les métriques locales principales."""

    contexts = FakeRetriever().retrieve("question")

    metrics = compute_local_metrics(
        prediction="Concert jazz au JASS CLUB à Paris.",
        reference="Concert jazz à Paris.",
        contexts=contexts,
    )

    assert set(metrics) == {"source_count", "avg_retrieval_distance"}
    assert metrics["source_count"] == 1.0
    assert metrics["avg_retrieval_distance"] == 0.25


def test_run_predictions_builds_evaluation_rows() -> None:
    """Vérifie la collecte des prédictions RAG."""

    predictions = run_predictions(
        [EvaluationExample("Quels concerts jazz ?", "Concert jazz à Paris.")],
        qa_service=FakeQAService(),
    )

    assert len(predictions) == 1
    assert predictions[0].answer == "Concert jazz au JASS CLUB à Paris."
    assert predictions[0].sources[0]["title"] == "Concert jazz"
    assert predictions[0].contexts


def test_build_report_summarizes_predictions() -> None:
    """Vérifie la structure du rapport final."""

    predictions = run_predictions(
        [EvaluationExample("Quels concerts jazz ?", "Concert jazz à Paris.")],
        qa_service=FakeQAService(),
    )

    report = build_report(predictions, ragas_report=None)

    assert report["summary"]["questions_count"] == 1
    assert "source_count" in report["summary"]["local_metrics"]
    assert report["examples"][0]["question"] == "Quels concerts jazz ?"


def test_build_report_includes_required_ragas_metrics() -> None:
    """Vérifie les alias des métriques attendues par la grille."""

    predictions = run_predictions(
        [EvaluationExample("Quels concerts jazz ?", "Concert jazz à Paris.")],
        qa_service=FakeQAService(),
    )
    ragas_report = {
        "summary": {
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "llm_context_precision_with_reference": 0.9,
            "context_recall": 0.85,
        },
        "required_metrics": {
            "faithfulness": 0.8,
            "answer_relevance": 0.7,
            "context_precision": 0.9,
            "context_recall": 0.85,
        },
        "rows": [],
    }

    report = build_report(predictions, ragas_report=ragas_report)

    assert report["summary"]["required_ragas_metrics"] == {
        "faithfulness": 0.8,
        "answer_relevance": 0.7,
        "context_precision": 0.9,
        "context_recall": 0.85,
    }


def test_build_required_metrics_summary_maps_ragas_names() -> None:
    """Vérifie la traduction des noms internes Ragas."""

    summary = build_required_metrics_summary(
        {
            "faithfulness": 0.1,
            "answer_relevancy": 0.2,
            "llm_context_precision_with_reference": 0.3,
            "context_recall": 0.4,
        }
    )

    assert summary == {
        "faithfulness": 0.1,
        "answer_relevance": 0.2,
        "context_precision": 0.3,
        "context_recall": 0.4,
    }


def test_build_ragas_judge_model_uses_ollama_by_default(monkeypatch) -> None:
    """Vérifie que le juge Ragas local est le choix par défaut."""

    monkeypatch.setattr("scripts.evaluate_rag.settings.ragas_llm_provider", "ollama")
    monkeypatch.setattr("scripts.evaluate_rag.settings.ragas_llm_model", "qwen3:30b")

    judge = build_ragas_judge_model()

    assert judge.model == "qwen3:30b"
    assert judge._llm_type == "ollama-chat"


def test_load_examples_reads_fixture(tmp_path) -> None:
    """Vérifie le chargement d'un jeu annoté."""

    dataset_path = tmp_path / "qa.json"
    dataset_path.write_text(
        '[{"question": "Q", "reference_answer": "A"}]',
        encoding="utf-8",
    )

    examples = load_examples(dataset_path)

    assert examples == [EvaluationExample(question="Q", reference_answer="A")]
