"""Evaluation automatique du systeme RAG."""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics._answer_relevance import ResponseRelevancy  # noqa: E402
from ragas.metrics._answer_similarity import SemanticSimilarity  # noqa: E402
from ragas.metrics._context_precision import LLMContextPrecisionWithReference  # noqa: E402
from ragas.metrics._faithfulness import Faithfulness  # noqa: E402

from app.config import settings  # noqa: E402
from app.rag.embeddings import MistralEmbeddingModel  # noqa: E402
from app.rag.llm import MistralChatModel  # noqa: E402
from app.rag.vector_store import LangChainEmbeddingAdapter, SearchResult  # noqa: E402
from app.services.qa_service import QAService, build_sources  # noqa: E402
from app.utils.io import read_json, write_json  # noqa: E402


DEFAULT_QA_DATASET_PATH = PROJECT_ROOT / "tests" / "fixtures" / "qa_dataset.json"


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """Question annotee pour l'evaluation."""

    question: str
    reference_answer: str


@dataclass(frozen=True, slots=True)
class EvaluationPrediction:
    """Prediction RAG et contexte associe."""

    question: str
    reference_answer: str
    answer: str
    contexts: list[str]
    sources: list[dict[str, Any]]
    metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    """Parse les arguments de ligne de commande."""

    parser = argparse.ArgumentParser(description="Evalue le chatbot RAG.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_QA_DATASET_PATH,
        help="Chemin du jeu de test annote.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.evaluation_data_dir / "results",
        help="Dossier de sortie du rapport d'evaluation.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Ignore les metriques Ragas qui utilisent les embeddings Mistral.",
    )
    return parser.parse_args()


def load_examples(path: str | Path) -> list[EvaluationExample]:
    """Charge le jeu de test annote."""

    rows = read_json(path)
    return [
        EvaluationExample(
            question=str(row["question"]),
            reference_answer=str(row["reference_answer"]),
        )
        for row in rows
    ]


def run_predictions(
    examples: list[EvaluationExample],
    qa_service: QAService | None = None,
) -> list[EvaluationPrediction]:
    """Interroge le systeme RAG sur chaque question annotee."""

    service = qa_service or QAService()
    predictions: list[EvaluationPrediction] = []

    for example in examples:
        contexts = service.retriever.retrieve(example.question)
        answer = service.answer_generator.generate(example.question, contexts)
        predictions.append(
            EvaluationPrediction(
                question=example.question,
                reference_answer=example.reference_answer,
                answer=answer,
                contexts=[result.chunk.text for result in contexts],
                sources=[asdict(source) for source in build_sources(contexts)],
                metrics=compute_local_metrics(
                    prediction=answer,
                    reference=example.reference_answer,
                    contexts=contexts,
                ),
            )
        )

    return predictions


def compute_local_metrics(
    prediction: str,
    reference: str,
    contexts: list[SearchResult],
) -> dict[str, float]:
    """Calcule des metriques rapides centrees sur le retrieval."""

    del prediction, reference
    return {
        "source_count": float(len(contexts)),
        "avg_retrieval_score": round(
            sum(result.score for result in contexts) / len(contexts),
            4,
        )
        if contexts
        else 0.0,
    }


def run_ragas_metrics(
    predictions: list[EvaluationPrediction],
) -> dict[str, Any]:
    """Calcule les metriques Ragas disponibles pour le POC."""

    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY doit etre renseignee pour lancer Ragas.")

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=prediction.question,
                response=prediction.answer,
                reference=prediction.reference_answer,
                retrieved_contexts=prediction.contexts,
            )
            for prediction in predictions
        ]
    )
    embeddings = LangChainEmbeddingAdapter(MistralEmbeddingModel())
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="LangchainLLMWrapper is deprecated.*",
            category=DeprecationWarning,
        )
        llm = LangchainLLMWrapper(
            MistralChatModel(
                api_key=settings.mistral_api_key,
                model=settings.mistral_chat_model,
                temperature=0.0,
                max_tokens=max(settings.llm_max_tokens, 2000),
            )
        )
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            SemanticSimilarity(),
        ],
        llm=llm,
        embeddings=embeddings,
        show_progress=False,
        raise_exceptions=False,
    )
    rows = result.to_pandas().to_dict(orient="records")
    metric_names = [key for key in result.scores[0] if result.scores] if rows else []
    summary = summarize_metric_rows(rows, metric_names)
    return {
        "summary": summary,
        "required_metrics": build_required_metrics_summary(summary),
        "rows": rows,
    }


def summarize_metric_rows(
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, float]:
    """Agrege les lignes Ragas par nom de metrique."""

    return {
        metric_name: round(
            sum(to_finite_float(row.get(metric_name, 0.0)) for row in rows) / len(rows),
            4,
        )
        if rows
        else 0.0
        for metric_name in metric_names
    }


def to_finite_float(value: Any) -> float:
    """Convertit une valeur de metrique en float JSON-compatible."""

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return 0.0
    return converted if math.isfinite(converted) else 0.0


def build_required_metrics_summary(summary: dict[str, float]) -> dict[str, float | None]:
    """Expose les metriques attendues par la grille avec des noms lisibles."""

    return {
        "faithfulness": summary.get("faithfulness"),
        "answer_relevance": summary.get("answer_relevancy"),
        "context_precision": summary.get("llm_context_precision_with_reference"),
    }


def summarize_predictions(
    predictions: list[EvaluationPrediction],
    ragas_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Agrege les metriques par question en resume global."""

    local_metric_names = predictions[0].metrics.keys() if predictions else []
    summary: dict[str, Any] = {
        "questions_count": len(predictions),
        "local_metrics": {
            metric_name: round(
                sum(prediction.metrics[metric_name] for prediction in predictions)
                / len(predictions),
                4,
            )
            if predictions
            else 0.0
            for metric_name in local_metric_names
        },
    }
    if ragas_report is not None:
        summary["ragas_metrics"] = ragas_report["summary"]
        summary["required_ragas_metrics"] = ragas_report["required_metrics"]
    return summary


def build_report(
    predictions: list[EvaluationPrediction],
    ragas_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Construit le rapport final serialisable."""

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": summarize_predictions(predictions, ragas_report),
        "examples": [
            {
                "question": prediction.question,
                "reference_answer": prediction.reference_answer,
                "answer": prediction.answer,
                "sources": prediction.sources,
                "contexts": prediction.contexts,
                "metrics": prediction.metrics,
            }
            for prediction in predictions
        ],
        "ragas": ragas_report,
    }


def write_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    """Ecrit un rapport horodate et une copie latest."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = target_dir / f"rag_evaluation_{timestamp}.json"
    latest_path = target_dir / "rag_evaluation_latest.json"
    write_json(report, report_path)
    write_json(report, latest_path)
    return report_path


def main() -> int:
    """Execute l'evaluation RAG."""

    args = parse_args()
    examples = load_examples(args.dataset_path)
    predictions = run_predictions(examples)
    ragas_report = None if args.skip_ragas else run_ragas_metrics(predictions)
    report = build_report(predictions, ragas_report)
    report_path = write_report(report, args.output_dir)

    print(f"Rapport d'evaluation ecrit : {report_path}")
    print(f"Questions evaluees : {report['summary']['questions_count']}")
    print(f"Metriques locales : {report['summary']['local_metrics']}")
    if ragas_report is not None:
        print(f"Metriques Ragas : {report['summary']['ragas_metrics']}")
        print(
            "Metriques attendues : "
            f"{report['summary']['required_ragas_metrics']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
