"""Script de reconstruction progressive du pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.build_dataset import build_dataset  # noqa: E402
from app.ingestion.fetch_events import fetch_events  # noqa: E402
from app.services.rebuild_service import rebuild_vector_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse les arguments de ligne de commande."""

    parser = argparse.ArgumentParser(
        description="Recupere et/ou normalise les evenements du POC."
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Recupere les evenements publics OpenAgenda via OpenDataSoft.",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Ville cible. Defaut : EVENTS_LOCATION.",
    )
    parser.add_argument(
        "--search",
        default=None,
        help="Terme de recherche optionnel.",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=None,
        help="Mot-cle optionnel. Peut etre repete plusieurs fois.",
    )
    parser.add_argument(
        "--raw-events-path",
        type=Path,
        default=None,
        help="Chemin du fichier JSON brut lu ou ecrit. Defaut : data/raw/events_raw.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Chemin du dataset normalise. Defaut : data/processed/events_processed.json",
    )
    parser.add_argument(
        "--quality-report-path",
        type=Path,
        default=None,
        help="Chemin du rapport qualite. Defaut : data/processed/events_quality_report.json",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Reconstruit aussi l'index vectoriel FAISS avec Mistral.",
    )
    parser.add_argument(
        "--vector-store-dir",
        type=Path,
        default=None,
        help="Dossier de sortie de l'index FAISS. Defaut : data/vector_store.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Limite optionnelle d'evenements a indexer pour un test rapide.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute la construction du dataset depuis le terminal."""

    args = parse_args()
    raw_events_path = args.raw_events_path

    if args.fetch:
        raw_events_path = fetch_events(
            output_path=args.raw_events_path,
            city=args.city,
            search=args.search,
            keywords=args.keyword,
        )
        print(f"Evenements bruts ecrits : {raw_events_path}")

    output_path = build_dataset(
        raw_events_path=raw_events_path,
        output_path=args.output_path,
        quality_report_path=args.quality_report_path,
    )

    print(f"Dataset normalise ecrit : {output_path}")

    if args.index:
        result = rebuild_vector_index(
            dataset_path=output_path,
            vector_store_dir=args.vector_store_dir,
            max_events=args.max_events,
        )
        print(f"Index vectoriel ecrit : {result.vector_store_dir}")
        print(f"Chunks indexes : {result.chunks_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
