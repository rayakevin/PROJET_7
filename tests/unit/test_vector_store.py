"""Tests unitaires du filtrage des résultats FAISS."""

from langchain_core.documents import Document

from app.rag.chunking import TextChunk
from app.rag.vector_store import (
    FaissVectorStore,
    SearchResult,
    deduplicate_by_event,
    matches_query_focus,
    tokenize,
)


class FakeFaissStore:
    """Store FAISS minimal pour tester le filtrage sans index réel."""

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        """Retourne deux documents avec distances FAISS contrôlées."""

        assert query == "concert jazz"
        assert k == 3
        return [
            (
                Document(
                    page_content="Document tres proche",
                    metadata={"chunk_id": "evt-001::chunk-0", "title": "A"},
                ),
                0.2,
            ),
            (
                Document(
                    page_content="Document moyen",
                    metadata={"chunk_id": "evt-002::chunk-0", "title": "B"},
                ),
                0.5,
            ),
        ]


def test_search_filters_results_above_max_score() -> None:
    """Vérifie qu'un seuil retire les contextes trop éloignés."""

    vector_store = FaissVectorStore(store=FakeFaissStore(), chunks=[])

    results = vector_store.search(
        "concert jazz",
        top_k=3,
        max_score=0.45,
    )

    assert len(results) == 1
    assert results[0].chunk.id == "evt-001::chunk-0"


def test_search_keeps_best_result_when_threshold_filters_everything() -> None:
    """Vérifie le fallback pour éviter un contexte vide."""

    vector_store = FaissVectorStore(store=FakeFaissStore(), chunks=[])

    results = vector_store.search(
        "concert jazz",
        top_k=3,
        max_score=0.1,
    )

    assert len(results) == 1
    assert results[0].chunk.id == "evt-001::chunk-0"


def test_search_adds_lexical_candidates_from_metadata() -> None:
    """Vérifie que le reranking peut récupérer un candidat lexical."""

    chunks = [
        TextChunk(
            id="evt-003::chunk-0",
            text="Titre : Cinema documentaire",
            metadata={
                "event_uid": "evt-003",
                "title": "Festival de cinema",
                "keywords": ["cinema"],
            },
        )
    ]
    vector_store = FaissVectorStore(store=FakeFaissStore(), chunks=chunks)

    results = vector_store.search(
        "concert jazz",
        top_k=3,
        max_score=None,
    )

    assert any(result.chunk.id == "evt-001::chunk-0" for result in results)


def test_deduplicate_by_event_keeps_first_result() -> None:
    """Vérifie la diversification des sources par événement."""

    results = [
        SearchResult(
            chunk=TextChunk(
                id="evt-001::chunk-0",
                text="A",
                metadata={"event_uid": "evt-001"},
            ),
            score=0.1,
        ),
        SearchResult(
            chunk=TextChunk(
                id="evt-001::chunk-1",
                text="B",
                metadata={"event_uid": "evt-001"},
            ),
            score=0.2,
        ),
    ]

    deduplicated = deduplicate_by_event(results)

    assert [result.chunk.id for result in deduplicated] == ["evt-001::chunk-0"]


def test_matches_query_focus_rejects_candidate_without_explicit_theme() -> None:
    """Vérifie qu'un thème culturel explicite reste discriminant."""

    chunk = TextChunk(
        id="evt-001::chunk-0",
        text="Titre : Nuit europeenne des musees",
        metadata={"event_uid": "evt-001", "title": "Nuit europeenne des musees"},
    )

    assert not matches_query_focus(tokenize("jazz nuit des musees"), chunk)


def test_matches_query_focus_accepts_young_public_alternative() -> None:
    """Vérifie le cas jeune public/enfant."""

    chunk = TextChunk(
        id="evt-001::chunk-0",
        text="Spectacle familial pour enfant",
        metadata={"event_uid": "evt-001", "keywords": ["enfant"]},
    )

    assert matches_query_focus(tokenize("spectacle jeune public"), chunk)


def test_tokenize_normalizes_simple_plurals() -> None:
    """Vérifie la normalisation singulier/pluriel."""

    assert "exposition" in tokenize("Quelles expositions ?")
    assert "festival" in tokenize("Quels festivals ?")
