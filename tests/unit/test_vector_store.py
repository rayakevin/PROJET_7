"""Tests unitaires du vector store FAISS simplifié."""

from langchain_core.documents import Document

from app.rag.chunking import TextChunk
from app.rag.vector_store import FaissVectorStore, SearchResult, deduplicate_by_event


class FakeFaissStore:
    """Store FAISS minimal pour tester sans index réel."""

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        """Retourne deux documents avec distances contrôlées."""

        assert query == "concert jazz"
        assert k == 3
        return [
            (
                Document(
                    page_content="Document proche",
                    metadata={
                        "chunk_id": "evt-001::chunk-0",
                        "event_uid": "evt-001",
                        "title": "Concert jazz",
                    },
                ),
                0.2,
            ),
            (
                Document(
                    page_content="Document plus éloigné",
                    metadata={
                        "chunk_id": "evt-002::chunk-0",
                        "event_uid": "evt-002",
                        "title": "Exposition",
                    },
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
    """Vérifie le fallback simple pour éviter un contexte vide."""

    vector_store = FaissVectorStore(store=FakeFaissStore(), chunks=[])

    results = vector_store.search(
        "concert jazz",
        top_k=3,
        max_score=0.1,
    )

    assert len(results) == 1
    assert results[0].chunk.id == "evt-001::chunk-0"


def test_search_removes_langchain_chunk_id_from_metadata() -> None:
    """Vérifie que le chunk_id technique ne pollue pas les métadonnées."""

    vector_store = FaissVectorStore(store=FakeFaissStore(), chunks=[])

    results = vector_store.search("concert jazz", top_k=3, max_score=None)

    assert results[0].chunk.id == "evt-001::chunk-0"
    assert "chunk_id" not in results[0].chunk.metadata


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
