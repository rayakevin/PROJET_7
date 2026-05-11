"""Stockage vectoriel FAISS via LangChain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.rag.chunking import TextChunk
from app.rag.embeddings import EmbeddingModel
from app.utils.io import read_json, write_json


INDEX_NAME = "index"
CHUNKS_FILENAME = "chunks.json"


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Résultat de recherche dans FAISS.

    Le score retourné par LangChain est une distance : plus il est bas, plus le
    chunk est proche de la question.
    """

    chunk: TextChunk
    score: float


class LangChainEmbeddingAdapter(Embeddings):
    """Adaptateur entre notre modèle d'embeddings et LangChain."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        """Stocke le modèle d'embeddings utilisé par LangChain."""

        self.embedding_model = embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectorise une liste de documents."""

        return self.embedding_model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Vectorise une question utilisateur."""

        return self.embedding_model.embed_query(text)


class FaissVectorStore:
    """Index FAISS LangChain et chunks associés."""

    def __init__(self, store: FAISS, chunks: list[TextChunk]) -> None:
        """Associe l'index FAISS aux chunks sauvegardés localement."""

        self.store = store
        self.chunks = chunks

    @classmethod
    def from_chunks(
        cls,
        chunks: list[TextChunk],
        embedding_model: EmbeddingModel,
    ) -> "FaissVectorStore":
        """Construit un index FAISS LangChain à partir de chunks."""

        if not chunks:
            raise ValueError("Impossible de construire un index sans chunks.")

        store = FAISS.from_texts(
            texts=[chunk.text for chunk in chunks],
            embedding=LangChainEmbeddingAdapter(embedding_model),
            metadatas=[chunk.metadata | {"chunk_id": chunk.id} for chunk in chunks],
            ids=[chunk.id for chunk in chunks],
            normalize_L2=True,
        )
        return cls(store=store, chunks=chunks)

    def save(self, directory: str | Path = settings.vector_store_dir) -> Path:
        """Sauvegarde l'index FAISS et les chunks."""

        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(target_dir), index_name=INDEX_NAME)
        write_json([chunk.to_dict() for chunk in self.chunks], target_dir / CHUNKS_FILENAME)
        return target_dir

    @classmethod
    def load(
        cls,
        embedding_model: EmbeddingModel,
        directory: str | Path = settings.vector_store_dir,
    ) -> "FaissVectorStore":
        """Recharge un index FAISS sauvegardé localement."""

        source_dir = Path(directory)
        store = FAISS.load_local(
            str(source_dir),
            embeddings=LangChainEmbeddingAdapter(embedding_model),
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True,
        )
        chunk_rows = read_json(source_dir / CHUNKS_FILENAME)
        chunks = [
            TextChunk(
                id=row["id"],
                text=row["text"],
                metadata=row["metadata"],
            )
            for row in chunk_rows
        ]
        return cls(store=store, chunks=chunks)

    def search(
        self,
        query: str,
        top_k: int = settings.top_k,
        max_score: float | None = settings.retrieval_max_score,
    ) -> list[SearchResult]:
        """Recherche les chunks les plus proches d'une question."""

        if top_k <= 0:
            return []

        documents_with_scores = self.store.similarity_search_with_score(
            query,
            k=top_k,
        )
        results = [
            SearchResult(
                chunk=TextChunk(
                    id=str(document.metadata.get("chunk_id", "")),
                    text=document.page_content,
                    metadata={
                        key: value
                        for key, value in dict(document.metadata).items()
                        if key != "chunk_id"
                    },
                ),
                score=float(score),
            )
            for document, score in documents_with_scores
        ]
        results = deduplicate_by_event(results)

        if max_score is None:
            return results[:top_k]

        filtered_results = [result for result in results if result.score <= max_score]
        return (filtered_results or results[:1])[:top_k]


def deduplicate_by_event(results: list[SearchResult]) -> list[SearchResult]:
    """Garde le meilleur chunk par événement."""

    deduplicated: list[SearchResult] = []
    seen_events: set[str] = set()
    for result in results:
        event_uid = str(result.chunk.metadata.get("event_uid") or result.chunk.id)
        if event_uid in seen_events:
            continue
        deduplicated.append(result)
        seen_events.add(event_uid)
    return deduplicated


def build_vector_store(
    chunks: list[TextChunk],
    embedding_model: EmbeddingModel,
    output_dir: str | Path = settings.vector_store_dir,
) -> Path:
    """Construit et sauvegarde un vector store FAISS LangChain."""

    vector_store = FaissVectorStore.from_chunks(chunks, embedding_model)
    return vector_store.save(output_dir)
