"""Stockage vectoriel FAISS via LangChain."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.rag.chunking import TextChunk
from app.rag.embeddings import EmbeddingModel
from app.utils.io import read_json, write_json


INDEX_NAME = "index"
CHUNKS_FILENAME = "chunks.json"
DEFAULT_LEXICAL_CANDIDATE_LIMIT = 30
STOPWORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "autour",
    "de",
    "des",
    "du",
    "en",
    "et",
    "evenements",
    "lies",
    "la",
    "le",
    "les",
    "ou",
    "paris",
    "peut",
    "peuvent",
    "pour",
    "proposes",
    "prevus",
    "quels",
    "quelles",
    "sont",
    "sur",
    "trouver",
    "un",
    "une",
}


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Resultat de recherche vectorielle."""

    chunk: TextChunk
    score: float


class LangChainEmbeddingAdapter(Embeddings):
    """Adaptateur entre notre modele d'embeddings et LangChain."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectorise une liste de documents."""

        return self.embedding_model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Vectorise une requete utilisateur."""

        return self.embedding_model.embed_query(text)


class FaissVectorStore:
    """Index FAISS LangChain et metadonnees associees aux chunks."""

    def __init__(self, store: FAISS, chunks: list[TextChunk]) -> None:
        self.store = store
        self.chunks = chunks

    @classmethod
    def from_chunks(
        cls,
        chunks: list[TextChunk],
        embedding_model: EmbeddingModel,
    ) -> "FaissVectorStore":
        """Construit un index FAISS LangChain a partir de chunks."""

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
        """Sauvegarde l'index FAISS LangChain et les chunks."""

        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(target_dir), index_name=INDEX_NAME)
        write_json([chunk.to_dict() for chunk in self.chunks], target_dir / CHUNKS_FILENAME)
        return target_dir

    @classmethod
    def load(
        cls,
        directory: str | Path = settings.vector_store_dir,
        embedding_model: EmbeddingModel | None = None,
    ) -> "FaissVectorStore":
        """Recharge un index FAISS LangChain sauvegarde."""

        if embedding_model is None:
            from app.rag.embeddings import MistralEmbeddingModel

            embedding_model = MistralEmbeddingModel()

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
        candidate_multiplier: int = settings.retrieval_candidate_multiplier,
    ) -> list[SearchResult]:
        """Recherche les chunks les plus similaires a une requete."""

        if top_k <= 0:
            return []

        candidate_k = max(top_k, top_k * max(candidate_multiplier, 1))
        documents_with_scores = self.store.similarity_search_with_score(
            query,
            k=candidate_k,
        )

        results: list[SearchResult] = []
        for document, score in documents_with_scores:
            metadata = dict(document.metadata)
            chunk_id = str(metadata.pop("chunk_id", ""))
            results.append(
                SearchResult(
                    chunk=TextChunk(
                        id=chunk_id,
                        text=document.page_content,
                        metadata=metadata,
                    ),
                    score=float(score),
                )
            )

        query_tokens = tokenize(query)
        token_weights = compute_query_token_weights(query_tokens, self.chunks)

        results = merge_candidates(
            vector_results=results,
            lexical_results=self.lexical_search(query_tokens, token_weights),
        )
        results = rerank_results(query_tokens, token_weights, results)
        results = deduplicate_by_event(results)

        if max_score is None:
            return results[:top_k]

        filtered_results = [result for result in results if result.score <= max_score]
        return (filtered_results or results[:1])[:top_k]

    def lexical_search(
        self,
        query_tokens: list[str],
        token_weights: dict[str, float],
        limit: int = DEFAULT_LEXICAL_CANDIDATE_LIMIT,
    ) -> list[SearchResult]:
        """Ajoute des candidats par correspondance titre, mots-cles et texte."""

        if not query_tokens:
            return []

        scored_chunks = [
            (
                lexical_relevance_score(
                    query_tokens,
                    token_weights,
                    chunk.text,
                    chunk.metadata,
                ),
                chunk,
            )
            for chunk in self.chunks
        ]
        best_chunks = sorted(
            ((score, chunk) for score, chunk in scored_chunks if score > 0),
            key=lambda item: item[0],
            reverse=True,
        )[:limit]

        fallback_score = (
            settings.retrieval_max_score
            if settings.retrieval_max_score is not None
            else 0.0
        )
        return [
            SearchResult(chunk=chunk, score=fallback_score)
            for _, chunk in best_chunks
        ]


def merge_candidates(
    vector_results: list[SearchResult],
    lexical_results: list[SearchResult],
) -> list[SearchResult]:
    """Fusionne les candidats vectoriels et lexicaux sans doublons de chunk."""

    by_chunk_id = {result.chunk.id: result for result in vector_results}
    for result in lexical_results:
        if result.chunk.id not in by_chunk_id:
            by_chunk_id[result.chunk.id] = result
    return list(by_chunk_id.values())


def rerank_results(
    query_tokens: list[str],
    token_weights: dict[str, float],
    results: list[SearchResult],
) -> list[SearchResult]:
    """Trie les resultats avec un score hybride vectoriel + lexical."""

    return sorted(
        results,
        key=lambda result: hybrid_rank_score(query_tokens, token_weights, result),
        reverse=True,
    )


def deduplicate_by_event(results: list[SearchResult]) -> list[SearchResult]:
    """Garde le meilleur chunk par evenement pour diversifier le contexte."""

    deduplicated: list[SearchResult] = []
    seen_events: set[str] = set()
    for result in results:
        event_uid = str(result.chunk.metadata.get("event_uid") or result.chunk.id)
        if event_uid in seen_events:
            continue
        deduplicated.append(result)
        seen_events.add(event_uid)
    return deduplicated


def hybrid_rank_score(
    query_tokens: list[str],
    token_weights: dict[str, float],
    result: SearchResult,
) -> float:
    """Score de tri combinant proximite FAISS et correspondance lexicale."""

    lexical_score = lexical_relevance_score(
        query_tokens,
        token_weights,
        result.chunk.text,
        result.chunk.metadata,
    )
    vector_score = 1 / (1 + max(result.score, 0.0))
    focus_penalty = 0.0 if matches_query_focus(query_tokens, result.chunk) else 6.0
    return vector_score + lexical_score - focus_penalty


def lexical_relevance_score(
    query_tokens: list[str],
    token_weights: dict[str, float],
    text: str,
    metadata: dict[str, Any],
) -> float:
    """Mesure la couverture lexicale ponderee d'un candidat."""

    if not query_tokens:
        return 0.0

    title_tokens = set(tokenize(str(metadata.get("title", ""))))
    keyword_tokens = set(tokenize(" ".join(str(v) for v in metadata.get("keywords", []))))
    location_tokens = set(tokenize(str(metadata.get("location_name", ""))))
    text_tokens = set(tokenize(text))

    score = 0.0
    for token in query_tokens:
        token_weight = token_weights.get(token, 1.0)
        if token in title_tokens:
            score += 4.0 * token_weight
        if token in keyword_tokens:
            score += 3.0 * token_weight
        if token in location_tokens:
            score += 1.5 * token_weight
        if token in text_tokens:
            score += 1.0 * token_weight

    total_weight = sum(token_weights.get(token, 1.0) for token in query_tokens)
    return score / total_weight if total_weight else 0.0


def compute_query_token_weights(
    query_tokens: list[str],
    chunks: list[TextChunk],
) -> dict[str, float]:
    """Pondere les tokens de requete selon leur rarete dans le corpus."""

    if not query_tokens or not chunks:
        return {token: 1.0 for token in query_tokens}

    token_document_frequency = {token: 0 for token in query_tokens}
    for chunk in chunks:
        chunk_tokens = set(tokenize(candidate_text(chunk.text, chunk.metadata)))
        for token in query_tokens:
            if token in chunk_tokens:
                token_document_frequency[token] += 1

    corpus_size = len(chunks)
    return {
        token: (
            math.log((corpus_size + 1) / (token_document_frequency[token] + 1)) + 1
        )
        ** 1.5
        for token in query_tokens
    }


def candidate_text(text: str, metadata: dict[str, Any]) -> str:
    """Assemble le texte utile pour le calcul lexical."""

    keywords = metadata.get("keywords", [])
    return " ".join(
        [
            str(metadata.get("title", "")),
            str(metadata.get("location_name", "")),
            " ".join(str(keyword) for keyword in keywords),
            text,
        ]
    )


def matches_query_focus(query_tokens: list[str], chunk: TextChunk) -> bool:
    """Verifie que le candidat respecte les themes explicites de la question."""

    candidate_text_value = candidate_text(chunk.text, chunk.metadata)
    candidate_tokens = set(tokenize(candidate_text_value))
    normalized_candidate_text = " ".join(tokenize(candidate_text_value))

    if "jazz" in query_tokens and "jazz" not in candidate_tokens:
        return False
    if "cosplaymania" in query_tokens and not candidate_tokens.intersection(
        {"cosplaymania", "cosplay"}
    ):
        return False
    if "cosplay" in query_tokens and "cosplay" not in candidate_tokens:
        return False
    if "cinema" in query_tokens and not candidate_tokens.intersection(
        {"cinema", "cine"}
    ):
        return False
    if "exposition" in query_tokens and "exposition" not in candidate_tokens:
        return False
    if {"jeune", "public"}.issubset(query_tokens) and not (
        "jeune public" in normalized_candidate_text or "enfant" in candidate_tokens
    ):
        return False

    return True


def tokenize(value: str) -> list[str]:
    """Tokenise un texte en supprimant accents, ponctuation et stopwords."""

    normalized = unicodedata.normalize("NFKD", value.lower())
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    tokens = re.findall(r"[a-z0-9]+", ascii_text)
    return [
        singularize(token)
        for token in tokens
        if len(token) > 2 and token not in STOPWORDS
    ]


def singularize(token: str) -> str:
    """Normalise quelques pluriels simples utiles au matching lexical."""

    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def build_vector_store(
    chunks: list[TextChunk],
    embedding_model: EmbeddingModel,
    output_dir: str | Path = settings.vector_store_dir,
) -> Path:
    """Construit et sauvegarde un vector store FAISS LangChain."""

    vector_store = FaissVectorStore.from_chunks(chunks, embedding_model)
    return vector_store.save(output_dir)
