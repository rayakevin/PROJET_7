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

# Les poids ci-dessous rendent le reranking explicite :
# un mot trouvé dans le titre est plus important qu'un mot trouvé seulement
# dans le texte complet, car le titre résume souvent le thème de l'événement.
TITLE_TOKEN_WEIGHT = 4.0
KEYWORD_TOKEN_WEIGHT = 3.0
LOCATION_TOKEN_WEIGHT = 1.5
TEXT_TOKEN_WEIGHT = 1.0
FOCUS_MISMATCH_PENALTY = 6.0
IDF_POWER = 1.5

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
    """Résultat de recherche.

    Le champ `score` correspond à la distance FAISS retournée par LangChain :
    plus elle est basse, plus le chunk est proche de la question.
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
        """Vectorise une requête utilisateur."""

        return self.embedding_model.embed_query(text)


class FaissVectorStore:
    """Index FAISS LangChain et métadonnées associées aux chunks."""

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
        """Recharge un index FAISS LangChain sauvegardé."""

        if embedding_model is None:
            from app.rag.embeddings import MistralEmbeddingModel

            embedding_model = MistralEmbeddingModel()

        source_dir = Path(directory)
        # LangChain sauvegarde une partie du vector store en pickle.
        # Ici le fichier vient du dossier local du projet, reconstruit par nos
        # scripts ; on autorise donc explicitement cette désérialisation.
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
        """Recherche les chunks les plus proches d'une requête.

        `max_score` est un seuil de distance FAISS, pas une similarité.
        Une valeur plus basse est donc plus stricte.
        """

        if top_k <= 0:
            return []

        documents_with_scores = self.store.similarity_search_with_score(
            query,
            k=top_k,
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

        # La recherche vectorielle retrouve les textes sémantiquement proches.
        # La recherche lexicale rattrape les cas où un titre, un lieu ou un mot-clé
        # exact est important pour la question.
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
        """Ajoute des candidats par correspondance titre, mots-clés et texte."""

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

        # Les candidats lexicaux n'ont pas de vraie distance FAISS, car ils ne
        # viennent pas directement de `similarity_search_with_score`. On leur
        # donne une distance de repli compatible avec le filtre final.
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
    """Trie les résultats avec un score hybride vectoriel + lexical."""

    return sorted(
        results,
        key=lambda result: hybrid_rank_score(query_tokens, token_weights, result),
        reverse=True,
    )


def deduplicate_by_event(results: list[SearchResult]) -> list[SearchResult]:
    """Garde le meilleur chunk par événement pour diversifier le contexte."""

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
    """Score de tri combinant proximité FAISS et correspondance lexicale.

    Ce score ne remplace pas la distance FAISS exposée dans l'API. Il sert
    uniquement à ordonner les candidats avant de renvoyer les sources.
    """

    lexical_score = lexical_relevance_score(
        query_tokens,
        token_weights,
        result.chunk.text,
        result.chunk.metadata,
    )
    vector_score = 1 / (1 + max(result.score, 0.0))
    focus_penalty = (
        0.0
        if matches_query_focus(query_tokens, result.chunk)
        else FOCUS_MISMATCH_PENALTY
    )
    return vector_score + lexical_score - focus_penalty


def lexical_relevance_score(
    query_tokens: list[str],
    token_weights: dict[str, float],
    text: str,
    metadata: dict[str, Any],
) -> float:
    """Mesure la couverture lexicale pondérée d'un candidat."""

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
            score += TITLE_TOKEN_WEIGHT * token_weight
        if token in keyword_tokens:
            score += KEYWORD_TOKEN_WEIGHT * token_weight
        if token in location_tokens:
            score += LOCATION_TOKEN_WEIGHT * token_weight
        if token in text_tokens:
            score += TEXT_TOKEN_WEIGHT * token_weight

    total_weight = sum(token_weights.get(token, 1.0) for token in query_tokens)
    return score / total_weight if total_weight else 0.0


def compute_query_token_weights(
    query_tokens: list[str],
    chunks: list[TextChunk],
) -> dict[str, float]:
    """Pondère les tokens de requête selon leur rareté dans le corpus.

    C'est une forme simple d'IDF : un mot rare dans les chunks compte plus
    qu'un mot très fréquent.
    """

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
        ** IDF_POWER
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
    """Vérifie que le candidat respecte les thèmes explicites de la question.

    Cette petite couche métier évite par exemple qu'une question sur le jazz
    récupère un événement culturel générique seulement parce que la description
    contient des mots proches.
    """

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
