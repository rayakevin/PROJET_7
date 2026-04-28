"""Découpage des événements normalisés en chunks indexables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.config import settings


@dataclass(frozen=True, slots=True)
class TextChunk:
    """Segment de texte avec métadonnées conservées pour le retrieval."""

    id: str
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convertit le chunk en dictionnaire serialisable."""

        return asdict(self)


def validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    """Valide les paramètres de découpage."""

    if chunk_size <= 0:
        raise ValueError("chunk_size doit être strictement positif.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap doit être positif ou nul.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap doit être inférieur à chunk_size.")


def split_text(
    text: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[str]:
    """Découpe un texte en segments avec chevauchement."""

    validate_chunk_parameters(chunk_size, chunk_overlap)
    cleaned_text = " ".join(str(text or "").split())
    if not cleaned_text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        adjusted_end = end

        if end < text_length:
            last_space = cleaned_text.rfind(" ", start, end)
            if last_space > start:
                adjusted_end = last_space

        chunk = cleaned_text[start:adjusted_end].strip()
        if chunk:
            chunks.append(chunk)

        if adjusted_end >= text_length:
            break

        next_start = max(adjusted_end - chunk_overlap, start + 1)
        start = next_start

    return chunks


def chunk_event(
    event: dict[str, Any],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[TextChunk]:
    """Transforme un événement normalisé en chunks avec métadonnées."""

    event_uid = str(event.get("uid") or "")
    text_parts = split_text(
        str(event.get("full_text") or ""),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[TextChunk] = []
    for index, text in enumerate(text_parts):
        chunks.append(
            TextChunk(
                id=f"{event_uid}::chunk-{index}",
                text=text,
                metadata={
                    "event_uid": event_uid,
                    "chunk_index": index,
                    "title": event.get("title", ""),
                    "city": event.get("city", ""),
                    "location_name": event.get("location_name", ""),
                    "start": event.get("start", ""),
                    "end": event.get("end", ""),
                    "keywords": event.get("keywords", []),
                },
            )
        )

    return chunks


def chunk_events(
    events: list[dict[str, Any]],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[TextChunk]:
    """Découpe une liste d'événements normalisés."""

    chunks: list[TextChunk] = []
    for event in events:
        chunks.extend(
            chunk_event(
                event,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks
