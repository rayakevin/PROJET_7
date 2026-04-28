"""Normalisation des événements culturels.

Ce module transforme des événements bruts en dictionnaires propres,
stables et faciles à indexer dans une base vectorielle.

Dans le pipeline RAG, cette étape sert de contrat entre :
- la récupération OpenAgenda ;
- la préparation du dataset ;
- le chunking ;
- l'indexation FAISS.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NormalizedEvent:
    """Représente un événement nettoyé et prêt pour l'indexation.

    Attributes
    ----------
    uid:
        Identifiant stable de l'événement.
    title:
        Titre lisible de l'événement.
    description:
        Description textuelle courte ou longue.
    location_name:
        Nom du lieu ou de la zone de l'événement.
    city:
        Ville associée à l'événement.
    start:
        Date de début au format texte ISO si disponible.
    end:
        Date de fin au format texte ISO si disponible.
    keywords:
        Liste de mots-clés nettoyés.
    full_text:
        Texte consolidé utilisé ensuite pour le RAG.
    """

    uid: str
    title: str
    description: str
    location_name: str
    city: str
    start: str
    end: str
    keywords: list[str]
    full_text: str


def clean_text(value: Any) -> str:
    """Nettoie une valeur textuelle issue d'une source brute.

    Parameters
    ----------
    value:
        Valeur potentiellement absente, non textuelle ou mal espacée.

    Returns
    -------
    str
        Texte sans espaces superflus. Une valeur absente devient une chaîne
        vide pour simplifier le reste du pipeline.
    """

    if value is None:
        return ""

    text = re.sub(r"<[^>]+>", " ", str(value))
    text = text.replace("&nbsp;", " ")

    # `split()` sans argument compacte tous les espaces, tabulations et retours ligne.
    return " ".join(text.split())


def normalize_keywords(value: Any) -> list[str]:
    """Transforme un champ de mots-clés en liste propre.

    Parameters
    ----------
    value:
        Liste, tuple, chaîne ou valeur absente contenant des mots-clés.

    Returns
    -------
    list[str]
        Mots-clés nettoyés, dédupliqués et non vides.
    """

    if value is None:
        return []

    if isinstance(value, str):
        raw_keywords = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_keywords = value
    else:
        raw_keywords = [value]

    keywords: list[str] = []
    seen: set[str] = set()

    for keyword in raw_keywords:
        cleaned_keyword = clean_text(keyword).lower()

        # On ignore les mots-clés vides et les doublons.
        if cleaned_keyword and cleaned_keyword not in seen:
            keywords.append(cleaned_keyword)
            seen.add(cleaned_keyword)

    return keywords


def localized_value(value: Any, language: str = "fr") -> Any:
    """Recupere une valeur monolingue depuis les champs OpenAgenda."""

    if isinstance(value, dict):
        if language in value:
            return value[language]
        if value:
            return next(iter(value.values()))
        return ""
    return value


def build_description(raw_event: dict[str, Any]) -> str:
    """Fusionne description courte et description longue si disponible."""

    short_description = clean_text(
        localized_value(raw_event.get("description") or raw_event.get("description_fr"))
    )
    long_description = clean_text(
        localized_value(
            raw_event.get("longDescription") or raw_event.get("longdescription_fr")
        )
    )

    if long_description and long_description != short_description:
        return clean_text(f"{short_description} {long_description}")

    return short_description


def extract_location_name(raw_event: dict[str, Any]) -> str:
    """Extrait le nom du lieu depuis les formats fixture ou OpenAgenda."""

    location = raw_event.get("location")
    if isinstance(location, dict):
        return clean_text(
            raw_event.get("location_name")
            or raw_event.get("locationName")
            or location.get("name")
        )

    return clean_text(raw_event.get("location_name") or raw_event.get("locationName"))


def extract_city(raw_event: dict[str, Any]) -> str:
    """Extrait la ville depuis les formats fixture ou OpenAgenda."""

    location = raw_event.get("location")
    if isinstance(location, dict):
        return clean_text(raw_event.get("city") or location.get("city"))

    return clean_text(raw_event.get("city") or raw_event.get("location_city"))


def extract_start_end(raw_event: dict[str, Any]) -> tuple[str, str]:
    """Extrait les dates de début et fin depuis les horaires OpenAgenda."""

    start = clean_text(
        raw_event.get("start")
        or raw_event.get("firstDate")
        or raw_event.get("firstdate_begin")
    )
    end = clean_text(
        raw_event.get("end")
        or raw_event.get("lastDate")
        or raw_event.get("lastdate_end")
        or raw_event.get("firstdate_end")
    )

    timings = raw_event.get("timings")
    if isinstance(timings, list) and timings:
        first_timing = timings[0]
        last_timing = timings[-1]
        if isinstance(first_timing, dict):
            start = clean_text(first_timing.get("start") or first_timing.get("begin"))
        if isinstance(last_timing, dict):
            end = clean_text(last_timing.get("end") or last_timing.get("finish"))

    return start, end


def build_full_text(event: NormalizedEvent) -> str:
    """Construit le texte documentaire qui sera indexé.

    Parameters
    ----------
    event:
        Evênement dejà normalisé, sans son champ `full_text` définitif.

    Returns
    -------
    str
        Texte consolidé regroupant les informations utiles pour la recherche.
    """

    parts = [
        f"Titre : {event.title}",
        f"Mots-clés : {', '.join(event.keywords)}",
        f"Ville : {event.city}",
        f"Lieu : {event.location_name}",
        f"Début : {event.start}",
        f"Fin : {event.end}",
        f"Description : {event.description}",
    ]

    # Les segments vides sont retirés pour éviter de polluer les embeddings.
    return "\n".join(part for part in parts if not part.endswith(": "))


def normalize_event(raw_event: dict[str, Any]) -> dict[str, Any]:
    """Normalise un événement brut.

    Parameters
    ----------
    raw_event:
        Événement brut provenant d'une fixture ou, plus tard, d'OpenAgenda.

    Returns
    -------
    dict[str, Any]
        Événement nettoyé sous forme de dictionnaire sérialisable en JSON.
    """

    start, end = extract_start_end(raw_event)

    event_without_full_text = NormalizedEvent(
        uid=clean_text(raw_event.get("uid")),
        title=clean_text(localized_value(raw_event.get("title") or raw_event.get("title_fr"))),
        description=build_description(raw_event),
        location_name=extract_location_name(raw_event),
        city=extract_city(raw_event),
        start=start,
        end=end,
        keywords=normalize_keywords(
            localized_value(raw_event.get("keywords") or raw_event.get("keywords_fr"))
        ),
        full_text="",
    )

    normalized_event = NormalizedEvent(
        **{
            **asdict(event_without_full_text),
            "full_text": build_full_text(event_without_full_text),
        }
    )

    return asdict(normalized_event)


def normalize_events(raw_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise une liste d'événements bruts.

    Parameters
    ----------
    raw_events:
        Liste d'événements au format source.

    Returns
    -------
    list[dict[str, Any]]
        Liste d'événements propres et prêts pour les étapes suivantes.
    """

    return [normalize_event(raw_event) for raw_event in raw_events]
