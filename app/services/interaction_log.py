"""Journal local des interactions utilisateur.

Ce module reprend l'idée du cours RAG : conserver les questions, les réponses,
les sources et les feedbacks pour pouvoir analyser l'usage du chatbot après une
démo. L'implémentation reste volontairement légère avec SQLite standard.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.config import settings


FeedbackScore = Literal["positive", "negative"]


def init_interaction_db(db_path: str | Path | None = None) -> None:
    """Crée la table SQLite des interactions si elle n'existe pas."""

    path = Path(db_path or settings.interaction_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                feedback_score TEXT,
                feedback_comment TEXT,
                feedback_timestamp_utc TEXT
            )
            """
        )
        connection.commit()


def log_interaction(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    parameters: dict[str, Any],
    db_path: str | Path | None = None,
) -> int:
    """Enregistre une question/réponse et retourne son identifiant."""

    path = Path(db_path or settings.interaction_db_path)
    init_interaction_db(path)
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO interactions (
                timestamp_utc,
                question,
                answer,
                sources_json,
                parameters_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                question,
                answer,
                json.dumps(sources, ensure_ascii=False),
                json.dumps(parameters, ensure_ascii=False),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def save_feedback(
    interaction_id: int,
    score: FeedbackScore,
    comment: str | None = None,
    db_path: str | Path | None = None,
) -> bool:
    """Ajoute ou remplace le feedback associé à une interaction."""

    path = Path(db_path or settings.interaction_db_path)
    init_interaction_db(path)
    cleaned_comment = comment.strip() if comment else None
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            UPDATE interactions
            SET feedback_score = ?,
                feedback_comment = ?,
                feedback_timestamp_utc = ?
            WHERE id = ?
            """,
            (
                score,
                cleaned_comment,
                datetime.now(UTC).isoformat(),
                interaction_id,
            ),
        )
        connection.commit()
        return cursor.rowcount > 0
