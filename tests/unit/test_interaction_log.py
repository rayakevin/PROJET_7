"""Tests du journal local des interactions."""

from __future__ import annotations

import sqlite3

from app.services.interaction_log import log_interaction, save_feedback


def test_log_interaction_creates_sqlite_row(tmp_path) -> None:
    """Vérifie qu'une question/réponse est enregistrée en SQLite."""

    db_path = tmp_path / "interactions.db"

    interaction_id = log_interaction(
        question="Quels concerts jazz ?",
        answer="Voici un concert.",
        sources=[{"title": "Concert jazz"}],
        parameters={"top_k": 3},
        db_path=db_path,
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT question, answer, sources_json, parameters_json FROM interactions"
        ).fetchone()

    assert interaction_id == 1
    assert row[0] == "Quels concerts jazz ?"
    assert row[1] == "Voici un concert."
    assert "Concert jazz" in row[2]
    assert '"top_k": 3' in row[3]


def test_save_feedback_updates_existing_interaction(tmp_path) -> None:
    """Vérifie la mise à jour d'un feedback utilisateur."""

    db_path = tmp_path / "interactions.db"
    interaction_id = log_interaction(
        question="Question",
        answer="Réponse",
        sources=[],
        parameters={},
        db_path=db_path,
    )

    saved = save_feedback(
        interaction_id=interaction_id,
        score="positive",
        comment="Utile",
        db_path=db_path,
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT feedback_score, feedback_comment FROM interactions WHERE id = ?",
            (interaction_id,),
        ).fetchone()

    assert saved is True
    assert row == ("positive", "Utile")


def test_save_feedback_returns_false_for_missing_interaction(tmp_path) -> None:
    """Vérifie qu'un feedback sur une interaction absente est refusé."""

    saved = save_feedback(
        interaction_id=999,
        score="negative",
        db_path=tmp_path / "interactions.db",
    )

    assert saved is False

