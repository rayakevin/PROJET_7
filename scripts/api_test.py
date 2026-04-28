"""Test fonctionnel manuel de l'API locale."""

from __future__ import annotations

import json
import os

import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEMO_QUESTION = os.getenv(
    "DEMO_QUESTION",
    "Je cherche un concert de Gospel Jazz pour la Fête de la musique à Paris, "
    "que peux-tu me proposer ?",
)


def main() -> None:
    """Appelle /health puis /ask sur l'API locale."""

    health_response = requests.get(f"{API_BASE_URL}/health", timeout=30)
    health_response.raise_for_status()
    print_json("health", health_response.json())

    ask_response = requests.post(
        f"{API_BASE_URL}/ask",
        json={
            "question": DEMO_QUESTION,
            "top_k": 3,
            "retrieval_max_score": 0.45,
            "temperature": 0.2,
            "max_tokens": 600,
        },
        timeout=120,
    )
    ask_response.raise_for_status()
    print_json("ask", ask_response.json())


def print_json(label: str, payload: dict) -> None:
    """Affiche une réponse JSON de façon compatible avec les consoles Windows."""

    print(f"{label}: {json.dumps(payload, ensure_ascii=True, indent=2)}")


if __name__ == "__main__":
    main()
