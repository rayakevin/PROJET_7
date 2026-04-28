"""Test fonctionnel manuel de l'API locale."""

from __future__ import annotations

import os

import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def main() -> None:
    """Appelle /health puis /ask sur l'API locale."""

    health_response = requests.get(f"{API_BASE_URL}/health", timeout=30)
    health_response.raise_for_status()
    print("health:", health_response.json())

    ask_response = requests.post(
        f"{API_BASE_URL}/ask",
        json={"question": "Quels concerts de jazz sont disponibles a Paris ?"},
        timeout=120,
    )
    ask_response.raise_for_status()
    print("ask:", ask_response.json())


if __name__ == "__main__":
    main()
