"""Tests unitaires de génération de prompt RAG."""

from app.rag.answer import (
    OllamaAnswerGenerator,
    format_context,
    remove_thinking_block,
    to_mistral_role,
    to_ollama_role,
)
from app.config import settings
from app.rag.chunking import TextChunk
from app.rag.vector_store import SearchResult


def test_format_context_includes_event_metadata() -> None:
    """Vérifie que le contexte injecté au LLM contient les informations métier."""

    result = SearchResult(
        chunk=TextChunk(
            id="evt-001::chunk-0",
            text="Titre : Concert jazz\nDescription : Concert gratuit.",
            metadata={
                "title": "Concert jazz",
                "location_name": "Parc central",
                "city": "Paris",
                "start": "2026-05-02T19:30:00Z",
                "end": "2026-05-02T22:00:00Z",
            },
        ),
        score=0.42,
    )

    context = format_context([result])

    assert "Concert jazz" in context
    assert "Parc central" in context
    assert "2026-05-02T19:30:00Z" in context
    assert "Distance FAISS : 0.4200" in context


def test_to_mistral_role_converts_human_to_user() -> None:
    """Vérifie la compatibilité des rôles LangChain avec Mistral."""

    assert to_mistral_role("human") == "user"
    assert to_mistral_role("system") == "system"


def test_to_ollama_role_converts_human_to_user() -> None:
    """Vérifie la compatibilité des rôles LangChain avec Ollama."""

    assert to_ollama_role("human") == "user"
    assert to_ollama_role("system") == "system"


def test_remove_thinking_block_removes_qwen_reasoning() -> None:
    """Vérifie le nettoyage des blocs de raisonnement éventuels."""

    content = "<think>analyse interne</think>\nVoici la réponse."

    assert remove_thinking_block(content) == "Voici la réponse."


def test_ollama_answer_generator_calls_local_chat_api(monkeypatch) -> None:
    """Vérifie l'appel Ollama sans démarrer de serveur local."""

    captured_payload = {}

    class FakeResponse:
        """Réponse HTTP factice du serveur Ollama."""

        def raise_for_status(self) -> None:
            """Simule une réponse HTTP valide."""

        def json(self) -> dict:
            """Retourne une réponse de chat Ollama minimale."""

            return {"message": {"content": "Réponse locale."}}

    def fake_post(url, json, timeout):  # noqa: ANN001
        """Capture l'appel HTTP envoyé à Ollama."""

        captured_payload["url"] = url
        captured_payload["json"] = json
        captured_payload["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("app.rag.answer.requests.post", fake_post)
    generator = OllamaAnswerGenerator(
        base_url="http://localhost:11434",
        model="qwen3:30b",
    )

    answer = generator.generate("Question ?", [], temperature=0.1, max_tokens=200)

    assert answer == "Réponse locale."
    assert captured_payload["url"] == "http://localhost:11434/api/chat"
    assert captured_payload["json"]["model"] == "qwen3:30b"
    assert captured_payload["json"]["options"]["temperature"] == 0.1
    assert captured_payload["json"]["options"]["num_predict"] == settings.ollama_min_tokens
