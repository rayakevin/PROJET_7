"""Tests unitaires de génération de prompt RAG."""

from app.rag.answer import (
    ANSWER_PROMPT,
    OllamaAnswerGenerator,
    format_context,
    remove_thinking_block,
    save_prompt_trace,
    to_mistral_role,
    to_ollama_role,
)
from app.config import settings
from app.rag.chunking import TextChunk
from app.rag.vector_store import SearchResult
from app.utils.io import read_json


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
        model="qwen2.5:7b",
    )

    answer = generator.generate("Question ?", [], temperature=0.1, max_tokens=200)

    assert answer == "Réponse locale."
    assert captured_payload["url"] == "http://localhost:11434/api/chat"
    assert captured_payload["json"]["model"] == "qwen2.5:7b"
    assert captured_payload["json"]["options"]["temperature"] == 0.1
    assert captured_payload["json"]["options"]["num_predict"] == settings.ollama_min_tokens
    assert captured_payload["json"]["options"]["num_ctx"] == settings.ollama_num_ctx


def test_save_prompt_trace_writes_complete_prompt(tmp_path) -> None:
    """Vérifie la sauvegarde locale du prompt complet envoyé au modèle."""

    result = SearchResult(
        chunk=TextChunk(
            id="evt-001::chunk-0",
            text="Titre : Concert jazz",
            metadata={
                "event_uid": "evt-001",
                "title": "Concert jazz",
                "start": "2026-05-02T19:30:00Z",
            },
        ),
        score=0.42,
    )
    messages = ANSWER_PROMPT.format_messages(
        question="Quels concerts jazz ?",
        context=format_context([result]),
        current_date="2026-05-01",
    )

    output_path = save_prompt_trace(
        provider="mistral",
        model="mistral-small-latest",
        question="Quels concerts jazz ?",
        contexts=[result],
        messages=messages,
        temperature=0.2,
        max_tokens=600,
        role_converter=to_mistral_role,
        output_dir=tmp_path,
    )
    payload = read_json(output_path)

    assert payload["provider"] == "mistral"
    assert payload["model"] == "mistral-small-latest"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert "Date du jour : 2026-05-01" in payload["prompt_text"]
    assert "ne recommande jamais une source" in payload["prompt_text"]
    assert "Question utilisateur" in payload["prompt_text"]
    assert payload["sources"][0]["metadata"]["title"] == "Concert jazz"
