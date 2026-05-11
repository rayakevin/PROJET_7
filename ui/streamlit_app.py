"""Interface Streamlit locale pour interroger l'API RAG."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEMO_QUESTIONS = [
    "Je cherche une masterclass ou un concert de jazz avec Mark PRIORE à Paris, que peux-tu me recommander ?",
    "Je cherche une nocturne gratuite à la Cité des sciences pour la Nuit européenne des musées 2026, que proposes-tu ?",
    "Quel événement cosplay avec Sikay, Corneline et Edes a eu lieu à la Cité des sciences ?",
    "Je cherche un événement autour de la Fête de la musique avec Le SPB descend dans la rue, où et quand a-t-il lieu ?",
    "Quelle avant-première du film RED BIRD a eu lieu à Paris ?",
]


def main() -> None:
    """Lance l'interface Streamlit."""

    st.set_page_config(
        page_title="Puls-Events RAG",
        page_icon=":material/theater_comedy:",
        layout="wide",
    )
    st.title("Assistant culturel Puls-Events")
    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    with st.sidebar:
        st.header("API")
        api_base_url = st.text_input("URL de l'API", DEFAULT_API_BASE_URL)
        if st.button("Tester /health", use_container_width=True):
            show_health(api_base_url)

        st.header("Retrieval")
        top_k = st.slider("Nombre de sources", min_value=1, max_value=10, value=3)
        retrieval_max_score = st.slider(
            "Distance FAISS maximale",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.01,
            help=(
                "FAISS retourne une distance, pas un score de similarité : "
                "plus la valeur est basse, plus le chunk est proche de la question. "
                "Le seuil garde les sources dont la distance est inférieure ou égale "
                "à cette valeur."
            ),
        )
        st.header("Génération")
        llm_provider_label = st.selectbox(
            "Fournisseur LLM",
            options=[
                "Mistral API",
                "Ollama local",
                "Auto : Ollama puis Mistral",
            ],
            index=2,
            help=(
                "Le mode Ollama utilise les ressources locales de la machine. "
                "Le mode auto tente Ollama, puis bascule sur Mistral si l'appel échoue."
            ),
        )
        llm_provider = {
            "Mistral API": "mistral",
            "Ollama local": "ollama",
            "Auto : Ollama puis Mistral": "auto",
        }[llm_provider_label]
        default_llm_model = "qwen2.5:7b" if llm_provider in {"ollama", "auto"} else ""
        llm_model = st.text_input(
            "Modèle LLM optionnel",
            value=default_llm_model,
            help=(
                "Exemples : mistral-small-latest côté Mistral, qwen2.5:7b côté Ollama. "
                "Laisser vide pour utiliser la configuration serveur."
            ),
        )
        temperature = st.slider(
            "Température",
            min_value=0.0,
            max_value=1.5,
            value=0.2,
            step=0.05,
        )
        max_tokens = st.slider(
            "Tokens maximum",
            min_value=100,
            max_value=2000,
            value=600,
            step=50,
        )

    selected_question = st.selectbox("Scénario de démo", DEMO_QUESTIONS)
    question = st.text_area("Question utilisateur", value=selected_question, height=110)

    if st.button("Interroger le RAG", type="primary"):
        payload = {
            "question": question,
            "top_k": top_k,
            "retrieval_max_score": retrieval_max_score,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "llm_provider": llm_provider,
            "llm_model": llm_model.strip() or None,
        }
        ask_api(api_base_url, payload)

    if st.session_state.last_response:
        display_response(st.session_state.last_response)
        show_feedback_form(api_base_url, st.session_state.last_response)


def show_health(api_base_url: str) -> None:
    """Affiche le statut de l'API."""

    try:
        response = requests.get(f"{api_base_url.rstrip('/')}/health", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"API indisponible : {exc}")
        return

    st.success("API disponible")
    st.json(response.json())


def ask_api(api_base_url: str, payload: dict[str, Any]) -> None:
    """Appelle /ask et mémorise la réponse."""

    if not payload["question"].strip():
        st.warning("La question ne peut pas être vide.")
        return

    try:
        with st.spinner("Recherche et génération en cours..."):
            response = requests.post(
                f"{api_base_url.rstrip('/')}/ask",
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Erreur API : {exc}")
        if getattr(exc, "response", None) is not None:
            st.code(exc.response.text)
        return

    st.session_state.last_response = response.json()


def display_response(data: dict[str, Any]) -> None:
    """Affiche la dernière réponse RAG et ses sources."""

    st.subheader("Réponse")
    st.markdown(data["answer"])

    st.subheader("Sources")
    for index, source in enumerate(data.get("sources", []), start=1):
        with st.expander(f"{index}. {source['title']}"):
            st.write(f"Lieu : {source['location_name']} ({source['city']})")
            st.write(f"Début : {source['start']}")
            st.write(f"Fin : {source['end']}")
            st.write(
                f"Distance FAISS : {source['score']:.4f} "
                "(plus bas = plus proche)"
            )
            st.write(f"UID événement : {source['event_uid']}")

    st.subheader("Paramètres appliqués")
    st.json(data.get("parameters", {}))


def show_feedback_form(api_base_url: str, data: dict[str, Any]) -> None:
    """Affiche un feedback simple inspiré du cours RAG."""

    interaction_id = data.get("interaction_id")
    if not interaction_id:
        return

    st.subheader("Feedback")
    st.caption(
        "Ce retour est stocké localement pour analyser les réponses après la démo."
    )
    comment = st.text_input(
        "Commentaire optionnel",
        key=f"feedback_comment_{interaction_id}",
        placeholder="Exemple : bonne réponse, source manquante, date imprécise...",
    )
    col_positive, col_negative = st.columns(2)
    with col_positive:
        if st.button("Réponse utile", use_container_width=True):
            submit_feedback(api_base_url, interaction_id, "positive", comment)
    with col_negative:
        if st.button("Réponse à revoir", use_container_width=True):
            submit_feedback(api_base_url, interaction_id, "negative", comment)


def submit_feedback(
    api_base_url: str,
    interaction_id: int,
    score: str,
    comment: str | None,
) -> None:
    """Envoie le feedback à l'API."""

    try:
        response = requests.post(
            f"{api_base_url.rstrip('/')}/feedback",
            json={
                "interaction_id": interaction_id,
                "score": score,
                "comment": comment or None,
            },
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Feedback non enregistré : {exc}")
        return
    st.success("Feedback enregistré.")


if __name__ == "__main__":
    main()
