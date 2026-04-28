"""Interface Streamlit locale pour interroger l'API RAG."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEMO_QUESTIONS = [
    "Je cherche un concert de Gospel Jazz pour la Fête de la musique à Paris, que peux-tu me proposer ?",
    "Je veux faire une activité cosplay à la Cité des sciences et de l'Industrie, quels événements existent ?",
    "Je cherche un spectacle jeune public dès 3 ans à Paris, as-tu une recommandation ?",
    "Je voudrais voir une exposition autour de l'art japonais à Paris, que proposes-tu ?",
    "Y a-t-il une avant-première du film RED BIRD à Paris ?",
]


def main() -> None:
    """Lance l'interface Streamlit."""

    st.set_page_config(
        page_title="Puls-Events RAG",
        page_icon=":material/theater_comedy:",
        layout="wide",
    )
    st.title("Assistant culturel Puls-Events")

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
                "Auto : Mistral puis Ollama",
            ],
            index=0,
            help=(
                "Le mode Ollama utilise les ressources locales de la machine. "
                "Le mode auto tente Mistral, puis bascule sur Ollama si l'appel échoue."
            ),
        )
        llm_provider = {
            "Mistral API": "mistral",
            "Ollama local": "ollama",
            "Auto : Mistral puis Ollama": "auto",
        }[llm_provider_label]
        default_llm_model = "qwen2.5:7b" if llm_provider == "ollama" else ""
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
    """Appelle /ask et affiche la réponse."""

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

    data = response.json()
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


if __name__ == "__main__":
    main()
