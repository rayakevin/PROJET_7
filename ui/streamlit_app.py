"""Interface Streamlit locale pour interroger l'API RAG."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEMO_QUESTIONS = [
    "Je cherche un concert de Gospel Jazz pour la Fete de la musique a Paris, que peux-tu me proposer ?",
    "Je veux faire une activite cosplay a la Cite des sciences et de l'Industrie, quels evenements existent ?",
    "Je cherche un spectacle jeune public des 3 ans a Paris, as-tu une recommandation ?",
    "Je voudrais voir une exposition autour de l'art japonais a Paris, que proposes-tu ?",
    "Y a-t-il une avant-premiere du film RED BIRD a Paris ?",
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
                "FAISS retourne une distance, pas un score de similarite : "
                "plus la valeur est basse, plus le chunk est proche de la question. "
                "Le seuil garde les sources dont la distance est inferieure ou egale "
                "a cette valeur."
            ),
        )
        candidate_multiplier = st.slider(
            "Candidats avant reranking",
            min_value=1,
            max_value=20,
            value=8,
        )

        st.header("Generation")
        temperature = st.slider(
            "Temperature",
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

    selected_question = st.selectbox("Scenario de demo", DEMO_QUESTIONS)
    question = st.text_area("Question utilisateur", value=selected_question, height=110)

    if st.button("Interroger le RAG", type="primary"):
        payload = {
            "question": question,
            "top_k": top_k,
            "retrieval_max_score": retrieval_max_score,
            "retrieval_candidate_multiplier": candidate_multiplier,
            "temperature": temperature,
            "max_tokens": max_tokens,
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
    """Appelle /ask et affiche la reponse."""

    if not payload["question"].strip():
        st.warning("La question ne peut pas etre vide.")
        return

    try:
        with st.spinner("Recherche et generation en cours..."):
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
    st.subheader("Reponse")
    st.markdown(data["answer"])

    st.subheader("Sources")
    for index, source in enumerate(data.get("sources", []), start=1):
        with st.expander(f"{index}. {source['title']}"):
            st.write(f"Lieu : {source['location_name']} ({source['city']})")
            st.write(f"Debut : {source['start']}")
            st.write(f"Fin : {source['end']}")
            st.write(
                f"Distance FAISS : {source['score']:.4f} "
                "(plus bas = plus proche)"
            )
            st.write(f"UID evenement : {source['event_uid']}")

    st.subheader("Parametres appliques")
    st.json(data.get("parameters", {}))


if __name__ == "__main__":
    main()
