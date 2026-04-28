# Autoévaluation - conformité au cahier des charges

Cette fiche relie les critères évaluateur aux preuves présentes dans le dépôt.

## Modèle d'apprentissage et RAG

| Critère | Statut | Preuves |
|---|---|---|
| Pertinence du modèle sélectionné | Validé | RAG avec embeddings Mistral, index FAISS, génération Mistral et fallback Ollama documentés dans `README.md`. |
| Rapport technique détaillé | Validé | `README.md` : architecture UML, composants, choix technologiques, résultats, limites et pistes d'amélioration. |
| Optimisation des hyperparamètres | Validé | Valeurs par défaut dans `app/config.py`, overrides optionnels dans `.env`, API `/ask` et UI Streamlit exposent `top_k`, distance FAISS, température et tokens. |
| Base de connaissance métier | Validé | Ingestion OpenDataSoft, normalisation, `full_text`, qualité dataset et index FAISS. |
| Descriptions vectorisées via Mistral | Validé | `app/rag/embeddings.py`, `app/rag/vector_store.py`, `scripts/rebuild_index.py`. Un mode Ollama local est disponible en option si l'index est reconstruit avec ce fournisseur. |
| LangChain entre FAISS et Mistral | Validé | FAISS LangChain dans `app/rag/vector_store.py`, prompt LangChain dans `app/rag/answer.py`, wrapper LLM Ragas dans `app/rag/llm.py`, fallback local Ollama dans `app/rag/answer.py`. |
| Réponses cohérentes avec le jeu de test | Validé | `tests/fixtures/qa_dataset.json`, `scripts/evaluate_rag.py`, résultats Ragas documentés. |
| Schéma UML et composants expliqués | Validé | Section "Rapport technique synthétique" du `README.md`. |

## Processus de test

| Critère | Statut | Preuves |
|---|---|---|
| Tests récupération | Validé | `tests/unit/test_fetch_events.py`, `tests/unit/test_opendatasoft_client.py`. |
| Tests nettoyage | Validé | `tests/unit/test_normalize_events.py`, `tests/unit/test_quality.py`. |
| Tests vectorisation/indexation | Validé | `tests/unit/test_chunking.py`, `tests/unit/test_vector_store.py`, `tests/integration/test_real_vector_store.py`. |
| Tests interrogation/génération | Validé | `tests/unit/test_qa_service.py`, `tests/unit/test_answer.py`, `tests/integration/test_real_qa_service.py`. |
| Jeu de test annoté | Validé | `tests/fixtures/qa_dataset.json`. |
| Comparaison réponses attendues/générées | Validé | `scripts/evaluate_rag.py`. |
| Relance automatique des tests | Validé | `pytest`, `ruff check .`, `scripts/evaluate_rag.py`. |
| Résultats documentés et lisibles | Validé | Rapports JSON dans `data/evaluation/results/` et résumé dans le `README.md`. |
| Indicateurs qualité | Validé | Qualité dataset, distance FAISS moyenne, source count, Ragas `faithfulness`, `answer_relevance`, `context_precision`. |
| Identification des limites | Validé | Sections limites/pistes d'amélioration dans `README.md` et `docs/soutenance/plan_slides.md`. |

## API et intégration

| Critère | Statut | Preuves |
|---|---|---|
| Appels API robustes | Validé | Validation Pydantic, gestion erreurs 400/403/503, pagination OpenDataSoft. |
| Données collectées prêtes pour indexation | Validé | Dataset normalisé, rapport qualité, champs titre/lieu/ville/dates/description. |
| Docker local RAG + API | Validé | `Dockerfile`, `docker-compose.yml`, `docs/soutenance/demo_docker.md`. |
| Vérification/enrichissement données | Validé | `app/ingestion/quality.py`, construction de `full_text`. |
| Documentation endpoints | Validé | Swagger FastAPI, `README.md`, `scripts/api_test.py`. |
| Routes cohérentes | Validé | `/health`, `/metadata`, `/ask`, `/rebuild`. |
| Test local de l'API | Validé | `tests/integration/test_api.py`, `scripts/api_test.py`. |

## Exploitation métier et soutenance

| Critère | Statut | Preuves |
|---|---|---|
| Réponses enrichies via API REST | Validé | `/ask` retourne `question`, `answer`, `sources`, `parameters`. |
| JSON structuré exploitable | Validé | Schémas Pydantic dans `app/api/schemas.py`. |
| Cas limites API | Validé | Question vide, index absent, erreur LLM Mistral/Ollama, token `/rebuild`. |
| Démonstration Docker fluide | Validé | Guide `docs/soutenance/demo_docker.md`, Docker Compose API + Streamlit. |
| Cas d'usage réalistes | Validé | Gospel Jazz, Cosplaymania, jeune public, exposition, avant-première RED BIRD. |
| Justification métier/technique | Validé | `README.md`, `docs/soutenance/plan_slides.md`, présentation PowerPoint. |

## Points de vigilance à annoncer à l'oral

- Les métriques Ragas sont mesurées sur 5 questions annotées : c'est suffisant
  pour un POC, mais pas pour une validation statistique complète.
- L'index doit être construit avant la démonstration Docker pour éviter de
  dépendre d'un rebuild long pendant la soutenance.
- Le mode Ollama réduit la dépendance à l'API Mistral pour la génération, mais
  les embeddings doivent rester cohérents avec ceux utilisés pour construire
  l'index.
- `/rebuild` est protégé par token optionnel ; une authentification plus forte
  serait nécessaire en production.
- Les questions temporelles fines peuvent encore nécessiter des filtres dédiés
  dans l'API.
