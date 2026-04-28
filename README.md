# Projet 7 - POC RAG culturel

POC de chatbot de recommandation culturelle base sur le dataset public
`evenements-publics-openagenda` expose par OpenDataSoft.

```text
OpenDataSoft -> ingestion -> dataset -> chunks -> embeddings -> vector store -> API
                            \_______________________________________________/
                                         evaluation + tests
```

## Source de donnees

Source unique du projet :

```text
https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records
```

Le projet n'utilise pas l'API OpenAgenda directe et ne necessite pas de cle
OpenAgenda. Les filtres principaux sont appliques via les parametres
OpenDataSoft : ville, periode, recherche texte et mots-cles.

## Arborescence

```text
app/
|-- clients/          # client OpenDataSoft
|-- ingestion/        # collecte, normalisation, dataset
|-- rag/              # chunking, embeddings, vector store, retrieval
|-- services/         # orchestration metier
|-- api/              # routes et schemas FastAPI
`-- config.py         # variables d'environnement
scripts/              # commandes locales
tests/                # unitaires, integration, fixtures
data/                 # donnees et artefacts locaux ignores par Git
docs/                 # rapport technique et soutenance
```

## Mise en route

```bash
uv sync --group dev
cp .env.example .env
python scripts/check_environment.py
pytest
```

## Ingestion

Construire le dataset brut puis normalise :

```bash
python scripts/rebuild_index.py --fetch
```

Exemple avec une ville explicite :

```bash
python scripts/rebuild_index.py --fetch --city Paris
```

Sorties par defaut :

- `data/raw/events_raw.json`
- `data/processed/events_processed.json`
- `data/processed/events_quality_report.json`

## Indexation RAG

Le dataset normalise est decoupe en chunks a partir du champ `full_text`, puis
les chunks sont vectorises avec Mistral et sauvegardes dans un index FAISS.

Construire uniquement l'index a partir du dataset deja present :

```bash
python scripts/rebuild_index.py --index
```

Faire un test limite pour eviter de vectoriser tout le dataset :

```bash
python scripts/rebuild_index.py --index --max-events 20
```

Reconstruire toute la chaine ingestion + dataset + index :

```bash
python scripts/rebuild_index.py --fetch --index --city Paris
```

Sorties FAISS par defaut :

- `data/vector_store/index.faiss`
- `data/vector_store/index.pkl`
- `data/vector_store/chunks.json`

## Chatbot RAG

Le service de question-reponse charge l'index FAISS, recupere les chunks les
plus proches avec LangChain, construit un prompt contextualise, puis genere une
reponse naturelle avec Mistral.

Exemple Python local :

```bash
python -c "from app.services.qa_service import QAService; r=QAService().ask('Quels concerts de jazz sont disponibles a Paris ?'); print(r.to_dict())"
```

La reponse contient :

- `question` : question utilisateur ;
- `answer` : reponse generee par Mistral ;
- `sources` : chunks/evenements utilises avec titre, lieu, dates et score.

## API REST

Lancer l'API localement :

```bash
python scripts/run_api.py
```

Swagger est disponible a l'adresse :

```text
http://127.0.0.1:8000/docs
```

Endpoints principaux :

- `GET /health` : etat de l'API et presence de l'index local ;
- `POST /ask` : question utilisateur vers le chatbot RAG ;
- `POST /rebuild` : reconstruction du dataset et de l'index FAISS.

Exemple `/ask` :

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Quels concerts de jazz sont disponibles a Paris ?\"}"
```

Exemple `/rebuild` limite a 20 evenements pour un test rapide :

```bash
curl -X POST http://127.0.0.1:8000/rebuild \
  -H "Content-Type: application/json" \
  -d "{\"fetch\":false,\"max_events\":20}"
```

Si `API_REBUILD_TOKEN` est renseigne, `/rebuild` exige l'en-tete
`X-Rebuild-Token`.

Test fonctionnel manuel :

```bash
python scripts/api_test.py
```

## Docker

L'image Docker embarque l'API et les artefacts locaux presents dans `data/`.
Docker Desktop doit etre lance avant le build.
Pour une demo fluide, construire l'index avant le build :

```bash
python scripts/rebuild_index.py --index
```

Construire puis lancer l'API :

```bash
docker build -t projet7-rag-api .
docker run --rm --env-file .env -p 8000:8000 projet7-rag-api
```

Avec Docker Compose :

```bash
docker compose up --build
```

Interfaces disponibles :

- API Swagger : `http://127.0.0.1:8000/docs`
- UI Streamlit : `http://127.0.0.1:8501`

Verification :

```bash
curl http://127.0.0.1:8000/health
python scripts/api_test.py
```

Le guide de demo est disponible dans `docs/soutenance/demo_docker.md`.

## Interface Streamlit

Une interface locale permet d'interroger l'API avec des controles sur les
principaux hyperparametres :

- temperature du LLM ;
- nombre de sources `top_k` ;
- distance FAISS maximale, plus basse signifie plus proche de la question ;
- nombre de candidats avant reranking ;
- longueur maximale de reponse.

Lancer l'API puis l'interface en local :

```bash
python scripts/run_api.py
streamlit run ui/streamlit_app.py
```

Ou lancer les deux via Docker Compose :

```bash
docker compose up --build
```

## Evaluation

Le jeu de test annote se trouve dans `tests/fixtures/qa_dataset.json`.
Le script d'evaluation interroge le chatbot, stocke les reponses et calcule :

- metriques locales : nombre de sources et score moyen de retrieval ;
- metriques Ragas attendues par la grille : `faithfulness`, `answer_relevance`
  et `context_precision` ;
- metrique Ragas complementaire : similarite semantique des reponses.

Lancer l'evaluation complete :

```bash
python scripts/evaluate_rag.py
```

Lancer uniquement les metriques locales :

```bash
python scripts/evaluate_rag.py --skip-ragas
```

Sorties par defaut :

- `data/evaluation/results/rag_evaluation_<timestamp>.json`
- `data/evaluation/results/rag_evaluation_latest.json`

Le rapport expose les noms internes Ragas et un resume lisible dans
`summary.required_ragas_metrics` :

```json
{
  "faithfulness": 0.9417,
  "answer_relevance": 0.8681,
  "context_precision": 0.8667
}
```

Dernier resultat observe sur 5 questions annotees :

```text
Faithfulness: 0.9417
Answer relevance: 0.8681
Context precision: 0.8667
Ragas semantic_similarity: 0.9479
Sources moyennes: 3.0
```

## Variables cles

| Variable | Usage |
|---|---|
| `MISTRAL_API_KEY` | Embeddings et generation Mistral |
| `API_REBUILD_TOKEN` | Token optionnel pour proteger `/rebuild` |
| `MISTRAL_EMBEDDING_MODEL` | Modele d'embeddings, par defaut `mistral-embed` |
| `MISTRAL_CHAT_MODEL` | Modele de generation, par defaut `mistral-small-latest` |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | Parametres de generation |
| `EMBEDDING_BATCH_SIZE` | Taille des lots envoyes a Mistral |
| `OPENDATASOFT_RECORDS_URL` | Endpoint OpenDataSoft source |
| `EVENTS_LOCATION` | Ville cible |
| `EVENTS_LOOKBACK_DAYS` | Historique recupere, en jours |
| `EVENTS_LOOKAHEAD_DAYS` | Evenements futurs recuperes, en jours |
| `EVENTS_PAGE_SIZE` | Taille de page OpenDataSoft |
| `DATA_DIR` | Racine des donnees locales |
| `VECTOR_STORE_DIR` | Index vectoriel |
| `TOP_K` | Nombre maximum de chunks retournes, par defaut `3` |
| `RETRIEVAL_MAX_SCORE` | Distance FAISS maximale conservee, par defaut `0.45` |
| `RETRIEVAL_CANDIDATE_MULTIPLIER` | Nombre de candidats FAISS avant reranking |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Decoupage documentaire |

## Stack prevue

`Python 3.11` | `uv` | `FastAPI` | `OpenDataSoft` | `LangChain` |
`FAISS` | `Mistral` | `Ragas` | `pytest`

## Prochaine sequence

1. Tester le build Docker apres lancement de Docker Desktop.
2. Finaliser le rapport technique.
