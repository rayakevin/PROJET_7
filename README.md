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

## Variables cles

| Variable | Usage |
|---|---|
| `MISTRAL_API_KEY` | Embeddings et generation Mistral |
| `OPENDATASOFT_RECORDS_URL` | Endpoint OpenDataSoft source |
| `EVENTS_LOCATION` | Ville cible |
| `EVENTS_LOOKBACK_DAYS` | Historique recupere, en jours |
| `EVENTS_LOOKAHEAD_DAYS` | Evenements futurs recuperes, en jours |
| `EVENTS_PAGE_SIZE` | Taille de page OpenDataSoft |
| `DATA_DIR` | Racine des donnees locales |
| `VECTOR_STORE_DIR` | Index vectoriel |
| `TOP_K` | Nombre de chunks retournes |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Decoupage documentaire |

## Stack prevue

`Python 3.11` | `uv` | `FastAPI` | `OpenDataSoft` | `LangChain` |
`FAISS` | `Mistral` | `Ragas` | `pytest`

## Prochaine sequence

1. Brancher le chunking sur `data/processed/events_processed.json`.
2. Generer les embeddings Mistral.
3. Construire et sauvegarder l'index FAISS.
4. Exposer `/ask`, `/rebuild` et `/health` via FastAPI.
5. Automatiser l'evaluation avec le jeu de test annote.
