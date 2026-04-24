# Projet 7 - POC RAG pour des recommandations d'événements culturels

## Vue d'ensemble

Ce dépôt contient un **Proof of Concept (POC)** de système **RAG** (*Retrieval-Augmented Generation*) capable de répondre à des questions sur des événements culturels en s'appuyant sur des données issues de l'API **OpenAgenda**.

Le projet est conçu pour couvrir l'ensemble du besoin de mission :

- récupération et préparation des données ;
- indexation vectorielle ;
- génération de réponses augmentées ;
- exposition via une API REST ;
- évaluation automatique ;
- exécution locale reproductible.

## Objectifs

- Démontrer la faisabilité technique d'un chatbot culturel.
- Construire une chaîne de traitement simple à rejouer localement.
- Préparer une base propre pour intégrer ensuite **LangChain**, **FAISS** et **Mistral** de manière complète.
- Exposer une API testable rapidement.

## Stack technique

- **Python 3.11**
- **uv** pour l'environnement et les dépendances
- **FastAPI** pour l'API
- **Uvicorn** pour le serveur ASGI
- **Requests** pour les appels HTTP
- **Pandas** pour la manipulation de données
- **LangChain** pour l'orchestration RAG
- **FAISS** pour la recherche vectorielle
- **Mistral** pour les modèles et embeddings
- **Pytest** pour les tests
- **Ragas** pour l'évaluation automatique

## Installation

### Prérequis

- Python `3.11.x`
- `uv` installé

### Création de l'environnement

```bash
uv python install 3.11
uv python pin 3.11
uv venv --python 3.11
```

Sous PowerShell :

```powershell
.venv\Scripts\Activate.ps1
```

### Installation des dépendances

```bash
uv sync
```

Pour installer aussi les dépendances de développement :

```bash
uv sync --group dev
```

## Variables d'environnement

Le fichier `.env` local n'est pas versionné. Un exemple est fourni dans [.env.example].

Variables principales :

- `MISTRAL_API_KEY`
- `OPENAGENDA_API_KEY`
- `OPENAGENDA_BASE_URL`
- `OPENAGENDA_LOCATION`
- `DATA_DIR`
- `VECTOR_STORE_DIR`
- `TOP_K`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

## Architecture générale

### Schéma fonctionnel

```text
                         +----------------------+
                         |   OpenAgenda API     |
                         |  Source événements   |
                         +----------+-----------+
                                    |
                                    v
                    +---------------+----------------+
                    | app/clients/openagenda_client.py |
                    | Appels HTTP + paramètres API     |
                    +---------------+------------------+
                                    |
                                    v
                +-------------------+--------------------+
                |          app/ingestion/                |
                | fetch_events -> normalize_events       |
                | -> build_dataset                       |
                +-------------------+--------------------+
                                    |
                                    v
                 +------------------+-------------------+
                 |       data/processed/                |
                 | Dataset nettoyé prêt à indexer       |
                 +------------------+-------------------+
                                    |
                                    v
                  +-----------------+------------------+
                  |            app/rag/               |
                  | chunking -> embeddings            |
                  | -> vector_store -> retriever      |
                  | -> answer                         |
                  +-----------------+------------------+
                                    |
                +-------------------+-------------------+
                |                                       |
                v                                       v
   +------------+-------------+          +--------------+-------------+
   | app/services/rebuild_    |          | app/services/qa_service.py |
   | service.py               |          | Question -> contexte ->    |
   | Reconstruit l'index      |          | réponse                    |
   +------------+-------------+          +--------------+-------------+
                |                                       |
                v                                       v
   +------------+-------------+          +--------------+-------------+
   |  POST /rebuild           |          |  POST /ask                 |
   |  app/api/routes.py       |          |  app/api/routes.py         |
   +--------------------------+          +----------------------------+
```

### Lecture rapide

| Bloc | Rôle |
|---|---|
| `clients/` | Accès aux sources externes, ici OpenAgenda |
| `ingestion/` | Récupération, nettoyage et structuration des événements |
| `rag/` | Pipeline de recherche sémantique et génération |
| `services/` | Orchestration métier réutilisable |
| `api/` | Exposition HTTP via FastAPI |
| `data/` | Stockage local des jeux de données et artefacts |
| `scripts/` | Lancement rapide des opérations clés |
| `tests/` | Validation unitaire et intégration |

## Arborescence du projet

```text
PROJET_7/
├─ app/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ main.py
│  ├─ clients/
│  │  ├─ __init__.py
│  │  └─ openagenda_client.py
│  ├─ ingestion/
│  │  ├─ __init__.py
│  │  ├─ fetch_events.py
│  │  ├─ normalize_events.py
│  │  └─ build_dataset.py
│  ├─ rag/
│  │  ├─ __init__.py
│  │  ├─ chunking.py
│  │  ├─ embeddings.py
│  │  ├─ retriever.py
│  │  ├─ vector_store.py
│  │  └─ answer.py
│  ├─ services/
│  │  ├─ __init__.py
│  │  ├─ rebuild_service.py
│  │  └─ qa_service.py
│  ├─ api/
│  │  ├─ __init__.py
│  │  ├─ schemas.py
│  │  └─ routes.py
│  └─ utils/
│     ├─ __init__.py
│     ├─ io.py
│     └─ logging.py
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  ├─ vector_store/
│  └─ evaluation/
│     └─ results/
├─ docs/
│  ├─ rapport_technique/
│  └─ soutenance/
├─ scripts/
│  ├─ rebuild_index.py
│  ├─ run_api.py
│  └─ evaluate_rag.py
├─ tests/
│  ├─ fixtures/
│  │  ├─ qa_dataset.json
│  │  └─ sample_events.json
│  ├─ integration/
│  │  ├─ test_api.py
│  │  └─ test_rebuild_index.py
│  └─ unit/
│     ├─ test_chunking.py
│     ├─ test_normalize_events.py
│     └─ test_vector_store.py
├─ .env.example
├─ .gitignore
├─ Dockerfile
├─ main.py
├─ pyproject.toml
└─ README.md
```

## Référentiel des dossiers et fichiers

### Racine du projet

| Élément | Rôle |
|---|---|
| [pyproject.toml](C:/Users/kevin/Documents/PROJET_7/pyproject.toml:1) | Dépendances, métadonnées projet, groupe `dev` |
| [README.md](C:/Users/kevin/Documents/PROJET_7/README.md:1) | Documentation du projet, architecture, consignes |
| [.env.example](C:/Users/kevin/Documents/PROJET_7/.env.example:1) | Exemple des variables d'environnement à renseigner |
| [.gitignore](C:/Users/kevin/Documents/PROJET_7/.gitignore:1) | Exclusion des secrets, caches et artefacts locaux |
| [Dockerfile](C:/Users/kevin/Documents/PROJET_7/Dockerfile:1) | Futur conteneur de démo pour l'API |
| [main.py](C:/Users/kevin/Documents/PROJET_7/main.py:1) | Point d'entrée racine pour lancer l'API rapidement |

### `app/`

| Fichier | Rôle |
|---|---|
| [app/__init__.py](C:/Users/kevin/Documents/PROJET_7/app/__init__.py:1) | Déclare `app` comme package Python |
| [app/config.py](C:/Users/kevin/Documents/PROJET_7/app/config.py:1) | Centralise la configuration, les chemins et les paramètres globaux |
| [app/main.py](C:/Users/kevin/Documents/PROJET_7/app/main.py:1) | Crée l'application FastAPI et branche les routes |

### `app/clients/`

| Fichier | Rôle |
|---|---|
| [app/clients/__init__.py](C:/Users/kevin/Documents/PROJET_7/app/clients/__init__.py:1) | Déclare le sous-package |
| [app/clients/openagenda_client.py](C:/Users/kevin/Documents/PROJET_7/app/clients/openagenda_client.py:1) | Encapsule les appels à OpenAgenda et fournit un fallback local pour le scaffold |

### `app/ingestion/`

| Fichier | Rôle |
|---|---|
| [app/ingestion/__init__.py](C:/Users/kevin/Documents/PROJET_7/app/ingestion/__init__.py:1) | Déclare le sous-package |
| [app/ingestion/fetch_events.py](C:/Users/kevin/Documents/PROJET_7/app/ingestion/fetch_events.py:1) | Récupère les événements bruts et les écrit dans `data/raw/` |
| [app/ingestion/normalize_events.py](C:/Users/kevin/Documents/PROJET_7/app/ingestion/normalize_events.py:1) | Homogénéise les champs métier et construit `full_text` |
| [app/ingestion/build_dataset.py](C:/Users/kevin/Documents/PROJET_7/app/ingestion/build_dataset.py:1) | Génère le dataset final prêt à indexer dans `data/processed/` |

### `app/rag/`

| Fichier | Rôle |
|---|---|
| [app/rag/__init__.py](C:/Users/kevin/Documents/PROJET_7/app/rag/__init__.py:1) | Déclare le sous-package |
| [app/rag/chunking.py](C:/Users/kevin/Documents/PROJET_7/app/rag/chunking.py:1) | Découpe les textes en chunks |
| [app/rag/embeddings.py](C:/Users/kevin/Documents/PROJET_7/app/rag/embeddings.py:1) | Fournit le composant d'embeddings, actuellement simplifié |
| [app/rag/vector_store.py](C:/Users/kevin/Documents/PROJET_7/app/rag/vector_store.py:1) | Stockage vectoriel local minimal, futur point de remplacement par FAISS |
| [app/rag/retriever.py](C:/Users/kevin/Documents/PROJET_7/app/rag/retriever.py:1) | Recherche les chunks les plus proches d'une question |
| [app/rag/answer.py](C:/Users/kevin/Documents/PROJET_7/app/rag/answer.py:1) | Génère une réponse textuelle à partir du contexte récupéré |

### `app/services/`

| Fichier | Rôle |
|---|---|
| [app/services/__init__.py](C:/Users/kevin/Documents/PROJET_7/app/services/__init__.py:1) | Déclare le sous-package |
| [app/services/rebuild_service.py](C:/Users/kevin/Documents/PROJET_7/app/services/rebuild_service.py:1) | Orchestration complète du rebuild de l'index |
| [app/services/qa_service.py](C:/Users/kevin/Documents/PROJET_7/app/services/qa_service.py:1) | Orchestration de la chaîne question -> retrieval -> réponse |

### `app/api/`

| Fichier | Rôle |
|---|---|
| [app/api/__init__.py](C:/Users/kevin/Documents/PROJET_7/app/api/__init__.py:1) | Déclare le sous-package |
| [app/api/schemas.py](C:/Users/kevin/Documents/PROJET_7/app/api/schemas.py:1) | Schémas Pydantic des requêtes et réponses |
| [app/api/routes.py](C:/Users/kevin/Documents/PROJET_7/app/api/routes.py:1) | Endpoints `/health`, `/ask`, `/rebuild` |

### `app/utils/`

| Fichier | Rôle |
|---|---|
| [app/utils/__init__.py](C:/Users/kevin/Documents/PROJET_7/app/utils/__init__.py:1) | Déclare le sous-package |
| [app/utils/io.py](C:/Users/kevin/Documents/PROJET_7/app/utils/io.py:1) | Lecture et écriture JSON |
| [app/utils/logging.py](C:/Users/kevin/Documents/PROJET_7/app/utils/logging.py:1) | Logger standard du projet |

### `scripts/`

| Fichier | Rôle |
|---|---|
| [scripts/rebuild_index.py](C:/Users/kevin/Documents/PROJET_7/scripts/rebuild_index.py:1) | Lance une reconstruction d'index depuis le terminal |
| [scripts/run_api.py](C:/Users/kevin/Documents/PROJET_7/scripts/run_api.py:1) | Lance l'API FastAPI avec Uvicorn |
| [scripts/evaluate_rag.py](C:/Users/kevin/Documents/PROJET_7/scripts/evaluate_rag.py:1) | Exécute une évaluation simple sur un dataset de test |

### `tests/`

| Fichier | Rôle |
|---|---|
| [tests/fixtures/sample_events.json](C:/Users/kevin/Documents/PROJET_7/tests/fixtures/sample_events.json:1) | Jeu de données d'événements de test |
| [tests/fixtures/qa_dataset.json](C:/Users/kevin/Documents/PROJET_7/tests/fixtures/qa_dataset.json:1) | Questions / réponses de référence |
| [tests/unit/test_normalize_events.py](C:/Users/kevin/Documents/PROJET_7/tests/unit/test_normalize_events.py:1) | Tests unitaires de normalisation |
| [tests/unit/test_chunking.py](C:/Users/kevin/Documents/PROJET_7/tests/unit/test_chunking.py:1) | Tests unitaires du découpage |
| [tests/unit/test_vector_store.py](C:/Users/kevin/Documents/PROJET_7/tests/unit/test_vector_store.py:1) | Tests unitaires du vector store local |
| [tests/integration/test_api.py](C:/Users/kevin/Documents/PROJET_7/tests/integration/test_api.py:1) | Test d'intégration du endpoint `/health` |
| [tests/integration/test_rebuild_index.py](C:/Users/kevin/Documents/PROJET_7/tests/integration/test_rebuild_index.py:1) | Test d'intégration du rebuild minimal |

### `data/`

| Élément | Rôle |
|---|---|
| `data/raw/` | Données brutes récupérées depuis la source |
| `data/interim/` | Données intermédiaires si une étape de transformation le nécessite |
| `data/processed/` | Dataset normalisé prêt à être chunké et indexé |
| `data/vector_store/` | Index vectoriel et artefacts associés |
| `data/evaluation/` | Jeux et sorties d'évaluation |

## Commandes utiles

Reconstruire l'index local :

```bash
python scripts/rebuild_index.py
```

Lancer l'API :

```bash
python scripts/run_api.py
```

Ou :

```bash
python main.py
```

Évaluer le système :

```bash
python scripts/evaluate_rag.py
```

Lancer les tests :

```bash
pytest
```

## Conventions de nommage

- fichiers et dossiers : minuscules avec underscores ;
- variables Python : `snake_case` ;
- classes : `PascalCase` ;
- constantes : `UPPER_SNAKE_CASE` ;
- routes API : courtes, explicites, en minuscules.

## Conventions Git

### Nommage des branches

Format recommandé :

```text
type/description-courte
```

Types conseillés :

- `feature/` pour une nouvelle fonctionnalité ;
- `fix/` pour une correction ;
- `docs/` pour la documentation ;
- `test/` pour les tests ;
- `refactor/` pour une réorganisation du code sans changement fonctionnel ;
- `chore/` pour l'environnement, les dépendances, Docker ou la maintenance.

Exemples :

- `feature/openagenda-ingestion`
- `feature/faiss-indexing`
- `feature/fastapi-endpoints`
- `feature/rag-answer-generation`
- `docs/readme-architecture`
- `test/api-endpoints`
- `chore/docker-setup`
- `fix/rebuild-endpoint`

Règles retenues :

- utiliser uniquement des minuscules ;
- séparer les mots avec des tirets ;
- éviter les accents, espaces et caractères spéciaux ;
- garder un nom de branche court mais explicite ;
- limiter une branche à un seul sujet principal.

### Nommage des commits

Format recommandé :

```text
type(scope): message court
```

Types conseillés :

- `feat` : ajout de fonctionnalité ;
- `fix` : correction ;
- `docs` : documentation ;
- `test` : ajout ou mise à jour de tests ;
- `refactor` : refonte sans changement fonctionnel ;
- `chore` : maintenance, configuration, dépendances, Docker ;
- `perf` : amélioration de performance.

Scopes recommandés pour ce projet :

- `readme`
- `config`
- `ingestion`
- `openagenda`
- `rag`
- `embeddings`
- `vector-store`
- `api`
- `tests`
- `docker`

Exemples :

- `docs(readme): add architecture overview`
- `chore(config): initialize project dependencies`
- `feat(ingestion): add event normalization pipeline`
- `feat(api): add ask and rebuild endpoints`
- `refactor(vector-store): prepare faiss integration`
- `test(api): add health endpoint integration test`
- `fix(openagenda): handle missing event fields`

Règles retenues :

- utiliser l'anglais pour les messages de commit ;
- garder la première ligne courte et explicite ;
- ne pas terminer le message par un point ;
- décrire une action concrète.

## État actuel

Le dépôt est actuellement **structuré et documenté**, avec une base de travail propre pour démarrer l'implémentation :

- l'arborescence du projet est définie ;
- les dossiers et fichiers principaux sont créés ;
- les fichiers Python contiennent des commentaires de cadrage sur leur futur rôle ;
- le `README.md` documente l'architecture, les conventions et l'organisation du dépôt ;
- le `pyproject.toml` contient une première base de dépendances ;
- le `.gitignore` et le `.env.example` sont en place.

À ce stade, le dépôt ne contient pas encore l'implémentation métier du pipeline RAG. La suite logique consistera à développer progressivement les briques suivantes :

- l'ingestion OpenAgenda ;
- la normalisation des événements ;
- le chunking et les embeddings ;
- l'indexation avec **FAISS** ;
- l'orchestration avec **LangChain** ;
- la génération avec **Mistral** ;
- l'API FastAPI ;
- les tests et l'évaluation avec **Ragas** ;
- la conteneurisation Docker pour la démonstration finale.
