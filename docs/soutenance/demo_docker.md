# Démo Docker - POC RAG Puls-Events

## Objectif

Montrer en soutenance que le chatbot RAG est exécutable localement dans un
conteneur Docker et requêtable via une API REST FastAPI.

## Prérequis

- Docker Desktop lancé, avec le moteur Linux disponible.
- Fichier `.env` présent à la racine avec `MISTRAL_API_KEY`.
- Index vectoriel déjà construit dans `data/vector_store`.
- Docker Compose monte le dossier local `./data` dans `/app/data`.
- Ollama lancé sur la machine hôte si le mode local ou fallback est utilisé.

Vérifier les artefacts locaux :

```powershell
Test-Path data\vector_store\index.faiss
Test-Path data\vector_store\index.pkl
Test-Path data\vector_store\chunks.json
```

## Build de l'image

```powershell
docker build -t projet7-rag-api .
```

## Lancement du conteneur

Avec Docker Compose :

```powershell
docker compose up --build
```

Lancement de l'interface Streamlit :

```text
http://127.0.0.1:8501
```

L'interface permet de régler la température, le nombre de sources, la distance
FAISS maximale, la longueur maximale de génération et le fournisseur LLM
(`mistral`, `ollama` ou `auto`). La distance FAISS n'est pas un score de
similarité : plus elle est basse, plus le chunk est proche de la question.

Ou sans Compose :

```powershell
docker run --rm --env-file .env -p 8000:8000 -v "${PWD}/data:/app/data" projet7-rag-api
```

Pour tester le fallback local avec Docker Compose, vérifier qu'Ollama répond
sur la machine hôte :

```powershell
ollama list
ollama pull qwen2.5:7b
```

Puis ajouter dans `.env` :

```text
LLM_PROVIDER=auto
OLLAMA_CHAT_MODEL=qwen2.5:7b
OLLAMA_MIN_TOKENS=600
OLLAMA_NUM_CTX=8192
```

Dans Docker Compose, `OLLAMA_BASE_URL` pointe vers
`http://host.docker.internal:11434` pour que le conteneur contacte Ollama sur
Windows.

## Vérification rapide

Swagger :

```text
http://127.0.0.1:8000/docs
```

Healthcheck :

```powershell
curl http://127.0.0.1:8000/health
```

Test fonctionnel :

```powershell
$env:API_BASE_URL="http://127.0.0.1:8000"
python scripts/api_test.py
```

## Scénarios de démo

### 1. Concert cible

```powershell
curl -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Je cherche un concert de Gospel Jazz pour la Fête de la musique à Paris, que peux-tu me proposer ?\"}"
```

Attendu : proposition du Concert de Gospel Jazz au 132 avenue de Versailles,
avec date et source.

### 2. Activité cosplay

```powershell
curl -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Je veux faire une activité cosplay à la Cité des sciences et de l'Industrie, quels événements existent ?\"}"
```

Attendu : recommandations Cosplaymania, concours/défilé ou atelier associé.

### 3. Jeune public

```powershell
curl -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Je cherche un spectacle jeune public dès 3 ans à Paris, as-tu une recommandation ?\"}"
```

Attendu : recommandation d'un spectacle jeune public avec lieu, date et sources.

## Reconstruction de l'index

Hors démo live, l'index peut être reconstruit avec :

```powershell
python scripts/rebuild_index.py --index
```

Dans le conteneur :

```powershell
docker compose exec rag-api python scripts/rebuild_index.py --index
```

Cette opération dépend de l'API Mistral et peut prendre plusieurs minutes sur
le dataset complet. Pour la soutenance, utiliser l'index local déjà présent
dans `data/vector_store` et monté dans le conteneur afin de garder une démo
fluide.
