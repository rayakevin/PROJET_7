# Demo Docker - POC RAG Puls-Events

## Objectif

Montrer en soutenance que le chatbot RAG est executable localement dans un
conteneur Docker et requetable via une API REST FastAPI.

## Prerequis

- Docker Desktop lance, avec le moteur Linux disponible.
- Fichier `.env` present a la racine avec `MISTRAL_API_KEY`.
- Index vectoriel deja construit dans `data/vector_store`.

Verifier les artefacts locaux :

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

L'interface permet de regler la temperature, le nombre de sources, la distance
FAISS maximale, le nombre de candidats avant reranking et la longueur maximale
de generation. La distance FAISS n'est pas un score de similarite : plus elle
est basse, plus le chunk est proche de la question.

Ou sans Compose :

```powershell
docker run --rm --env-file .env -p 8000:8000 projet7-rag-api
```

## Verification rapide

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

## Scenarios de demo

### 1. Concert cible

```powershell
curl -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Je cherche un concert de Gospel Jazz pour la Fete de la musique a Paris, que peux-tu me proposer ?\"}"
```

Attendu : proposition du Concert de Gospel Jazz au 132 avenue de Versailles,
avec date et source.

### 2. Activite cosplay

```powershell
curl -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Je veux faire une activite cosplay a la Cite des sciences et de l'Industrie, quels evenements existent ?\"}"
```

Attendu : recommandations Cosplaymania, concours/defile ou atelier associe.

### 3. Jeune public

```powershell
curl -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Je cherche un spectacle jeune public des 3 ans a Paris, as-tu une recommandation ?\"}"
```

Attendu : recommandation d'un spectacle jeune public avec lieu, date et sources.

## Reconstruction de l'index

Hors demo live, l'index peut etre reconstruit avec :

```powershell
python scripts/rebuild_index.py --index
```

Dans le conteneur :

```powershell
docker compose exec rag-api python scripts/rebuild_index.py --index
```

Cette operation depend de l'API Mistral et peut prendre plusieurs minutes sur
le dataset complet. Pour la soutenance, utiliser l'index deja embarque dans
l'image afin de garder une demo fluide.
