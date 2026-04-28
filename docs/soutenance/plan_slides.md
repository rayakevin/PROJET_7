# Plan de présentation - Puls-Events

## Slide 1 - Titre

POC RAG pour un assistant de recommandations culturelles Puls-Events.

## Slide 2 - Contexte métier

Puls-Events veut tester un chatbot capable de répondre à des questions sur des
événements culturels récents et à venir, à partir de données OpenAgenda exposées
via OpenDataSoft.

## Slide 3 - Objectifs du POC

- Récupérer et nettoyer les événements.
- Construire un index vectoriel FAISS.
- Générer des réponses augmentées avec Mistral.
- Exposer le système via une API REST.
- Évaluer la qualité avec un jeu annoté et Ragas.

## Slide 4 - Données utilisées

- Source unique : dataset public OpenDataSoft `evenements-publics-openagenda`.
- Zone cible : Paris.
- Fenêtre temporelle : événements récents et à venir selon la configuration.
- Champs exploités : titre, description, lieu, ville, dates, mots-clés.

## Slide 5 - Pipeline ingestion

OpenDataSoft -> JSON brut -> normalisation Pandas/Python -> contrôle qualité ->
dataset traité.

Points de vigilance : données manquantes, HTML résiduel, champs multilingues,
qualité du champ `full_text`.

## Slide 6 - Indexation vectorielle

- Construction de chunks depuis `full_text`.
- Embeddings Mistral.
- Index FAISS via LangChain.
- Conservation des métadonnées pour sourcer les réponses.

## Slide 7 - Fonctionnement RAG

Question utilisateur -> embedding de la requête -> retrieval FAISS + reranking
hybride -> prompt contextualisé -> génération Mistral ou Ollama -> JSON avec
sources.

Mode local : `LLM_PROVIDER=ollama` force la génération locale. Mode fallback :
`LLM_PROVIDER=auto` tente Mistral puis bascule sur Ollama.

## Slide 8 - API REST

- `GET /health` : statut API et disponibilité de l'index.
- `POST /ask` : question utilisateur et réponse augmentée.
- `POST /rebuild` : reconstruction du dataset et de l'index.
- Swagger disponible sur `/docs`.

## Slide 9 - Conteneurisation Docker

- Image Docker avec API et code applicatif.
- Montage local de `./data` dans `/app/data` pour utiliser l'index FAISS.
- Lancement via `docker compose up --build`.
- Valeurs par défaut dans `app/config.py`, secrets et overrides dans `.env`.
- Démo locale via Swagger, curl, Streamlit ou `scripts/api_test.py`.

## Slide 10 - Évaluation Ragas

Métriques suivies :

- Faithfulness : fidélité au contexte.
- Answer relevance : pertinence de la réponse.
- Context precision : précision des contextes récupérés.

Dernier résultat observé :

- Faithfulness : 0.9464.
- Answer relevance : 0.8618.
- Context precision : 1.0000.

## Slide 11 - Démo live

Scénarios :

- Concert Gospel Jazz pour la Fête de la musique.
- Activité cosplay à la Cité des sciences et de l'Industrie.
- Spectacle jeune public des 3 ans.

## Slide 12 - Limites et perspectives

- Dépendance à l'API Mistral si l'on garde les embeddings et la génération par
  défaut.
- Possibilité de réduire cette dépendance avec Ollama, surtout pour la
  génération après retrieval.
- Améliorer encore le reranking.
- Ajouter filtres temporels explicites dans l'API.
- Augmenter le jeu de test annoté.
- Mettre en place une CI d'évaluation.
