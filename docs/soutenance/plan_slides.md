# Plan de presentation - Puls-Events

## Slide 1 - Titre

POC RAG pour un assistant de recommandations culturelles Puls-Events.

## Slide 2 - Contexte metier

Puls-Events veut tester un chatbot capable de repondre a des questions sur des
evenements culturels recents et a venir, a partir de donnees OpenAgenda exposees
via OpenDataSoft.

## Slide 3 - Objectifs du POC

- Recuperer et nettoyer les evenements.
- Construire un index vectoriel FAISS.
- Generer des reponses augmentees avec Mistral.
- Exposer le systeme via une API REST.
- Evaluer la qualite avec un jeu annote et Ragas.

## Slide 4 - Donnees utilisees

- Source unique : dataset public OpenDataSoft `evenements-publics-openagenda`.
- Zone cible : Paris.
- Fenetre temporelle : evenements recents et a venir selon la configuration.
- Champs exploites : titre, description, lieu, ville, dates, mots-cles.

## Slide 5 - Pipeline ingestion

OpenDataSoft -> JSON brut -> normalisation Pandas/Python -> controle qualite ->
dataset traite.

Points de vigilance : donnees manquantes, HTML residuel, champs multilingues,
qualite du champ `full_text`.

## Slide 6 - Indexation vectorielle

- Construction de chunks depuis `full_text`.
- Embeddings Mistral.
- Index FAISS via LangChain.
- Conservation des metadonnees pour sourcer les reponses.

## Slide 7 - Fonctionnement RAG

Question utilisateur -> embedding de la requete -> retrieval FAISS + reranking
hybride -> prompt contextualise -> generation Mistral -> JSON avec sources.

## Slide 8 - API REST

- `GET /health` : statut API et disponibilite de l'index.
- `POST /ask` : question utilisateur et reponse augmentee.
- `POST /rebuild` : reconstruction du dataset et de l'index.
- Swagger disponible sur `/docs`.

## Slide 9 - Conteneurisation Docker

- Image Docker avec API, code applicatif et artefacts locaux.
- Lancement via `docker compose up --build`.
- Configuration par `.env`.
- Demo locale via Swagger, curl ou `scripts/api_test.py`.

## Slide 10 - Evaluation Ragas

Metriques suivies :

- Faithfulness : fidelite au contexte.
- Answer relevance : pertinence de la reponse.
- Context precision : precision des contextes recuperes.

Dernier resultat observe :

- Faithfulness : 0.9417.
- Answer relevance : 0.8681.
- Context precision : 0.8667.

## Slide 11 - Demo live

Scenarios :

- Concert Gospel Jazz pour la Fete de la musique.
- Activite cosplay a la Cite des sciences et de l'Industrie.
- Spectacle jeune public des 3 ans.

## Slide 12 - Limites et perspectives

- Dependance a l'API Mistral pour embeddings et generation.
- Ameliorer encore le reranking.
- Ajouter filtres temporels explicites dans l'API.
- Augmenter le jeu de test annote.
- Mettre en place une CI d'evaluation.
