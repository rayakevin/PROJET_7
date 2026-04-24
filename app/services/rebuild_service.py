"""Service de reconstruction d'index.

Ce fichier servira à orchestrer tout le pipeline de rebuild :
- récupération des données ;
- normalisation ;
- chunking ;
- embeddings ;
- création ou mise à jour de l'index vectoriel.
"""
