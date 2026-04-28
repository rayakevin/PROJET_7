"""Utilitaires d'entrée / sortie.

Ce module centralise les petites opérations fichier du projet.
L'objectif est d'éviter de réécrire la lecture/écriture JSON dans chaque
script du pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Lit un fichier JSON.

    Parameters
    ----------
    path:
        Chemin du fichier JSON à lire.

    Returns
    -------
    Any
        Contenu Python issu du JSON : liste, dictionnaire, chaîne, etc.
    """

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: Any, path: str | Path) -> Path:
    """Ecrit des données Python dans un fichier JSON.

    Parameters
    ----------
    data:
        Données sérialisables en JSON.
    path:
        Chemin de sortie.

    Returns
    -------
    Path
        Chemin du fichier écrit, pratique pour les scripts et les tests.
    """

    output_path = Path(path)

    # On crée le dossier parent pour permettre une exécution depuis zéro.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return output_path
