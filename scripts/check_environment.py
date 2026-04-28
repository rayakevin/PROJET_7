"""Contrôle rapide de l'environnement projet.

Ce script valide que les bibliothèques principales du POC sont importées.
Il sert de test manuel avant de développer les modules métier :
- ingestion OpenAgenda ;
- indexation FAISS ;
- orchestration LangChain ;
- appels Mistral ;
- API FastAPI.

On le garde volontairement simple pour pouvoir le relancer après une
installation propre sur une autre machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


@dataclass(frozen=True, slots=True)
class ImportCheck:
    """Décrit un import à vérifier.

    Attribute
    ----------
    label:
        Nom lisible affiché dans le terminal.
    module_path:
        Chemin du module Python à importer.
    attribute:
        Attribut optionnel à récupérer dans le module importé.
        Exemple : `FAISS` dans `langchain_community.vectorstores`.
    package_name:
        Nom du paquet installé, utilisé pour afficher sa version.
    """

    label: str
    module_path: str
    attribute: str | None = None
    package_name: str | None = None


CHECKS = [
    ImportCheck("FAISS natif", "faiss", package_name="faiss-cpu"),
    ImportCheck(
        "FAISS via LangChain",
        "langchain_community.vectorstores",
        attribute="FAISS",
        package_name="langchain-community",
    ),
    ImportCheck(
        "Embeddings LangChain",
        "langchain_community.embeddings",
        attribute="HuggingFaceEmbeddings",
        package_name="langchain-community",
    ),
    ImportCheck(
        "Client Mistral",
        "mistralai",
        attribute="Mistral",
        package_name="mistralai",
    ),
    ImportCheck("FastAPI", "fastapi", package_name="fastapi"),
    ImportCheck("Pandas", "pandas", package_name="pandas"),
    ImportCheck("Ragas", "ragas", package_name="ragas"),
]


def get_package_version(package_name: str | None) -> str:
    """Retourne la version installée d'un paquet.

    Parameters
    ----------
    package_name:
        Nom du paquet tel qu'il apparaît dans l'environnement Python.

    Returns
    -------
    str
        Version installée, ou `version inconnue` si elle n'est pas disponible.
    """

    if package_name is None:
        return "version inconnue"

    try:
        return version(package_name)
    except PackageNotFoundError:
        return "version inconnue"


def run_import_check(check: ImportCheck) -> bool:
    """Exécute une vérification d'import et affiche un résultat lisible.

    Parameters
    ----------
    check:
        Description de l'import à tester.

    Returns
    -------
    bool
        `True` si l'import réussit, sinon `False`.
    """

    try:
        module = import_module(check.module_path)

        # Certains contrôles visent une classe précise dans le module.
        if check.attribute is not None:
            getattr(module, check.attribute)

        package_version = get_package_version(check.package_name)
        print(f"[OK] {check.label} ({package_version})")
        return True
    except Exception as exc:
        print(f"[KO] {check.label} -> {exc.__class__.__name__}: {exc}")
        return False


def main() -> int:
    """Lance tous les contrôles d'import.

    Returns
    -------
    int
        Code de sortie compatible terminal :
        `0` si tout est valide, `1` si au moins un import échoue.
    """

    print("Contrôle de l'environnement Projet 7\n")
    results = [run_import_check(check) for check in CHECKS]

    if all(results):
        print("\nEnvironnement prêt.")
        return 0

    print("\nEnvironnement incomplet : corriger les imports en erreur.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
