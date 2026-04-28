"""Tests du contrôle d'environnement.

Ces tests restent volontairement simples : ils vérifient que les briques
techniques attendues par le POC sont bien importables dans l'environnement.
"""

from scripts.check_environment import CHECKS, run_import_check


def test_environment_imports_are_available() -> None:
    """Vérifie que tous les imports critiques du projet sont disponibles."""

    results = [run_import_check(check) for check in CHECKS]

    assert all(results)
