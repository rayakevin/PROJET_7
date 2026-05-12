"""Lance pytest et sauvegarde un rapport lisible sur disque."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse les arguments de ligne de commande."""

    parser = argparse.ArgumentParser(
        description="Exécute pytest et enregistre le résultat dans des fichiers."
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Arguments optionnels transmis tels quels à pytest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.evaluation_data_dir / "results",
        help="Dossier où écrire les rapports de tests.",
    )
    return parser.parse_args()


def extract_summary(output: str) -> str:
    """Extrait la dernière ligne de résumé pytest si elle existe."""

    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if " passed" in stripped or " failed" in stripped or " skipped" in stripped:
            return stripped
    return "Résumé pytest introuvable."


def write_reports(
    output_dir: Path,
    command: list[str],
    return_code: int,
    duration_seconds: float,
    stdout: str,
) -> tuple[Path, Path]:
    """Écrit un rapport texte complet et un résumé JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary_line = extract_summary(stdout)

    text_report = "\n".join(
        [
            f"Date UTC : {datetime.now(UTC).isoformat()}",
            f"Commande : {' '.join(command)}",
            f"Code retour : {return_code}",
            f"Duree secondes : {duration_seconds:.2f}",
            f"Resume : {summary_line}",
            "",
            "===== Sortie complete pytest =====",
            stdout.rstrip(),
            "",
        ]
    )
    text_path = output_dir / f"pytest_report_{timestamp}.txt"
    latest_text_path = output_dir / "pytest_report_latest.txt"
    text_path.write_text(text_report, encoding="utf-8")
    latest_text_path.write_text(text_report, encoding="utf-8")

    json_report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": command,
        "return_code": return_code,
        "duration_seconds": round(duration_seconds, 2),
        "summary": summary_line,
        "success": return_code == 0,
        "stdout": stdout,
    }
    json_path = output_dir / f"pytest_report_{timestamp}.json"
    latest_json_path = output_dir / "pytest_report_latest.json"
    json_text = json.dumps(json_report, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    latest_json_path.write_text(json_text, encoding="utf-8")

    return latest_text_path, latest_json_path


def main() -> int:
    """Exécute pytest puis persiste le résultat."""

    args = parse_args()
    command = [sys.executable, "-m", "pytest", *args.pytest_args]
    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    duration_seconds = time.perf_counter() - started_at
    combined_output = completed.stdout
    if completed.stderr:
        combined_output = f"{combined_output}\n{completed.stderr}".strip()

    latest_text_path, latest_json_path = write_reports(
        output_dir=args.output_dir,
        command=command,
        return_code=completed.returncode,
        duration_seconds=duration_seconds,
        stdout=combined_output,
    )

    print(f"Rapport texte pytest : {latest_text_path}")
    print(f"Rapport JSON pytest : {latest_json_path}")
    print(f"Résumé : {extract_summary(combined_output)}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
