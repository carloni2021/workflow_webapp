# main.py
from __future__ import annotations
from pathlib import Path

from view.validation_plot1 import sweep_response_vs_lambda

from model.scenario import Scenario

# File di output CSV
OUT_CSV_FINITE = "out/summary_finite.csv"
OUT_FILE_PATH = "out/"  # Usato solo per creare la cartella

SEED0 = 1234 # Seed base per tutte le simulazioni

def run_phase_finite(config_dir: str = "config") -> None:
    """Esegue la fase a orizzonte finito (repliche) per TUTTI gli scenari."""
    yaml_files = sorted(Path(config_dir).glob("*.yaml"))
    if not yaml_files:
        raise SystemExit(f"Nessun file .yaml trovato in ./{config_dir}")

    for i, path in enumerate(yaml_files):
        scn = Scenario.from_yaml(str(path))
        sweep_response_vs_lambda(
            scn, lam_start=0.5, lam_end=1.2, lam_step=0.05, seed0=SEED0
        )

def main() -> None:

    config_dir="config"
    run_phase_finite(config_dir)

if __name__ == "__main__":
    main()
