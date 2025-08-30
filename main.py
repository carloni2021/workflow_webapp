# main.py
from __future__ import annotations
from pathlib import Path

from view.validation_plot1 import sweep_response_vs_lambda
from view.validation_plot2 import sweep_response_vs_lambda_steady

from model.scenario import Scenario

# File di output CSV
OUT_CSV_FINITE = "out/summary_finite.csv"
OUT_FILE_PATH = "out/"  # Usato solo per creare la cartella

SEED0 = 1234 # Seed base per tutte le simulazioni

# Fase Orizzonte finito -> "finite"
FINITE_HORIZON_S = 24 * 60 * 60  # 1 giorno in secondi

def run_phase_finite(config_dir: str = "config") -> None:
    """Esegue la fase a orizzonte finito (repliche) per TUTTI gli scenari."""
    yaml_files = sorted(Path(config_dir).glob("*.yaml"))
    if not yaml_files:
        raise SystemExit(f"Nessun file .yaml trovato in ./{config_dir}")

    for i, path in enumerate(yaml_files):
        scn = Scenario.from_yaml(str(path))
        seed_finite = SEED0
        sweep_response_vs_lambda(
            scn, lam_start=0.5, lam_end=1.2, lam_step=0.1, seed0=SEED0
        )


def run_phase_steady(config_dir: str = "config") -> None:
    yaml_files = sorted(Path(config_dir).glob("*.yaml"))
    if not yaml_files:
        raise SystemExit(f"Nessun file .yaml trovato in ./{config_dir}")

    for i, path in enumerate(yaml_files):
        scn = Scenario.from_yaml(str(path))
        label = f"{i:02d}_{path.stem}"
        sweep_response_vs_lambda_steady(
            scn,
            lam_start=0.5, lam_end=1.2, lam_step=0.05,   # come nel libro
            n_batches=64, jobs_per_batch=1024,
            seed0=SEED0,
            outdir="out", save_png=True, save_csv=True, show=False,
            scenario_label=label, verbose=True
        )


def main() -> None:

    config_dir="config"
    run_phase_finite(config_dir)
    # run_phase_steady(config_dir)ni

if __name__ == "__main__":
    main()
