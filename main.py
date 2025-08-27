# main.py
from __future__ import annotations
from pathlib import Path
import argparse
from typing import Optional

from model.scenario import Scenario
from controller.sim_runner import run_and_save_finite, run_and_save_steady

# File di output CSV
OUT_CSV_FINITE = "out/summary_finite.csv"
OUT_CSV_STEADY = "out/summary_steady.csv"
OUT_FILE_PATH = "out/"  # Usato solo per creare la cartella

SEED0 = 1234 # Seed base per tutte le simulazioni

# Fase Orizzonte finito -> "finite"
FINITE_HORIZON_S = 24 * 60 * 60  # 1 giorno in secondi

# Fase Orizzonte infinito -> "steady"
N_BATCHES = 64
JOBS_PER_BATCH = 1024


def run_phase_finite(config_dir: str = "config") -> None:
    """Esegue la fase a orizzonte finito (repliche) per TUTTI gli scenari."""
    yaml_files = sorted(Path(config_dir).glob("*.yaml"))
    if not yaml_files:
        raise SystemExit(f"Nessun file .yaml trovato in ./{config_dir}")

    for i, path in enumerate(yaml_files):
        scn = Scenario.from_yaml(str(path))
        seed_finite = SEED0
        res_fin = run_and_save_finite(
            scn,
            OUT_CSV_FINITE,
            seed0=seed_finite,
            horizon_s=FINITE_HORIZON_S,
            label_suffix=" (finite)",
        )
        print(f"[finite] {scn.name}: {res_fin}")


def run_phase_steady(config_dir: str = "config") -> None:
    """Esegue la fase steady-state (batch means) per TUTTI gli scenari."""
    yaml_files = sorted(Path(config_dir).glob("*.yaml"))
    if not yaml_files:
        raise SystemExit(f"Nessun file .yaml trovato in ./{config_dir}")

    for i, path in enumerate(yaml_files):
        scn = Scenario.from_yaml(str(path))
        # Si separano i seed per evitare correlazioni con la fase finite
        seed_steady = SEED0 + 50_000
        res_ss = run_and_save_steady(
            scn,
            OUT_CSV_STEADY,
            seed0=seed_steady,
            n_batches=N_BATCHES,
            jobs_per_batch=JOBS_PER_BATCH,
            label_suffix=" (steady)",
        )
        print(f"[steady] {scn.name}: {res_ss}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Esegui simulazioni finite e/o steady-state per tutti gli scenari in ./config"
    )
    parser.add_argument(
        "--mode",
        choices=["finite", "steady", "both"],
        help="Quale fase eseguire: finite, steady o both (finite poi steady). "
             "Se omesso, verrà chiesto a runtime."
    )
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Directory degli YAML di scenario (default: ./config)"
    )
    return parser.parse_args()


def main() -> None:

    args = parse_args()

    mode: Optional[str] = args.mode
    config_dir: str = args.config_dir

    if mode is None:
        # Prompt interattivo se non è stato passato --mode
        print("Seleziona modalità: [F]inite, [S]teady, [B]oth (finite poi steady)")
        choice = input("Modalità (F/S/B): ").strip().lower()
        mode = {"f": "finite", "s": "steady", "b": "both"}.get(choice)
        if mode is None:
            raise SystemExit("Scelta non valida. Usa --mode finite|steady|both o riprova.")

    # Crea la cartella di output se non esiste
    Path(OUT_FILE_PATH).mkdir(parents=True, exist_ok=True)

    if mode == "finite":
        run_phase_finite(config_dir)
    elif mode == "steady":
        run_phase_steady(config_dir)
    elif mode == "both":
        # Prima completa TUTTI gli scenari in finite
        run_phase_finite(config_dir)
        # Poi esegue TUTTI gli scenari in steady-state
        run_phase_steady(config_dir)
    else:
        raise SystemExit("Modalità non riconosciuta.")


if __name__ == "__main__":
    main()
