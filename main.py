# main.py
from __future__ import annotations
from pathlib import Path

from view.validation_plot1 import sweep_response_vs_lambda
from controller.experiments import run_and_save_steady
from model.scenario import Scenario

# File di output CSV
OUT_CSV_FINITE = "out/summary_finite.csv"
OUT_FILE_PATH = "out/"  # Usato solo per creare la cartella

SEED0 = 1234 # Seed base per tutte le simulazioni


"""Esegue la fase a regime (batch means) per TUTTI gli scenari."""
def run_phase_steady(config_dir: str = "config") -> None:
    # Esegue la fase a regime (batch means) per TUTTI gli scenari.
    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = str(outdir / "summary_steady.csv")

    yaml_files = sorted(Path(config_dir).glob("*.y*ml"))
    if not yaml_files:
        print(f"[WARN] Nessun YAML trovato in '{config_dir}'.")
        return

    for path in yaml_files:
        scn = Scenario.from_yaml(str(path))
        print(f"[STEADY] Batch means | scenario: {scn.name}")
        agg = run_and_save_steady(
            scn,
            out_csv=out_csv,
            seed0=1234,
            n_batches=32,
            jobs_per_batch=2048,
            label_suffix=" (steady)",
        )
        print("  -> mean:", {k: v for k, v in agg.items() if k.endswith("_mean")})


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
    #In base all'input dell'utente, esegue la fase finita o quella a regime

    #ask for user input
    user_input = input("Vuoi eseguire la fase a orizzonte finito (f) o la fase a regime (r)? (f/r): ").strip().lower()
    if user_input == 'f':
        run_phase_finite(config_dir)
    elif user_input == 'r':
        run_phase_steady(config_dir)
    else:
        print("Input non valido. Eseguito il default: fase a orizzonte finito.")
        run_phase_finite(config_dir)

if __name__ == "__main__":
    main()
