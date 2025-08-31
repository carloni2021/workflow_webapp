# main.py
from __future__ import annotations
from pathlib import Path
import sys
import argparse

SEED0 = 1234
DEFAULT_CONFIG_DIR = "config"

# --- import dalla struttura a pacchetti ---
from model.scenario import Scenario
from view.validation_plot1 import sweep_response_vs_lambda            # R(λ) transiente
from view.validation_plot_users import sweep_users_vs_lambda          # N(λ) transiente


# ------------------------------- FINITE --------------------------------------
def run_phase_finite(config_dir: str = DEFAULT_CONFIG_DIR) -> None:
    """
    Caso TRANSIENTE (orizzonte finito):
    - genera il plot R(λ) (già presente)
    - genera il plot N(λ) = X·R (utenti medi nel sistema)
    per TUTTI gli scenari nella cartella config_dir.
    """
    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(Path(config_dir).glob("*.y*ml"))
    if not yaml_files:
        print(f"[WARN] Nessun YAML trovato in '{config_dir}'.")
        return

    for path in yaml_files:
        scn = Scenario.from_yaml(str(path))
        print(f"[FINITE] Sweep λ | scenario: {scn.name}")

        # Plot R(λ) (come già facevi)
        sweep_response_vs_lambda(
            scn,
            lam_start=0.5, lam_end=1.2, lam_step=0.05,
            seed0=SEED0,
        )

        # Plot N(λ) = X·R (nuovo)
        sweep_users_vs_lambda(
            scn,
            lam_start=0.5, lam_end=1.2, lam_step=0.05,
            n_reps=15,               # come nel tuo plot R(λ)
            measure_s=86_400.0,      # 1 giorno di misura
            warmup_s=8_000.0,        # warmup
            seed0=SEED0,
            outdir="out", save_png=True, save_csv=False, show=False,
        )


# ------------------------------- STEADY --------------------------------------
def _preflight_check_steady() -> tuple[bool, str]:
    """
    Verifica minima per la fase steady:
    - esistenza EcommerceModel.run_batch_means
    - esistenza del plotter view.validation_plot_steady (se vuoi i grafici)
    Restituisce (ok, msg_errore_o_vuoto).
    """
    # 1) run_batch_means nel modello
    try:
        from model.ecommerce import EcommerceModel  # noqa: F401
    except Exception as e:
        return False, f"[ERRORE] Import di model.ecommerce fallito: {e}"
    else:
        from model.ecommerce import EcommerceModel
        if not hasattr(EcommerceModel, "run_batch_means"):
            return False, "[ERRORE] EcommerceModel.run_batch_means(...) non trovato. Implementalo prima di eseguire la fase 'r'."

    # 2) plotter steady (opzionale, ma utile)
    try:
        from view.validation_plot_steady import sweep_response_vs_lambda_steady  # noqa: F401
    except Exception:
        # non blocchiamo, ma avvisiamo
        return True, "[INFO] Plotter steady mancante (view/validation_plot_steady.py). La fase 'r' proseguirà solo se lo aggiungi."
    return True, ""


def run_phase_steady(config_dir: str = DEFAULT_CONFIG_DIR) -> None:
    """
    Caso STEADY-STATE (batch means):
    Esegue, se disponibili, i plot steady per TUTTI gli scenari.
    Richiede:
      - EcommerceModel.run_batch_means(...)
      - view/validation_plot_steady.py con sweep_response_vs_lambda_steady(...)
    """
    ok, msg = _preflight_check_steady()
    if not ok:
        print(msg)
        return
    if msg:
        print(msg)  # info non bloccante

    # ora possiamo importare il plotter steady
    from view.validation_plot_steady import sweep_response_vs_lambda_steady

    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(Path(config_dir).glob("*.y*ml"))
    if not yaml_files:
        print(f"[WARN] Nessun YAML trovato in '{config_dir}'.")
        return

    for path in yaml_files:
        scn = Scenario.from_yaml(str(path))
        print(f"[STEADY] Sweep λ | scenario: {scn.name}")
        sweep_response_vs_lambda_steady(
            scn,
            lam_start=0.5, lam_end=1.2, lam_step=0.05,
            n_batches=32, jobs_per_batch=2048,
            seed0=SEED0,
            outdir="out", save_png=True, save_csv=False, show=False,
        )


# ------------------------------- ENTRYPOINT ----------------------------------
def _choose_mode_via_io() -> tuple[str, str]:
    """
    Restituisce (mode, config_dir) con mode ∈ {'finite','steady'}.
    Precedence: CLI --mode -> input interattivo -> default 'finite'.
    """
    # CLI
    parser = argparse.ArgumentParser(description="Selezione della fase da eseguire")
    parser.add_argument("--mode", choices=["finite", "steady"], help="Caso da eseguire")
    parser.add_argument("--config", default=DEFAULT_CONFIG_DIR, help="Cartella YAML degli scenari")
    args, _ = parser.parse_known_args()

    if args.mode:
        return args.mode, args.config

    # Interattivo
    if sys.stdin.isatty():
        scelta = input("Vuoi eseguire la fase a orizzonte finito (f) o la fase a regime (r)? (f/r): ").strip().lower()
        mode = "steady" if scelta in ("r", "s", "steady") else "finite"
        return mode, args.config

    # Default non interattivo
    return "finite", args.config


def main() -> None:
    mode, config_dir = _choose_mode_via_io()
    print(f"[INFO] modalità={mode} | config_dir={config_dir}")

    if mode == "steady":
        run_phase_steady(config_dir=config_dir)
    else:
        run_phase_finite(config_dir=config_dir)


if __name__ == "__main__":
    main()
