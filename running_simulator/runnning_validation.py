from view.validation_plot import sweep_response_vs_lambda_steady
from model.scenario import Scenario
from pathlib import Path

SEED0 = 1234
DEFAULT_CONFIG_DIR = "config"

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



def run_phase_validation(config_dir: str = DEFAULT_CONFIG_DIR) -> None:
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
