# main.py
from __future__ import annotations
from pathlib import Path
import sys
import argparse

from model.ecommerce import EcommerceModel
from view.warmup_plot import plot_warmup_R
from view.validation_plot_R_and_N import sweep_R_and_N_vs_lambda  # <-- nuovo sweep combinato

SEED0 = 1234
DEFAULT_CONFIG_DIR = "config"

# --- import dalla struttura a pacchetti ---
from model.scenario import Scenario
from view.validation_plot2 import sweep_response_vs_lambda_steady  # opzionale per fase steady


def _slug(s: str) -> str:
    s = s.lower()
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in s).strip("_")


def _frange(start: float, end: float, step: float):
    x = float(start)
    while x <= end + 1e-12:
        yield round(x, 10)
        x = round(x + step, 10)


def _run_warmup_plots_for_scenario(
    scn: Scenario,
    *,
    lam_start: float,
    lam_end: float,
    lam_step: float,
    measure_s: float,
    warmup_s: float,
    bins: int | None = 400,
    seed: int = 1234,
    outroot: str | Path = "out",
) -> None:
    """
    Per ogni λ nella sweep:
      - esegue una run finita (misura = measure_s, warmup = warmup_s)
      - salva PNG con R(t) cumulativo (e opzionalmente per-bin)
    """
    outdir = Path(outroot) / _slug(scn.name)
    outdir.mkdir(parents=True, exist_ok=True)
    scn_slug = _slug(scn.name)

    for lam in _frange(lam_start, lam_end, lam_step):
        model = EcommerceModel(scn, seed=seed)
        model.set_arrival_rate(lam)
        res = model.run_finite(horizon_s=measure_s, warmup_s=warmup_s, bins=bins, verbose=False)

        R_cum = res.get("R_series_cum", [])
        R_bin = res.get("R_series_bin", None)

        title = f"{scn.name} — R(t)  λ={lam:.3f}"
        png = outdir / f"warmup_R_{scn_slug}_lam{lam:.2f}_W{int(warmup_s)}_M{int(measure_s)}.png"

        t_warm = plot_warmup_R(R_cum, R_bin, title=title, outfile=str(png), show=False)
        if t_warm is not None:
            print(f"[OK] Warmup plot salvato: {png}  (t_warm≈{t_warm:.1f}s)")
        else:
            print(f"[OK] Warmup plot salvato: {png}")


# ------------------------------- FINITE --------------------------------------
def run_phase_finite(config_dir: str = DEFAULT_CONFIG_DIR) -> None:
    """
    Caso TRANSIENTE (orizzonte finito):
    - esegue un'unica tornata di repliche per λ e produce:
        * plot R(λ) ± CI95
        * plot N(λ) = X·R ± CI95
    - genera anche il plot R(t) per stimare il warmup (1 run per λ)
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

        # Plot combinato: UNA SOLA tornata di repliche → due PNG (R e N)
        sweep_R_and_N_vs_lambda(
            scn,
            lam_start=0.5, lam_end=1.2, lam_step=0.05,
            n_reps=15,
            measure_s=86_400.0,  # 1 giorno
            warmup_s=8_000.0,
            seed0=SEED0,
            min_completed=100,
            outdir="out",
            save_png=True, save_csv=False, show=False,
        )

        # Plot R(t) vs tempo per stimare warmup (1 run per λ)
        print("[USO] warmup — R(t) vs tempo (1 run per λ)")
        _run_warmup_plots_for_scenario(
            scn,
            lam_start=0.33, lam_end=0.33, lam_step=0.33,
            measure_s=86_400.0,  # finestra di misura (1 giorno)
            warmup_s=0.0,        # warmup escluso dal calcolo delle medie
            bins=400,            # opzionale: solo per la serie per-bin
            seed=SEED0,
            outroot="out",
        )


# ------------------------------- STEADY --------------------------------------
def _preflight_check_steady() -> tuple[bool, str]:
    """
    Verifica minima per la fase steady:
    - esistenza EcommerceModel.run_batch_means
    - esistenza del plotter view.validation_plot_steady (se vuoi i grafici)
    Restituisce (ok, msg_errore_o_vuoto).
    """
    try:
        from model.ecommerce import EcommerceModel  # noqa: F401
    except Exception as e:
        return False, f"[ERRORE] Import di model.ecommerce fallito: {e}"
    else:
        from model.ecommerce import EcommerceModel
        if not hasattr(EcommerceModel, "run_batch_means"):
            return False, "[ERRORE] EcommerceModel.run_batch_means(...) non trovato. Implementalo prima di eseguire la fase 'r'."

    try:
        from view.validation_plot_steady import sweep_response_vs_lambda_steady  # noqa: F401
    except Exception:
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
#commento per evitare questa funzione
''' 
def _choose_mode_via_io() -> tuple[str, str]:

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
'''

def main() -> None:
    config_dir=DEFAULT_CONFIG_DIR
    mode="finite"
    print(f"[INFO] modalità={mode} | config_dir={config_dir}")

    if mode == "steady":
        run_phase_steady(config_dir=config_dir)
    else:
        run_phase_finite(config_dir=config_dir)


#init
if __name__ == "__main__":
    main()
