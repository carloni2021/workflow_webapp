import re

from pathlib import Path
from model.ecommerce import EcommerceModel
from model.scenario import Scenario
from view.convergence_plot import plot_convergence_R, plot_convergence_N

DEFAULT_CONFIG_DIR = "config"
SEED0=1234

# ---------------- utils ----------------
def _slug(txt: str | None) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '-', str(txt or "scenario")).strip('-').lower()

# finite horizon run - convergence study
def _run_convergence_R_plot_for_scenario(
    scn: Scenario,
    *,
    lam,
    measure_s: float, warmup_s: float,
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


    model = EcommerceModel(scn, seed=seed)
    model.set_arrival_rate(lam)
    res = model.run_finite(horizon_s=measure_s, warmup_s=warmup_s, verbose=False)

    R_cum = res.get("R_series_cum", [])

    title = f"{scn.name} — R(t)  λ={lam:.3f}"
    png = outdir / f"warmup_R_{scn_slug}_lam{lam:.2f}_W{int(warmup_s)}_M{int(measure_s)}.png"

    t_warm = plot_convergence_R(R_cum, title=title, lam=lam, scn=scn, outfile=str(png), show=False)

    if t_warm is not None:
        print(f"[OK] Warmup plot salvato: {png}  (t_warm≈{t_warm:.1f}s)")
    else:
        print(f"[OK] Warmup plot salvato: {png}")

<<<<<<< Updated upstream
def run_phase_convergence(config_dir: str = DEFAULT_CONFIG_DIR) -> None:
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

        # Plot R(t) vs tempo per stimare warmup (1 run per λ)
        print("[USO] warmup — R(t) vs tempo (1 run per λ)")
        _run_convergence_R_plot_for_scenario(
            scn,
            lam=0.33,
            measure_s=86_400.0,   # finestra di misura (1 giorno)
            warmup_s=0.0,     # warmup escluso dal calcolo delle medie
            seed=SEED0,
            outroot="out",
        )
        # Plot N(t) vs tempo con stessa logica (1 run per λ)   # <— NEW
        print("[USO] warmup — N(t) vs tempo (1 run per λ)")
        _run_convergence_N_plot_for_scenario(
            scn,
            lam=0.33,
            measure_s=86_400.0,  # 1 giorno
            warmup_s=0.0,
            seed=SEED0,
            outroot="out",
        )

def _run_convergence_N_plot_for_scenario(
=======
#def run_phase_convergence(config_dir: str = DEFAULT_CONFIG_DIR) -> None:
#    """
#    Caso TRANSIENTE (orizzonte finito):
#    - esegue un'unica tornata di repliche per λ e produce:
#        * plot R(λ) ± CI95
#        * plot N(λ) = X·R ± CI95
#    - genera anche il plot R(t) per stimare il warmup (1 run per λ)
#    per TUTTI gli scenari nella cartella config_dir.
#    """
#    outdir = Path("out")
#    outdir.mkdir(parents=True, exist_ok=True)
#
#    yaml_files = sorted(Path(config_dir).glob("*.y*ml"))
#    if not yaml_files:
#        print(f"[WARN] Nessun YAML trovato in '{config_dir}'.")
#        return
#
#    for path in yaml_files:
#        scn = Scenario.from_yaml(str(path))
#
#        # Plot R(t) vs tempo per stimare warmup (1 run per λ)
#        print("[USO] warmup — R(t) vs tempo (1 run per λ)")
#        _run_convergence_R_plot_for_scenario(
#            scn,
#            lam=0.33,
#            measure_s=86_400.0,   # finestra di misura (1 giorno)
#            warmup_s=0.0,     # warmup escluso dal calcolo delle medie
#            seed=SEED0,
#            outroot="out",
#        )
#        # Plot N(t) vs tempo con stessa logica (1 run per λ)   # <— NEW
#        print("[USO] warmup — N(t) vs tempo (1 run per λ)")
#        _run_convergence_N_plot_for_scenario(
#            scn,
#            lam=0.33,
#            measure_s=86_400.0,  # 1 giorno
#            warmup_s=0.0,
#            seed=SEED0,
#            outroot="out",
#        )
#
def _run_convergence_N_plot_for_scenario(  # <— NEW
>>>>>>> Stashed changes
        scn: Scenario,
        *,
        lam,
        measure_s: float, warmup_s: float,
        seed: int = 1234,
        outroot: str | Path = "out",
) -> None:
    """
    Per ogni λ nella sweep:
      - esegue una run finita (misura = measure_s, warmup = warmup_s)
      - salva PNG con N(t) cumulativo (e linea teorica solo se tutto exp)
      - se la serie N cumulativa non è disponibile, usa Little: N(t)=λ·R(t)
    """
    outdir = Path(outroot) / _slug(scn.name)
    outdir.mkdir(parents=True, exist_ok=True)
    scn_slug = _slug(scn.name)

    model = EcommerceModel(scn, seed=seed)
    model.set_arrival_rate(lam)
    res = model.run_finite(horizon_s=measure_s, warmup_s=warmup_s, verbose=False)

    N_cum = res.get("N_series_cum", [])
    if not N_cum:
        R_cum = res.get("R_series_cum", [])
        if R_cum:
            N_cum = [(t, lam * y) for (t, y) in R_cum]  # Little (fallback)

    title = f"{scn.name} — N(t)  λ={lam:.3f}"
    png = outdir / f"warmup_N_{scn_slug}_lam{lam:.2f}_W{int(warmup_s)}_M{int(measure_s)}.png"

    t_warm = plot_convergence_N(N_cum, title=title, lam=lam, scn=scn, outfile=str(png), show=False)

    if t_warm is not None:
        print(f"[OK] Warmup plot salvato: {png}  (t_warm≈{t_warm:.1f}s)")
    else:
        print(f"[OK] Warmup plot salvato: {png}")
#------------------------------- FUNZIONI AUSILIARIE ----------------------------------
# ✨ 1) NUOVI IMPORT
from view.convergence_plot import (
    plot_convergence_R_multi_via_runner,
    plot_convergence_N_multi_via_runner,
)
# ✨ 2) DEFAULT SEEDS PER L’OVERLAY
SEEDS = (101, 102, 103, 104, 105)

# ✨ 3) RUNNER FACTORY: una run per seed -> serie cumulativa
def _make_runner_R(lam, measure_s, warmup_s):
    def _runner(scn, seed):
        model = EcommerceModel(scn, seed=seed)
        model.set_arrival_rate(lam)
        res = model.run_finite(horizon_s=measure_s, warmup_s=warmup_s, verbose=False)
        return res.get("R_series_cum", [])
    return _runner

def _make_runner_N(lam, measure_s, warmup_s):
    def _runner(scn, seed):
        model = EcommerceModel(scn, seed=seed)
        model.set_arrival_rate(lam)
        res = model.run_finite(horizon_s=measure_s, warmup_s=warmup_s, verbose=False)
        N_cum = res.get("N_series_cum", [])
        if not N_cum:
            R_cum = res.get("R_series_cum", [])
            N_cum = [(t, lam * y) for (t, y) in R_cum]  # fallback Little
        return N_cum
    return _runner

# ✨ 4) FUNZIONE CHE CREA I DUE OVERLAY (R e N) PER UNO SCENARIO
def _run_convergence_overlay_plots_for_scenario(
    scn: Scenario,
    *,
    lam,
    measure_s: float,
    warmup_s: float,
    seeds=SEEDS,
    outroot: str | Path = "out",
) -> None:
    outdir = Path(outroot) / _slug(scn.name)
    outdir.mkdir(parents=True, exist_ok=True)
    scn_slug = _slug(scn.name)

    # --- Overlay R(t)
    title_R = f"{scn.name} — R(t) overlay  λ={lam:.3f}"
    png_R = outdir / f"warmup_R_multi_{scn_slug}_lam{lam:.2f}_W{int(warmup_s)}_M{int(measure_s)}.png"
    plot_convergence_R_multi_via_runner(
        _make_runner_R(lam, measure_s, warmup_s),
        seeds=seeds,
        lam=lam,
        scn=scn,
        title=title_R,
        outfile=str(png_R),
        show=False,
    )
    print(f"[OK] Overlay R salvato: {png_R}")

    # --- Overlay N(t)
    title_N = f"{scn.name} — N(t) overlay  λ={lam:.3f}"
    png_N = outdir / f"warmup_N_multi_{scn_slug}_lam{lam:.2f}_W{int(warmup_s)}_M{int(measure_s)}.png"
    plot_convergence_N_multi_via_runner(
        _make_runner_N(lam, measure_s, warmup_s),
        seeds=seeds,
        lam=lam,
        scn=scn,
        title=title_N,
        outfile=str(png_N),
        show=False,
    )
    print(f"[OK] Overlay N salvato: {png_N}")

# ✨ 5) MODIFICA DENTRO run_phase_convergence: usa l’overlay (puoi lasciare anche i singoli)
def run_phase_convergence(config_dir: str = DEFAULT_CONFIG_DIR) -> None:
    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(Path(config_dir).glob("*.y*ml"))
    if not yaml_files:
        print(f"[WARN] Nessun YAML trovato in '{config_dir}'.")
        return

    for path in yaml_files:
        scn = Scenario.from_yaml(str(path))

        lam = 0.33
        measure_s = 86_400.0  # 1 giorno
        warmup_s = 0.0

        # (opzionale) mantieni il vecchio plot singolo:
        # print("[USO] warmup — R(t) vs tempo (1 run per λ)")
        # _run_convergence_R_plot_for_scenario(scn, lam=lam, measure_s=measure_s, warmup_s=warmup_s, seed=SEED0, outroot="out")
        # print("[USO] warmup — N(t) vs tempo (1 run per λ)")
        # _run_convergence_N_plot_for_scenario(scn, lam=lam, measure_s=measure_s, warmup_s=warmup_s, seed=SEED0, outroot="out")

        # 🔵 NUOVO: overlay 5 seed nello stesso plot
        print("[USO] warmup — overlay R(t), N(t) (5 seed)")
        _run_convergence_overlay_plots_for_scenario(
            scn,
            lam=lam,
            measure_s=measure_s,
            warmup_s=warmup_s,
            seeds=SEEDS,
            outroot="out",
        )

