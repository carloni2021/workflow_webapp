# main.py
from __future__ import annotations

from pathlib import Path

from model.ecommerce import EcommerceModel
from rndbook.ci_95 import ci95_safe
from view.warmup_plot import plot_convergence_R
from view.validation_plot_R_and_N import sweep_R_and_N_vs_lambda  # <-- nuovo sweep combinato

SEED0 = 1234
DEFAULT_CONFIG_DIR = "config"

# --- import dalla struttura a pacchetti ---
from model.scenario import Scenario
from view.validation_plot2 import sweep_response_vs_lambda_steady            # R(λ) transiente

def _slug(s: str) -> str:
    s = s.lower()
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in s).strip("_")

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

        if scn.name == "1FA (base)" or scn.name == "2FA (base)":
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
        _run_convergence_R_plot_for_scenario(
            scn,
            lam=0.33,
            measure_s=86_400.0,   # finestra di misura (1 giorno)
            warmup_s=0.0,     # warmup escluso dal calcolo delle medie
            seed=SEED0,
            outroot="out",
        )

def run_single_lambda_batch_means(config_dir: str = None,
                                  lam: float = 0.33,
                                  n_batches: int = 64) -> None:
    """
    Esegue il batch-means per UN SOLO λ su tutti gli scenari nella cartella di config.
    - Stima b via ACF-cutoff sui tempi di risposta per-job W_i a tasso 'lam'
    - Esegue batch-means con a=n_batches e b stimato
    - Stampa R̄, N̄ e CI95 + diagnostiche (Lcut, banda, intorno ACF)
    - Stampa tutte le osservazioni per-batch e salva un CSV in out/
    - Stampa le medie (con CI95) delle utilizzazioni U_A, U_B, U_P
    - Stampa le medie sui batch dei tempi di risposta per visita (A1, A2, A3, B, P)
    - Aggiunge al CSV le colonne per-batch dei tempi R_{A1,A2,A3,B,P}

    Parametri:
        config_dir: directory dei file YAML dello scenario (default: DEFAULT_CONFIG_DIR se definito, altrimenti "config")
        lam:        tasso di arrivo (req/s), es. 0.33
        n_batches:  numero di batch, es. 64
    """
    # --- import locali per evitare dipendenze in testa al file ---


    # --- risoluzione cartella config e seed ---
    if config_dir is None:
        try:
            base_dir = DEFAULT_CONFIG_DIR  # se definito nel tuo main
        except NameError:
            base_dir = "config"
    else:
        base_dir = config_dir

    try:
        seed0 = SEED0  # se definito nel tuo main
    except NameError:
        seed0 = 1234

    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)

    def sanitize_name(s: str) -> str:
        # per file name pulito
        for ch in r" /:\<>|?*\"'":
            s = s.replace(ch, "_")
        return s

    print(f"[INFO] batch-means one-λ | config_dir={base_dir} | λ={lam}")

    # --- cerca scenari YAML ---
    from pathlib import Path as _Path
    yaml_files = sorted(_Path(base_dir).glob("*.y*ml"))
    if not yaml_files:
        print(f"[WARN] Nessun YAML trovato in '{base_dir}'.")
        return

    for path in yaml_files:
        scn = Scenario.from_yaml(str(path))
        print(f"[ONE-λ] Scenario: {scn.name}  |  λ={lam:.3f}")

        # 1) run auto (stima b via cutoff ACF + batch-means)
        model = EcommerceModel(scn, seed=seed0)
        series, diag = model.run_batch_means_auto_single_lambda(
            lam=lam, n_batches=n_batches, K=200, n_jobs_calib=50_000, warmup_jobs=5_000
        )

        # 2) riassunto R̄, N̄, CI95
        Rb = series["R_mean_s_batches"]
        Xb = series["X_jobs_per_s_batches"]
        UA = series["U_A_batches"]
        UB = series["U_B_batches"]
        UP = series["U_P_batches"]
        Nb = series.get("N_mean_batches", [r * x for r, x in zip(Rb, Xb)])

        Rm, Rci = ci95_safe(Rb)
        Nm, Nci = ci95_safe(Nb)

        # --- medie (CI95) utilizzazioni richieste ---
        UA_m, UA_ci = ci95_safe(UA)
        UB_m, UB_ci = ci95_safe(UB)
        UP_m, UP_ci = ci95_safe(UP)

        print("  ---- Sintesi batch-means ----")
        print(f"  b={diag['b']} (Lcut={diag['L_cut']}, banda≈{diag['band_95']:.4f}, n_calib={diag['n_calib']})")
        print(f"  R̄={Rm:.4f}  CI95=[{Rci[0]:.4f}, {Rci[1]:.4f}]")
        print(f"  N̄={Nm:.4f}  CI95=[{Nci[0]:.4f}, {Nci[1]:.4f}]")
        print(f"  U_Ā={UA_m:.4f} CI95=[{UA_ci[0]:.4f}, {UA_ci[1]:.4f}] | "
              f"U_B̄={UB_m:.4f} CI95=[{UB_ci[0]:.4f}, {UB_ci[1]:.4f}] | "
              f"U_P̄={UP_m:.4f} CI95=[{UP_ci[0]:.4f}, {UP_ci[1]:.4f}]")

        # 3) diagnostiche ACF attorno al cut-off (se presenti nel diag)
        if all(k in diag for k in ("idx_from", "idx_to", "r_near_cut", "r_near_cut_abs", "band_95", "ok_run", "run", "L_cut")):
            idx_from = diag["idx_from"]; idx_to = diag["idx_to"]
            r_near   = diag["r_near_cut"]
            r_abs    = diag["r_near_cut_abs"]
            band     = diag["band_95"]
            L        = diag["L_cut"]
            ok_run   = diag["ok_run"]
            run_k    = diag["run"]
            print(f"  r[{idx_from}..{idx_to}] = {[f'{v:+.4f}' for v in r_near]}")
            print(f"  |r|[{idx_from}..{idx_to}] = {[f'{v:.4f}' for v in r_abs]}  vs band={band:.4f}")
            print(f"  check run={run_k} da j={L}: {ok_run}")

        # 4) stampa per-batch (rimane invariata)
        print("  ---- Per-batch (idx, R_mean_s, X_jobs_per_s, N_mean, U_A, U_B, U_P) ----")
        for i, (r, x, n, ua, ub, up) in enumerate(zip(Rb, Xb, Nb, UA, UB, UP), start=1):
            print(f"  {i:3d}  {r:9.5f}   {x:9.5f}   {n:9.5f}   {ua:7.4f} {ub:7.4f} {up:7.4f}")

        # 5) export CSV (aggiunte colonne R_A1/A2/A3/B/P per-batch)
        scenario_name = sanitize_name(scn.name)
        csv_path = (outdir / f"batch_means_{scenario_name}_lam{lam:.3f}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("idx,R_mean_s,X_jobs_per_s,N_mean,U_A,U_B,U_P\n")
            for i, (r, x, n, ua, ub, up) in enumerate(
                zip(Rb, Xb, Nb, UA, UB, UP), start=1
            ):
                f.write(f"{i},{r:.10f},{x:.10f},{n:.10f},{ua:.10f},{ub:.10f},{up:.10f}\n")
        print(f"  [saved] {csv_path}")



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

def main() -> None:
    config_dir = DEFAULT_CONFIG_DIR
    print(f"[INFO] finite horizon run - validation study + single lambda study")
    run_phase_finite(config_dir=config_dir)
    print(f"[INFO] batch-means one-λ | λ=0.33")
    run_single_lambda_batch_means(config_dir=config_dir, lam=0.33, n_batches=64)

if __name__ == "__main__":
    main()
