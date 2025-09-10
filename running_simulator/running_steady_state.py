from pathlib import Path
from model.ecommerce import EcommerceModel, Scenario
from rndbook.ci_95 import ci95_safe

DEFAULT_CONFIG_DIR = "config"
SEED0=1234

def run_single_lambda_batch_means(config_dir: str = None,
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


    # --- cerca scenari YAML ---
    from pathlib import Path as _Path
    yaml_files = sorted(_Path(base_dir).glob("*.y*ml"))
    if not yaml_files:
        print(f"[WARN] Nessun YAML trovato in '{base_dir}'.")
        return

    for path in yaml_files:

        scn = Scenario.from_yaml(str(path))
        lam = 1.0 / (float(scn.get_interarrival_mean()))
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