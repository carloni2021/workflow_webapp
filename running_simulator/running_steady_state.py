from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import re  # <--- IMPORT NECESSARIO per generare i nomi corretti

from model.ecommerce import EcommerceModel
from model.scenario import Scenario
from rndbook.ci_95 import ci95_safe

DEFAULT_CONFIG_DIR = "config"
SEED0 = 1234


def run_single_lambda_batch_means(config_dir: str = None,
                                  n_batches: int = 64) -> None:
    if config_dir is None:
        try:
            base_dir = DEFAULT_CONFIG_DIR
        except NameError:
            base_dir = "config"
    else:
        base_dir = config_dir

    try:
        seed_val = SEED0
    except NameError:
        seed_val = 1234

    # Cartella root
    outroot = Path("out")
    outroot.mkdir(parents=True, exist_ok=True)

    # --- FUNZIONE SLUG CORRETTA (Usa i trattini '-' per coincidere con le cartelle) ---
    def _slug(txt: str | None) -> str:
        # Trasforma "1FA (base)" in "1fa-base"
        return re.sub(r'[^A-Za-z0-9._-]+', '-', str(txt or "scenario")).strip('-').lower()

    def _plot_batch_convergence(data_batches, metric_name, unit, scn_name, lam, filename, color_main):
        arr = np.array(data_batches)
        n = len(arr)
        indices = np.arange(1, n + 1)

        cum_means = np.cumsum(arr) / indices

        cum_cis = []
        for k in indices:
            if k < 2:
                cum_cis.append(0.0)
                continue
            sample = arr[:k]
            sem = np.std(sample, ddof=1) / np.sqrt(k)
            h = sem * st.t.ppf((1 + 0.95) / 2., k - 1)
            cum_cis.append(h)
        cum_cis = np.array(cum_cis)

        plt.figure(figsize=(10, 6))
        plt.scatter(indices, arr, color='gray', alpha=0.3, s=15, label=f'Single Batch {metric_name}')
        plt.plot(indices, cum_means, color=color_main, linewidth=2,
                 label=rf'Cumulative Mean $\overline{{{metric_name}}}$')
        plt.fill_between(indices, cum_means - cum_cis, cum_means + cum_cis, color=color_main, alpha=0.2, label='95% CI')

        plt.title(f"Batch Means Convergence ({metric_name}) — {scn_name} (λ={lam:.3f})")
        plt.xlabel("Number of Batches processed")
        plt.ylabel(f"{metric_name} [{unit}]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  [plot] {filename}")

    yaml_files = sorted(Path(base_dir).glob("*.y*ml"))
    if not yaml_files:
        print(f"[WARN] Nessun YAML trovato in '{base_dir}'.")
        return

    for path in yaml_files:
        scn = Scenario.from_yaml(str(path))
        lam = 1.0 / (float(scn.get_interarrival_mean()))
        print(f"\n[ONE-λ BATCH] Scenario: {scn.name}  |  λ={lam:.3f}")

        # --- CREAZIONE/SELEZIONE CARTELLA SPECIFICA SCENARIO ---
        # Usa _slug per ottenere nomi tipo "1fa-base" invece di "1FA (base)"
        scenario_slug = _slug(scn.name)
        scenario_outdir = outroot / scenario_slug
        scenario_outdir.mkdir(parents=True, exist_ok=True)
        # ---------------------------------------------

        model = EcommerceModel(scn, seed=seed_val)
        series, diag = model.run_batch_means_auto_single_lambda(
            lam=lam, n_batches=n_batches, K=200, n_jobs_calib=50_000, warmup_jobs=5_000
        )

        Rb = series["R_mean_s_batches"]
        Xb = series["X_jobs_per_s_batches"]
        UA = series["U_A_batches"]
        UB = series["U_B_batches"]
        UP = series["U_P_batches"]
        Nb = series.get("N_mean_batches", [r * x for r, x in zip(Rb, Xb)])

        Rm, Rci = ci95_safe(Rb)
        Nm, Nci = ci95_safe(Nb)
        UA_m, UA_ci = ci95_safe(UA)
        UB_m, UB_ci = ci95_safe(UB)
        UP_m, UP_ci = ci95_safe(UP)

        print("  ---- Sintesi batch-means ----")
        print(f"  b={diag.get('b')} (Lcut={diag.get('L_cut')}, banda≈{diag.get('band_95', 0):.4f})")
        print(f"  R̄={Rm:.4f}  CI95=[{Rci[0]:.4f}, {Rci[1]:.4f}]")

        if all(k in diag for k in ("idx_from", "idx_to", "r_near_cut")):
            idx_from = diag["idx_from"]
            idx_to = diag["idx_to"]
            r_near = diag["r_near_cut"]
            print(f"  r[{idx_from}..{idx_to}] = {[f'{v:+.4f}' for v in r_near]}")

        # --- SALVATAGGIO GRAFICI NELLA CARTELLA SPECIFICA ---
        # I file verranno salvati dentro out/1fa-base/BM_R_...
        plot_R_path = scenario_outdir / f"BM_R_convergence_{scenario_slug}_lam{lam:.3f}.png"
        _plot_batch_convergence(Rb, "R", "s", scn.name, lam, plot_R_path, "tab:blue")

        plot_N_path = scenario_outdir / f"BM_N_convergence_{scenario_slug}_lam{lam:.3f}.png"
        _plot_batch_convergence(Nb, "N", "-", scn.name, lam, plot_N_path, "tab:orange")

        # --- SALVATAGGIO CSV NELLA CARTELLA SPECIFICA ---
        csv_path = (scenario_outdir / f"batch_means_{scenario_slug}_lam{lam:.3f}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("idx,R_mean_s,X_jobs_per_s,N_mean,U_A,U_B,U_P\n")
            for i, (r, x, n, ua, ub, up) in enumerate(zip(Rb, Xb, Nb, UA, UB, UP), start=1):
                f.write(f"{i},{r:.10f},{x:.10f},{n:.10f},{ua:.10f},{ub:.10f},{up:.10f}\n")
        print(f"  [saved] {csv_path}")