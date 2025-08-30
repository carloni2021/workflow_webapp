


import os, re, math
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt

from model.ecommerce import EcommerceModel

# -------- util --------
def _slug(txt: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(txt)).strip("-").lower()

def _ci95_hw_from_batches(values):
    n = len(values)
    if n < 2:
        return float("nan")
    s = stats.stdev(values)            # stdev campionaria sui batch
    return 1.96 * s / (n ** 0.5)       # 95% approx

# -------- sweep steady-state (batch means) --------
def sweep_response_vs_lambda_steady(
    scn,
    lam_start=0.5, lam_end=1.2, lam_step=0.05,
    n_batches=64, jobs_per_batch=1024,
    seed0=1234,
    outdir="out", save_png=True, save_csv=True, show=False,
    scenario_label: str | None = None,
    verbose=True,
):
    """
    Per ogni λ:
      - 1 run lunga con batch-means (n_batches × jobs_per_batch)
      - stima R come media dei batch; barre = half-width IC95% sui batch
    Ritorna: (lambdas, R_mean, R_ci95, saved_png, saved_csv)
    """
    if scenario_label is None:
        scenario_label = getattr(scn, "name", None) or "scenario"
    scenario_slug = _slug(scenario_label)

    lambdas = np.round(np.arange(lam_start, lam_end + 1e-12, lam_step), 2)
    means, hw95 = [], []

    for i, lam in enumerate(lambdas):
        if verbose:
            print(f"[STEADY] λ={lam:.2f}  ({i+1}/{len(lambdas)}) ...", flush=True)

        m = EcommerceModel(scn, seed=seed0 + i)  # seed diverso per λ
        m.set_arrival_rate(lam)

        series = m.run_batch_means(n_batches=n_batches, jobs_per_batch=jobs_per_batch)
        Rs = series["R_mean_s_batches"]              # lista per-batch
        R_bar = stats.mean(Rs) if Rs else float("nan")
        hw = _ci95_hw_from_batches(Rs) if Rs else float("nan")

        means.append(R_bar)
        hw95.append(hw)

    # ---- plot ----
    plt.figure(figsize=(7.5, 4.5))
    plt.errorbar(lambdas, means, yerr=hw95, fmt='-o', capsize=3)
    plt.xlabel("Tasso di arrivo λ (req/s)")
    plt.ylabel("Tempo di risposta medio R (s)")
    plt.title(f"{scenario_label} — R(λ) (batch-means: {n_batches}×{jobs_per_batch})")
    plt.grid(True)
    plt.tight_layout()

    # ---- salvataggio ----
    saved_png = saved_csv = None
    outdir_final = os.path.join(outdir, scenario_slug)
    os.makedirs(outdir_final, exist_ok=True)
    tag = f"{scenario_slug}_lam{lam_start}-{lam_end}_step{lam_step}_bm{n_batches}x{jobs_per_batch}"

    if save_png:
        saved_png = os.path.join(outdir_final, f"R_vs_lambda_STEADY_{tag}.png")
        plt.savefig(saved_png, dpi=150, bbox_inches="tight")
        if verbose: print(f"[OK] Plot salvato: {saved_png}")
    if save_csv:
        import csv
        saved_csv = os.path.join(outdir_final, f"R_vs_lambda_STEADY_{tag}.csv")
        with open(saved_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lambda", "R_mean_s", "CI95_halfwidth"])
            w.writerows(zip(lambdas, means, hw95))
        if verbose: print(f"[OK] Dati salvati: {saved_csv}")

    if show:
        plt.show()
    else:
        plt.close()

    return lambdas, means, hw95, saved_png, saved_csv
