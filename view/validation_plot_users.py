# view/validation_plot_users.py
import os, math
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
from model.ecommerce import EcommerceModel

def _slug(s: str | None) -> str:
    if not s:
        return "scenario"
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")

def _ci95_halfwidth(vals: list[float]) -> tuple[float, float, int]:
    """Ritorna (media, halfwidth CI95, n_eff). Se n<2 → hw=0."""
    vals = [v for v in vals if v == v]  # drop NaN
    n = len(vals)
    if n == 0:
        return (math.nan, math.nan, 0)
    m = stats.fmean(vals)
    if n == 1:
        return (m, 0.0, 1)
    sd = stats.stdev(vals)
    hw = 1.96 * sd / math.sqrt(n)
    return (m, hw, n)

def _run_reps_for_lambda(
    scn,
    lam: float,
    lam_index: int,
    *,
    n_reps: int,
    measure_s: float,
    warmup_s: float,
    seed0: int,
) -> tuple[float, float, int]:
    """
    Esegue n_reps repliche finite a tasso λ e calcola N=X·R per replica,
    poi (mean, CI95_halfwidth, n_eff) su quelle N.
    """
    Ns: list[float] = []
    horizon = warmup_s + measure_s
    for r in range(n_reps):
        m = EcommerceModel(scn, seed=seed0 + lam_index * 10_000 + r)
        m.set_arrival_rate(lam)
        res = m.run_finite(horizon_s=horizon, warmup_s=warmup_s, verbose=False)
        R = res.get("R_mean_s")
        X = res.get("X_jobs_per_s")
        if R == R and X == X:
            Ns.append(R * X)  # Little: N = X·R
    return _ci95_halfwidth(Ns)

def sweep_users_vs_lambda(
    scn,
    *,
    lam_start: float = 0.5,
    lam_end: float = 1.2,
    lam_step: float = 0.05,
    n_reps: int = 15,
    measure_s: float = 86_400.0,   # 1 giorno di misura
    warmup_s: float = 8_000.0,     # warmup default come nel tuo plot R(λ)
    seed0: int = 1234,
    outdir: str = "out",
    save_png: bool = True,
    save_csv: bool = False,
    show: bool = False,
):
    """
    Genera il grafico N(λ) con barre d'errore (CI95) usando repliche finite.
    Salva PNG (e opzionalmente CSV) nella cartella out/<scenario>/.
    """
    print("[USO] validation_plot_users — N(λ) con repliche finite")

    scenario_label = getattr(scn, "name", None)
    scenario_slug = _slug(scenario_label)

    lambdas = np.round(np.arange(lam_start, lam_end + 1e-12, lam_step), 3)
    means, errs, ns = [], [], []

    for idx, lam in enumerate(lambdas, start=1):
        meanN, hwN, n_eff = _run_reps_for_lambda(
            scn, lam, lam_index=idx,
            n_reps=n_reps, measure_s=measure_s, warmup_s=warmup_s, seed0=seed0
        )
        means.append(meanN); errs.append(hwN); ns.append(n_eff)
        print(f"[N] λ={lam:.3f}  N̄≈{meanN:.3f}  CI95±{hwN:.3f}  (n={n_eff})")

    # --- plot ---
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(lambdas, means, yerr=errs, fmt="o-", capsize=3, label="N = X·R")
    plt.title(f"{scenario_label} — N(λ)  |  n_reps={n_reps}, warmup={int(warmup_s)}s, measure={int(measure_s)}s")
    plt.xlabel("Tasso di arrivo λ (req/s)")
    plt.ylabel("Utenti medi nel sistema N")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()

    # --- salvataggi ---
    saved_png = saved_csv = None
    if save_png or save_csv:
        outdir_final = os.path.join(outdir, scenario_slug)
        os.makedirs(outdir_final, exist_ok=True)
        tag = f"{scenario_slug}_N_vs_lambda_L{lam_start}-{lam_end}_step{lam_step}_reps{n_reps}"
        if save_png:
            saved_png = os.path.join(outdir_final, f"plot_{tag}.png")
            plt.savefig(saved_png, dpi=150, bbox_inches="tight")
            print(f"[OK] Plot salvato: {saved_png}")
        if save_csv:
            import csv
            saved_csv = os.path.join(outdir_final, f"plot_{tag}.csv")
            with open(saved_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["lambda", "N_mean", "CI95_halfwidth", "n_eff"])
                w.writerows(zip(lambdas, means, errs, ns))
            print(f"[OK] Dati salvati: {saved_csv}")

    if show:
        plt.show()
    else:
        plt.close()
