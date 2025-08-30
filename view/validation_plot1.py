import os, re, math
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
from model.ecommerce import EcommerceModel

def run_replications_for_lambda(scn, lam, n_reps=15, horizon_s=20_000, seed0=1234, min_completed=100):
    rvals = []
    for r in range(n_reps):
        m = EcommerceModel(scn, seed=seed0 + r)
        m.set_arrival_rate(lam)  # override arrivi
        res = m.run_finite(horizon_s=horizon_s)
        if res.get("n_completed", 0) >= min_completed and not math.isnan(res.get("R_mean_s", float("nan"))):
            rvals.append(res["R_mean_s"])
    if not rvals:
        return math.nan, math.nan, 0
    mean_R = stats.mean(rvals)
    stdev_R = stats.pstdev(rvals) if len(rvals) > 1 else 0.0
    return mean_R, stdev_R, len(rvals)

def _slug(txt: str) -> str:
    # semplice slug: lettere/numeri/._- e spazi -> -
    return re.sub(r'[^A-Za-z0-9._-]+', '-', str(txt)).strip('-').lower()

def sweep_response_vs_lambda(
    scn,
    lam_start=0.5, lam_end=1.2, lam_step=0.1,
    n_reps=15, horizon_s=20_000, seed0=1234,
    outdir="out", save_png=True, save_csv=False, show=False,
):
    # nome scenario sicuro
    scenario_label = getattr(scn, "name", None)
    scenario_slug = _slug(scenario_label)

    lambdas = np.round(np.arange(lam_start, lam_end + 1e-9, lam_step), 3)
    means, errs, ns = [], [], []
    for lam in lambdas:
        mean_R, stdev_R, n_eff = run_replications_for_lambda(
            scn, lam, n_reps=n_reps, horizon_s=horizon_s, seed0=seed0
        )
        hw = (1.96 * stdev_R / math.sqrt(n_eff)) if (n_eff and stdev_R == stdev_R) else math.nan
        means.append(mean_R); errs.append(hw); ns.append(n_eff)

    # --- plot ---
    plt.figure(figsize=(7.5, 4.5))
    plt.errorbar(lambdas, means, yerr=errs, fmt='-o', capsize=3)
    plt.xlabel("Tasso di arrivo λ (req/s)")
    plt.ylabel("Tempo di risposta medio R (s)")
    plt.title(f"{scenario_label} — R(λ)  |  n_reps={n_reps}, horizon={horizon_s}s")
    plt.grid(True)
    plt.tight_layout()

    # --- salvataggio ---
    saved_png = saved_csv = None
    if save_png or save_csv:
        # cartella dedicata per scenario
        outdir_final = os.path.join(outdir, scenario_slug)
        os.makedirs(outdir_final, exist_ok=True)
        tag = f"{scenario_slug}_lam{lam_start}-{lam_end}_step{lam_step}_n{n_reps}_H{horizon_s}"
        if save_png:
            saved_png = os.path.join(outdir_final, f"plot_{tag}.png")
            plt.savefig(saved_png, dpi=150, bbox_inches="tight")
            print(f"[OK] Plot salvato: {saved_png}")
        if save_csv:
            import csv
            saved_csv = os.path.join(outdir_final, f"plot{tag}.csv")
            with open(saved_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["lambda", "R_mean_s", "CI95_halfwidth", "n_eff"])
                w.writerows(zip(lambdas, means, errs, ns))
            print(f"[OK] Dati salvati: {saved_csv}")

    if show:
        plt.show()
    else:
        plt.close()

    return
