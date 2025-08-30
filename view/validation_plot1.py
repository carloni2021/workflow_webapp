import math
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt

from model.ecommerce import EcommerceModel
# Usa il tuo Scenario già configurato (capacità, service demands, ecc.)
# Esempio: from model.scenario import Scenario
# scn = Scenario(...)

def run_replications_for_lambda(scn, lam, n_reps=1, horizon_s=20_000, seed0=1234, min_completed=100):
    """
    Esegue n_reps repliche finite per un dato λ (req/s) e ritorna (mean_R, stdev_R, n_eff).
    Se in una replica si completano troppo pochi job, la scarta per robustezza.
    """
    rvals = []
    for r in range(n_reps):
        m = EcommerceModel(scn, seed=seed0 + r)
        m.set_arrival_rate(lam)                  # <--- ignora lo Scenario per gli arrivi
        res = m.run_finite(horizon_s=horizon_s)
        if res.get("n_completed", 0) >= min_completed and not math.isnan(res.get("R_mean_s", float("nan"))):
            rvals.append(res["R_mean_s"])

    if len(rvals) == 0:
        return math.nan, math.nan, 0
    mean_R = stats.mean(rvals)
    stdev_R = stats.pstdev(rvals) if len(rvals) > 1 else 0.0  # pstdev per prudenza (o stdev)
    return mean_R, stdev_R, len(rvals)

def sweep_response_vs_lambda(
    scn,
    lam_start=0.5, lam_end=1.2, lam_step=0.1,
    n_reps=1, horizon_s=20_000, seed0=1234
):
    lambdas = np.round(np.arange(lam_start, lam_end + 1e-9, lam_step), 3)
    means, errs, ns = [], [], []
    for lam in lambdas:
        mean_R, stdev_R, n_eff = run_replications_for_lambda(
            scn, lam, n_reps=n_reps, horizon_s=horizon_s, seed0=seed0
        )
        # Half-width 95% ~ 1.96 * s/sqrt(n) (ok per n>=30; senza SciPy niente t-critico)
        hw = (1.96 * stdev_R / math.sqrt(n_eff)) if (n_eff and stdev_R == stdev_R) else math.nan
        means.append(mean_R)
        errs.append(hw)
        ns.append(n_eff)

    # Plot
    plt.figure(figsize=(7.5, 4.5))
    plt.errorbar(lambdas, means, yerr=errs, fmt='-o', capsize=3)
    plt.xlabel("Tasso di arrivo λ (req/s)")
    plt.ylabel("Tempo di risposta medio R (s)")
    plt.title(f"R(λ) con orizzonte finito — n_reps={n_reps}, horizon={horizon_s}s")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return lambdas, means, errs, ns

# ESEMPIO D’USO:
# lambdas, R_mean, R_ci, n_eff = sweep_response_vs_lambda(scn)
