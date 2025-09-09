from model.scenario import Scenario
from pathlib import Path

import os, re
import numpy as np
import matplotlib.pyplot as plt
from model.ecommerce import EcommerceModel

from rndbook.ci_95 import ci95_hw
from rndbook.rngs import plantSeeds, selectStream, getSeed, putSeed
from rndbook.rng_setup import STREAMS  # {"arrivals":0,"service_A":1,"service_B":2,"service_P":3}

DEFAULT_CONFIG_DIR = "config"
SEED0=1234

# ---------------- utils ----------------
def _slug(txt: str | None) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '-', str(txt or "scenario")).strip('-').lower()

# ------------- snapshot/restore RNG streams ----------------
def _snapshot_streams(stream_names=None) -> dict[str,int]:
    stream_names = list(stream_names) if stream_names is not None else list(STREAMS.keys())
    snap = {}
    for name in stream_names:
        selectStream(STREAMS[name])
        snap[name] = int(getSeed())
    return snap

def _restore_streams(state: dict[str,int]) -> None:
    for name, seed in state.items():
        selectStream(STREAMS[name])
        putSeed(int(seed))


# ------------- core: una sola run per metrica --------------

def _run_reps_for_lambda_R_and_N(
    scn, lam, lam_index, *,
    n_reps, measure_s, warmup_s, seed0,
    min_completed=100,
    reset_streams_between_lambdas=True,
):
    Rs, Ns = [], []
    per_rep_rows = []
    horizon = float(warmup_s) + float(measure_s)

    # 1) inizializza una volta gli stream per questo λ
    seed_base = seed0
    if reset_streams_between_lambdas:
        plantSeeds(seed_base)

    # 2) snapshot iniziale (stato da cui parte la replica r=0)
    stream_state = _snapshot_streams()

    for r in range(n_reps):

        # (A) PRIMA della replica: riparti dallo stato salvato
        _restore_streams(stream_state)

        # (B) reset del SISTEMA: nuovo modello...
        m = EcommerceModel(scn, seed=seed_base)   # fa plantSeeds(...) nel __init__
        m.set_arrival_rate(lam)

        # ...ma SUBITO DOPO ripristina gli stream allo stato voluto
        _restore_streams(stream_state)

        print(f"[Replica {r}] Stream seeds: {stream_state}")

        # (C) esegui la replica
        res = m.run_finite(horizon_s=horizon, warmup_s=warmup_s, verbose=False)
        R = res.get("R_mean_s", float("nan"))
        X = res.get("X_jobs_per_s", float("nan"))
        N = (R * X) if (R == R and X == X) else float("nan")
        n_completed = int(res.get("n_completed", 0))

        # log: seed iniziale ed endpoint per ogni stream
        seed_init = {f"seed_{k}_init": v for k, v in stream_state.items()}

        # (D) FINE replica: salva lo stato degli stream per la prossima
        stream_state = _snapshot_streams()
        seed_end = {f"seed_{k}_end": v for k, v in stream_state.items()}

        per_rep_rows.append({
            "lambda": lam, "rep": r, "seed_base": seed_base,
            **seed_init, **seed_end,
            "R_mean_s": R, "X_jobs_per_s": X, "N_mean": N,
            "n_completed": n_completed, "warmup_s": warmup_s, "measure_s": measure_s,
        })

        if n_completed >= min_completed and (R == R) and (X == X):
            Rs.append(R); Ns.append(N)

    # ritorna (R stats, N stats, righe per-replica con seed logging)
    return ci95_hw(Rs), ci95_hw(Ns), per_rep_rows

# ------------- sweep e plotting separato --------------------

def sweep_R_and_N_vs_lambda(
    scn,
    *,
    lam_start: float = 0.5,
    lam_end: float = 1.2,
    lam_step: float = 0.05,
    n_reps: int = 15,
    measure_s: float = 86_400.0,  # 1 giorno
    warmup_s: float = 8_000.0,
    seed0: int = 1234,
    min_completed: int = 100,
    outdir: str = "out",
    save_png: bool = True,
    save_csv: bool = False,
    show: bool = False,
):
    """
    Esegue un UNICO sweep di repliche per λ e produce:
      - PNG di R(λ) con IC95
      - PNG di N(λ) con IC95
    (CSV opzionali, separati)
    """
    print("[USO] validation_plot — R(λ) e N(λ) con repliche finite (una sola simulazione per metrica)")

    scenario_label = getattr(scn, "name", None)
    scenario_slug = _slug(scenario_label)

    lambdas = np.round(np.arange(lam_start, lam_end + 1e-12, lam_step), 3)

    # serie per R
    R_means, R_errs, R_ns = [], [], []
    # serie per N
    N_means, N_errs, N_ns = [], [], []

    for idx, lam in enumerate(lambdas, start=1):
        (meanR, hwR, n_eff), (meanN, hwN, n_effN), per_rep_rows = _run_reps_for_lambda_R_and_N(
            scn, lam, lam_index=idx,
            n_reps=n_reps, measure_s=measure_s, warmup_s=warmup_s,
            seed0=seed0, min_completed=min_completed
        )

        R_means.append(meanR)
        R_errs.append(hwR)
        R_ns.append(n_eff)
        N_means.append(meanN)
        N_errs.append(hwN)
        N_ns.append(n_effN)

        print(f"[λ={lam:.3f}]  R̄≈{meanR:.4g}  CI95±{hwR:.4g}  (n={n_eff})   |   "
              f"N̄≈{meanN:.4g}  CI95±{hwN:.4g}  (n={n_effN})")

    # ----- plot R(λ) -----
    plt.figure(figsize=(7.5, 4.5))
    plt.errorbar(lambdas, R_means, yerr=R_errs, fmt='-o', capsize=3, label="R mean ± CI95")
    plt.xlabel("Tasso di arrivo λ (req/s)")
    plt.ylabel("Tempo di risposta medio R (s)")
    plt.title(f"{scenario_label} — R(λ)  |  n_reps={n_reps}, warmup={int(warmup_s)}s, measure={int(measure_s)}s")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_png:
        outdir_final = os.path.join(outdir, scenario_slug)
        os.makedirs(outdir_final, exist_ok=True)
        tag = f"{scenario_slug}_R_vs_lambda_L{lam_start}-{lam_end}_step{lam_step}_reps{n_reps}_W{warmup_s}_M{measure_s}"
        r_png = os.path.join(outdir_final, f"plot_{tag}.png")
        plt.savefig(r_png, dpi=150, bbox_inches="tight")
        print(f"[OK] Plot R salvato: {r_png}")
    if show: plt.show()
    plt.close()

    # ----- plot N(λ) -----
    plt.figure(figsize=(7.5, 4.5))
    plt.errorbar(lambdas, N_means, yerr=N_errs, fmt='-o', capsize=3, label="N = X·R ± CI95")
    plt.xlabel("Tasso di arrivo λ (req/s)")
    plt.ylabel("Utenti medi nel sistema N")
    plt.title(f"{scenario_label} — N(λ)  |  n_reps={n_reps}, warmup={int(warmup_s)}s, measure={int(measure_s)}s")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_png:
        outdir_final = os.path.join(outdir, scenario_slug)
        os.makedirs(outdir_final, exist_ok=True)
        tag = f"{scenario_slug}_N_vs_lambda_L{lam_start}-{lam_end}_step{lam_step}_reps{n_reps}_W{warmup_s}_M{measure_s}"
        n_png = os.path.join(outdir_final, f"plot_{tag}.png")
        plt.savefig(n_png, dpi=150, bbox_inches="tight")
        print(f"[OK] Plot N salvato: {n_png}")
    if show: plt.show()
    plt.close()

    # ----- CSV (opzionali, separati) -----
    if save_csv:
        import csv
        outdir_final = os.path.join(outdir, scenario_slug)
        os.makedirs(outdir_final, exist_ok=True)

        tagR = f"{scenario_slug}_R_vs_lambda_L{lam_start}-{lam_end}_step{lam_step}_reps{n_reps}_W{warmup_s}_M{measure_s}"
        r_csv = os.path.join(outdir_final, f"plot_{tagR}.csv")
        with open(r_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lambda", "R_mean_s", "CI95_halfwidth", "n_eff"])
            w.writerows(zip(lambdas, R_means, R_errs, R_ns))
        print(f"[OK] Dati R salvati: {r_csv}")

        tagN = f"{scenario_slug}_N_vs_lambda_L{lam_start}-{lam_end}_step{lam_step}_reps{n_reps}_W{warmup_s}_M{measure_s}"
        n_csv = os.path.join(outdir_final, f"plot_{tagN}.csv")
        with open(n_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lambda", "N_mean", "CI95_halfwidth", "n_eff"])
            w.writerows(zip(lambdas, N_means, N_errs, N_ns))
        print(f"[OK] Dati N salvati: {n_csv}")

    # ritorno utile se vuoi riusare i dati a valle
    return {
        "lambdas": lambdas.tolist(),
        "R": {"mean": R_means, "ci95_hw": R_errs, "n": R_ns},
        "N": {"mean": N_means, "ci95_hw": N_errs, "n": N_ns},
    }

def run_phase_validation(config_dir: str = DEFAULT_CONFIG_DIR) -> None:
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