# sim_runner.py
from __future__ import annotations
from typing import Dict, Optional
from model.scenario import Scenario
from controller.replications import run_experiment_finite, run_batch_means_experiment
from view.reporters import write_csv_row


def run_and_save_finite(
    scn: Scenario,
    out_csv: str,
    seed0: int = 1234,
    *,
    horizon_s: Optional[float] = None,
    label_suffix: str = ""
) -> Dict[str, float]:
    """
    Orizzonte finito (repliche): calcola media, stdev e IC al 95% e scrive una riga nel CSV.
    """
    agg = run_experiment_finite(
        scn,
        seed0=seed0,
        horizon_s=horizon_s,
        label_suffix=label_suffix,
    )
    write_csv_row(out_csv, agg, header_if_new=True)
    return agg


def run_and_save_steady(
    scn: Scenario,
    out_csv: str,
    seed0: int = 1234,
    *,
    n_batches: int = 64,
    jobs_per_batch: int = 1024,
    label_suffix: str = " (steady)"
) -> Dict[str, float]:
    """
    Orizzonte infinito (batch means): esegue una run lunga, divide in batch,
    calcola media, stdev e IC al 95% sulle 64 medie di batch e scrive una riga nel CSV.

    Le colonne avranno forma:
      <serie>_mean, <serie>_stdev, <serie>_ci_low95, <serie>_ci_high95, scenario

    Esempi di <serie>:
      R_mean_s_batches, X_jobs_per_s_batches, U_A_batches, U_B_batches, U_P_batches
    """
    agg = run_batch_means_experiment(
        scn,
        seed0=seed0,
        n_batches=n_batches,
        jobs_per_batch=jobs_per_batch,
        label_suffix=label_suffix,
    )
    write_csv_row(out_csv, agg, header_if_new=True)
    return agg
