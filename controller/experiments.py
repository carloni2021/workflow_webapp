# controller/experiments.py
from __future__ import annotations
from typing import Dict, Any
from model.scenario import Scenario
from .finite_horizon import run_experiment_finite
from .steady_state import run_batch_means_experiment
from view.csv_view import write_csv_row

def run_and_save_finite(
    scn: Scenario,
    out_csv: str,
    seed0: int = 1234,
    *,
    horizon_s: float,
    label_suffix: str = " (finite)",
) -> Dict[str, Any]:
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
    label_suffix: str = " (steady)",
) -> Dict[str, Any]:
    agg = run_batch_means_experiment(
        scn,
        seed0=seed0,
        n_batches=n_batches,
        jobs_per_batch=jobs_per_batch,
        label_suffix=label_suffix,
    )
    write_csv_row(out_csv, agg, header_if_new=True)
    return agg
