# controller/steady.py
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
from model.ecommerce import EcommerceModel
from model.scenario import Scenario
from engineering.stats import aggregate_vals

CONF_LEVEL = 0.95

def run_batch_means_experiment(
    scn: Scenario,
    seed0: int = 1234,
    *,
    n_batches: int = 64,
    jobs_per_batch: int = 1024,
    label_suffix: str = " (steady)",
) -> Dict[str, Any]:
    """
    Esegue una run lunga (batch means), produce serie per-batch e
    aggrega (mean, stdev, CI 95%) su ciascuna serie.
    """
    model = EcommerceModel(scn, seed=seed0)
    series = model.run_batch_means(n_batches=n_batches, jobs_per_batch=jobs_per_batch)

    agg: Dict[str, Any] = {}
    for name, vals in series.items():
        agg.update(aggregate_vals(name, vals, conf_level=CONF_LEVEL))

    # CSV incrementali per-batch (utile per grafici di stabilizzazione)
    outdir = Path("out")
    scenolab = (scn.name + ("_HL" if getattr(scn, "heavy_load", False) else "")).replace(" ", "_")
    from view.csv_view import write_incremental_batches_csv
    write_incremental_batches_csv(outdir, scenolab, series)

    agg["scenario"] = scn.name + label_suffix
    return agg
