# controller/finite.py
from __future__ import annotations
from typing import Dict, List, Any
from model.ecommerce import EcommerceModel
from model.scenario import Scenario
from engineering.stats import aggregate_vals

N_REPS = 64
CONF_LEVEL = 0.95

def run_experiment_finite(
    scn: Scenario,
    seed0: int = 1234,
    *,
    horizon_s: float,
    label_suffix: str = " (finite)",
) -> Dict[str, Any]:
    """
    Orizzonte finito a repliche: esegue N_REPS repliche indipendenti e
    restituisce metriche aggregate (mean, stdev, CI 95%) per ciascuna metrica.
    """
    results: List[Dict[str, Any]] = []
    for r in range(N_REPS):
        model = EcommerceModel(scn, seed=seed0 + r)
        results.append(model.run_finite(horizon_s=horizon_s))

    agg: Dict[str, Any] = {}
    if results:
        keys = list(results[0].keys())
        for k in keys:
            vals = [res[k] for res in results]
            agg.update(aggregate_vals(k, vals, conf_level=CONF_LEVEL))
    agg["scenario"] = scn.name + label_suffix
    return agg
