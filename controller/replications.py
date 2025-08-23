from __future__ import annotations
import math
import statistics as stats
from typing import Dict, List
from model.ecommerce import EcommerceModel
from model.scenario import Scenario

def run_experiment(scn: Scenario, seed0: int = 1234) -> Dict[str, float]:
    results: List[Dict[str, float]] = []
    for r in range(scn.replications):
        model = EcommerceModel(scn, seed=seed0 + r)
        results.append(model.run())

    agg: Dict[str, float] = {}
    keys = results[0].keys() if results else []
    for key in keys:
        vals = [res[key] for res in results if not (isinstance(res[key], float) and math.isnan(res[key]))]
        agg[f"{key}_mean"] = stats.mean(vals) if vals else float("nan")
        agg[f"{key}_stdev"] = stats.stdev(vals) if len(vals) > 1 else 0.0
    agg["scenario"] = scn.name + (" +15% load" if scn.heavy_load else "")
    return agg
