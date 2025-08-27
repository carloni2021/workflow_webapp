from __future__ import annotations
import math, statistics as stats, csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from rndbook.rvms import idfStudent
from model.ecommerce import EcommerceModel
from model.scenario import Scenario

N_REPS = 64
CONF_LEVEL = 0.95


def _aggregate_vals(key: str, vals: List[float], conf_level: float = 0.95) -> Dict[str, float]:
    """
    Aggrega una lista di float in media, stdev e IC al livello di confidenza dato.
    Restituisce un dict con chiavi f"{key}_mean", f"{key}_stdev", f"{key}_ci_low95", f"{key}_ci_high95".
    """
    out: Dict[str, float] = {}

    # Filtra eventuali NaN
    clean = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]

    if not clean:
        out[f"{key}_mean"] = float("nan")
        out[f"{key}_stdev"] = 0.0
        out[f"{key}_ci_low95"] = float("nan")
        out[f"{key}_ci_high95"] = float("nan")
        return out

    m = stats.mean(clean)
    s = stats.stdev(clean) if len(clean) > 1 else 0.0
    h = _halfwidth_estimate_py(clean, loc=conf_level) if len(clean) > 1 else float("nan")

    out[f"{key}_mean"] = m
    out[f"{key}_stdev"] = s
    out[f"{key}_ci_low95"] = m - h if not math.isnan(h) else float("nan")
    out[f"{key}_ci_high95"] = m + h if not math.isnan(h) else float("nan")
    return out

def run_experiment_finite(
    scn: Scenario,
    seed0: int = 1234,
    *,
    horizon_s: Optional[float] = None,
    label_suffix: str = " (finite)"
) -> Dict[str, Any]:
    """
    Orizzonte finito con metodo delle repliche (N_REPS=64).
    Aggrega media, stdev e IC al 95% sulle repliche.
    """
    results: List[Dict[str, Any]] = []
    for r in range(N_REPS):
        model = EcommerceModel(scn, seed=seed0 + r)
        results.append(model.run_finite(horizon_s=horizon_s))

    agg: Dict[str, Any] = {}
    if not results:
        agg["scenario"] = scn.name + label_suffix
        return agg

    keys = list(results[0].keys())
    for k in keys:
        vals = [res[k] for res in results if not (isinstance(res[k], float) and math.isnan(res[k]))]
        agg.update(_aggregate_vals(k, vals, conf_level=CONF_LEVEL))

    agg["scenario"] = scn.name + label_suffix
    return agg


def _halfwidth_estimate_py(values: List[float], loc: float = CONF_LEVEL) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    stdev = stats.stdev(values)               # campionaria (n-1)
    u = 0.5 * (1.0 + loc)                     # es. 0.975
    tcrit = idfStudent(n - 1, u)
    return tcrit * stdev / math.sqrt(n - 1)   # coerente con estimate.py

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _write_incremental_batches_csv(outdir: Path, scenario_label: str, series: Dict[str, List[float]]):
    """
    Scrive un unico CSV con medie incrementali sui batch (non sulle repliche).
    Colonne: k, inc_<serie> per ogni chiave in 'series'.
    """
    _ensure_dir(outdir)
    fname = outdir / f"incremental_{scenario_label}_steady.csv"

    keys = list(series.keys())
    inc = {k: [] for k in keys}
    sums = {k: 0.0 for k in keys}
    counts = 0

    # numero di batch = lunghezza comune (le nostre serie sono di pari lunghezza)
    n_batches = len(series[keys[0]]) if keys else 0

    with fname.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k"] + [f"inc_{k}" for k in keys])
        for i in range(n_batches):
            counts += 1
            row = [counts]
            for k in keys:
                sums[k] += series[k][i]
                inc[k].append(sums[k] / counts)
                row.append(inc[k][-1])
            w.writerow(row)

def run_batch_means_experiment(
    scn: Scenario,
    seed0: int = 1234,
    *,
    n_batches: int = 64,
    jobs_per_batch: int = 1024,
    label_suffix: str = " (steady)"
) -> Dict[str, Any]:
    """
    Esegue UNA run lunga (batch means) e aggrega le metriche su 64 batch.
    Ritorna: <metric>_mean, _stdev, _ci_low95, _ci_high95, scenario
    e scrive un CSV con le medie incrementali sui batch.
    """
    model = EcommerceModel(scn, seed=seed0)
    series = model.run_batch_means(n_batches=n_batches, jobs_per_batch=jobs_per_batch)

    # Aggregazione "media delle medie di batch"
    agg: Dict[str, Any] = {}
    for name, vals in series.items():
        agg.update(_aggregate_vals(name, vals, conf_level=CONF_LEVEL))

    # CSV delle medie incrementali per batch (comodo per i grafici di stabilizzazione)
    outdir = Path("out")
    scenolab = (scn.name + ("_HL" if scn.heavy_load else "")).replace(" ", "_")
    _write_incremental_batches_csv(outdir, scenolab, series)

    agg["scenario"] = scn.name + label_suffix
    return agg
