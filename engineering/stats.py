# controller/stats.py
from __future__ import annotations
import math
import statistics as stats
from typing import Dict, List

# Half-width t-Student: IC = mean ± HW
def halfwidth_t(vals: List[float], conf_level: float = 0.95) -> float:
    n = len(vals)
    if n < 2:
        return float("nan")
    from rndbook.rvms import idfStudent
    tcrit = idfStudent(n - 1, 0.5 * (1.0 + conf_level))
    # formula standard HW = t * s / sqrt(n)
    return tcrit * stats.stdev(vals) / math.sqrt(n)

def aggregate_vals(key: str, vals: List[float], conf_level: float = 0.95) -> Dict[str, float]:
    """
    Aggrega una lista di float in media, stdev e IC al livello di confidenza dato.
    Restituisce un dict con chiavi f"{key}_mean", f"{key}_stdev", f"{key}_ci_low95", f"{key}_ci_high95".
    """
    clean = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return {
            f"{key}_mean": float("nan"),
            f"{key}_stdev": 0.0,
            f"{key}_ci_low95": float("nan"),
            f"{key}_ci_high95": float("nan"),
        }
    m = stats.mean(clean)
    s = stats.stdev(clean) if len(clean) > 1 else 0.0
    h = halfwidth_t(clean, conf_level) if len(clean) > 1 else float("nan")
    return {
        f"{key}_mean": m,
        f"{key}_stdev": s,
        f"{key}_ci_low95": (m - h) if not math.isnan(h) else float("nan"),
        f"{key}_ci_high95": (m + h) if not math.isnan(h) else float("nan"),
    }
