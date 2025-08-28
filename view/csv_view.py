# view/csv_view.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import csv

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_csv_row(csv_path: str, row: Dict[str, object], header_if_new: bool = True) -> None:
    path = Path(csv_path)
    ensure_dir(path.parent)
    write_header = header_if_new and (not path.exists())
    keys = list(row.keys())
    if "scenario" in keys:
        keys = ["scenario"] + [k for k in keys if k != "scenario"]
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in keys})

def write_incremental_batches_csv(outdir: str | Path, scenario_label: str, series: Dict[str, List[float]]) -> None:
    """
    Scrive un unico CSV con medie incrementali sui batch (non sulle repliche).
    Colonne: k, inc_<serie> per ogni chiave in 'series'.
    """
    outdir = Path(outdir)
    ensure_dir(outdir)
    fname = outdir / f"incremental_{scenario_label}_steady.csv"
    keys = list(series.keys())
    if not keys:
        return
    sums = {k: 0.0 for k in keys}
    with fname.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k"] + [f"inc_{k}" for k in keys])
        n_batches = len(series[keys[0]])
        for i in range(n_batches):
            row = [i + 1]
            for k in keys:
                sums[k] += series[k][i]
                row.append(sums[k] / (i + 1))
            w.writerow(row)
