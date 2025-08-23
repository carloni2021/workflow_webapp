from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


CSV_PATH = Path("out/summary.csv")
PLOTS_DIR = Path("out/plots")


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise SystemExit(f"File CSV non trovato: {csv_path}. Esegui prima main.py per generarlo.")
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise SystemExit(f"Nessun dato nel CSV: {csv_path}")
    return rows


def _num(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _collect_metric(rows: List[Dict[str, str]], key_mean: str, key_stdev: str) -> Tuple[List[str], List[float], List[float]]:
    labels, means, stdevs = [], [], []
    for r in rows:
        labels.append(r.get("scenario", ""))
        means.append(_num(r.get(key_mean, "nan")))
        stdevs.append(_num(r.get(key_stdev, "0")))
    return labels, means, stdevs


def _bar_with_error(labels: List[str], means: List[float], stdevs: List[float], title: str, ylabel: str, fname: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    x = range(len(labels))
    plt.bar(x, means, yerr=stdevs, capsize=5)
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path = PLOTS_DIR / fname
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"✅ salvato: {out_path}")


def main():
    rows = _read_rows(CSV_PATH)

    # 1) Tempo di risposta medio
    labels, m, s = _collect_metric(rows, "R_mean_s_mean", "R_mean_s_stdev")
    _bar_with_error(labels, m, s, "Tempo di risposta medio (R)", "secondi", "R_mean.png")

    # 2) Throughput
    labels, m, s = _collect_metric(rows, "X_jobs_per_s_mean", "X_jobs_per_s_stdev")
    _bar_with_error(labels, m, s, "Throughput (X)", "job/s", "X_mean.png")

    # 3) Utilizzazioni per stazione
    for node, ylabel in [("U_A", "utilizzazione A"), ("U_B", "utilizzazione B"), ("U_P", "utilizzazione P")]:
        labels, m, s = _collect_metric(rows, f"{node}_mean", f"{node}_stdev")
        _bar_with_error(labels, m, s, f"Utilizzazione {node}", "fraq. (0–1)", f"{node}.png")


if __name__ == "__main__":
    main()
