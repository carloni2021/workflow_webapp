from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import matplotlib.pyplot as plt

CSV_PATH = Path("out/summary.csv")
CFG_DIR = Path("config")
PLOTS_DIR = Path("out/plots")


# -----------------------
# Letture e utilità base
# -----------------------
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
    plt.figure(figsize=(11, 5))
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


# -----------------------
# Bound: ricavo dai YAML
# -----------------------
def _load_bounds_from_yaml(cfg_dir: Path) -> Dict[str, float]:
    """
    Per ogni YAML in config/, calcolo:
      D_A_tot = A.Class1 + A.Class2 + A.Class3   (3 visite in A)
      D_B_tot = B.Class1                          (1 visita in B)
      D_P_tot = P.Class2                          (1 visita in P)
      X_bound = 1 / max(D_A_tot, D_B_tot, D_P_tot)
    Ritorna: {scenario_name -> X_bound}
    """
    bounds: Dict[str, float] = {}
    for yml in sorted(cfg_dir.glob("*.yaml")):
        try:
            with open(yml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            name = data["name"]
            sd = data["service_demands"]
            d_a = float(sd["A"].get("Class1", 0)) + float(sd["A"].get("Class2", 0)) + float(sd["A"].get("Class3", 0))
            d_b = float(sd["B"].get("Class1", 0))
            d_p = float(sd["P"].get("Class2", 0))
            d_max = max(d_a, d_b, d_p) if all(not math.isnan(v) for v in (d_a, d_b, d_p)) else float("nan")
            x_bound = 1.0 / d_max if d_max and d_max > 0 else float("nan")
            bounds[name] = x_bound
        except Exception:
            # YAML non conforme? continuo
            continue
    return bounds


# -----------------------
# Plot specifici richiesti
# -----------------------
def plot_r_mean(rows: List[Dict[str, str]]):
    labels, m, s = _collect_metric(rows, "R_mean_s_mean", "R_mean_s_stdev")
    _bar_with_error(labels, m, s, "Tempo di risposta medio (R)", "secondi", "R_mean_ext.png")


def plot_x_vs_bound(rows: List[Dict[str, str]], bounds: Dict[str, float]):
    # preparo coppie (X_sim, X_bound) per scenari con bound disponibile
    labels, x_sim, x_bound = [], [], []
    for r in rows:
        name = r.get("scenario", "")
        if name in bounds:
            labels.append(name)
            x_sim.append(_num(r.get("X_jobs_per_s_mean", "nan")))
            x_bound.append(bounds[name])

    # disegno barre affiancate
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    x = range(len(labels))
    width = 0.42
    plt.bar([i - width/2 for i in x], x_sim, width=width, label="Throughput simulato")
    plt.bar([i + width/2 for i in x], x_bound, width=width, label="Throughput bound (teorico)")
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.title("Throughput: simulato vs bound teorico")
    plt.ylabel("job/s")
    plt.legend()
    plt.tight_layout()
    out_path = PLOTS_DIR / "X_vs_bound.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"✅ salvato: {out_path}")


def plot_u_grouped(rows: List[Dict[str, str]]):
    labels = [r.get("scenario", "") for r in rows]
    uA = [_num(r.get("U_A_mean", "nan")) for r in rows]
    uB = [_num(r.get("U_B_mean", "nan")) for r in rows]
    uP = [_num(r.get("U_P_mean", "nan")) for r in rows]

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    x = range(len(labels))
    width = 0.25
    plt.bar([i - width for i in x], uA, width=width, label="U_A")
    plt.bar([i for i in x],        uB, width=width, label="U_B")
    plt.bar([i + width for i in x], uP, width=width, label="U_P")
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.title("Utilizzazione per stazione (raggruppata)")
    plt.ylabel("fraq. (0–1)")
    plt.legend()
    plt.tight_layout()
    out_path = PLOTS_DIR / "U_grouped.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"✅ salvato: {out_path}")


# helper per filtri scenario
def _is_base(name: str) -> bool:
    return "heavy" not in name and "Bx2" not in name

def _is_heavy(name: str) -> bool:
    return "heavy" in name.lower()

def _is_bx2(name: str) -> bool:
    return "Bx2" in name

def _is_1fa(name: str) -> bool:
    return "1FA" in name

def _is_2fa(name: str) -> bool:
    return "2FA" in name


def plot_compare_1fa_2fa(rows: List[Dict[str, str]]):
    # prendo solo baseline (no heavy, no Bx2)
    base = [r for r in rows if _is_base(r.get("scenario", ""))]
    # ordino per dare precedenza a 1FA poi 2FA
    base.sort(key=lambda r: (0 if _is_1fa(r.get("scenario","")) else 1, r.get("scenario","")))

    labels = [r.get("scenario","") for r in base if _is_1fa(r.get("scenario","")) or _is_2fa(r.get("scenario",""))]
    r_vals = [_num(r.get("R_mean_s_mean","nan")) for r in base if _is_1fa(r.get("scenario","")) or _is_2fa(r.get("scenario",""))]
    x_vals = [_num(r.get("X_jobs_per_s_mean","nan")) for r in base if _is_1fa(r.get("scenario","")) or _is_2fa(r.get("scenario",""))]

    # R e X in due assi (due grafici separati per semplicità)
    _bar_with_error(labels, r_vals, [0]*len(labels), "Confronto 1FA vs 2FA — R medio", "secondi", "R_X_1FA_vs_2FA_R.png")
    _bar_with_error(labels, x_vals, [0]*len(labels), "Confronto 1FA vs 2FA — Throughput", "job/s", "R_X_1FA_vs_2FA_X.png")


def plot_compare_base_vs_heavy(rows: List[Dict[str, str]]):
    # per famiglia (1FA/2FA) affianco base e heavy
    groups = []
    for fam in ["1FA", "2FA"]:
        base = next((r for r in rows if _is_base(r.get("scenario","")) and fam in r.get("scenario","")), None)
        heavy = next((r for r in rows if _is_heavy(r.get("scenario","")) and fam in r.get("scenario","")), None)
        if base and heavy:
            groups.append((f"{fam} base", base, f"{fam} heavy", heavy))

    # R
    labels, vals = [], []
    for (l1, b, l2, h) in groups:
        labels.extend([l1, l2])
        vals.extend([_num(b.get("R_mean_s_mean","nan")), _num(h.get("R_mean_s_mean","nan"))])
    _bar_with_error(labels, vals, [0]*len(labels), "Base vs Heavy — R medio", "secondi", "R_base_vs_heavy.png")

    # X
    labels, vals = [], []
    for (l1, b, l2, h) in groups:
        labels.extend([l1, l2])
        vals.extend([_num(b.get("X_jobs_per_s_mean","nan")), _num(h.get("X_jobs_per_s_mean","nan"))])
    _bar_with_error(labels, vals, [0]*len(labels), "Base vs Heavy — Throughput", "job/s", "X_base_vs_heavy.png")


def plot_compare_bx2(rows: List[Dict[str, str]]):
    # confronto U_k tra base e Bx2 per famiglie
    def find_pair(fam: str):
        base = next((r for r in rows if _is_base(r.get("scenario","")) and fam in r.get("scenario","")), None)
        bx2  = next((r for r in rows if _is_bx2(r.get("scenario","")) and fam in r.get("scenario","")), None)
        return base, bx2

    pairs = []
    for fam in ["1FA", "2FA"]:
        b, x2 = find_pair(fam)
        if b and x2:
            pairs.append((fam, b, x2))

    # un grafico per famiglia: U_A/U_B/U_P raggruppate (base vs Bx2)
    for fam, b, x2 in pairs:
        labels = [f"{fam} base", f"{fam} Bx2"]
        uA = [_num(b.get("U_A_mean","nan")), _num(x2.get("U_A_mean","nan"))]
        uB = [_num(b.get("U_B_mean","nan")), _num(x2.get("U_B_mean","nan"))]
        uP = [_num(b.get("U_P_mean","nan")), _num(x2.get("U_P_mean","nan"))]

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 5))
        x = range(len(labels))
        width = 0.25
        plt.bar([i - width for i in x], uA, width=width, label="U_A")
        plt.bar([i for i in x],        uB, width=width, label="U_B")
        plt.bar([i + width for i in x], uP, width=width, label="U_P")
        plt.xticks(list(x), labels, rotation=0)
        plt.title(f"Utilizzazione — {fam}: base vs B×2")
        plt.ylabel("fraq. (0–1)")
        plt.legend()
        plt.tight_layout()
        out_path = PLOTS_DIR / f"U_compare_Bx2_{fam}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"✅ salvato: {out_path}")


def plot_n_sys(rows: List[Dict[str, str]]):
    # N_sys = R_mean * X (Little) — utile come indicatore complessivo
    labels = [r.get("scenario","") for r in rows]
    n_sys = [_num(r.get("R_mean_s_mean","nan")) * _num(r.get("X_jobs_per_s_mean","nan")) for r in rows]
    _bar_with_error(labels, n_sys, [0]*len(labels), "Numero medio nel sistema (N_sys = R * X)", "job", "N_sys.png")


def main(show: bool = False):
    rows = _read_rows(CSV_PATH)

    # Ordina in modo leggibile: 1FA prima di 2FA; base, heavy, Bx2
    def _sort_key(name: str) -> tuple:
        return (
            0 if "1FA" in name else 1,
            0 if "heavy" not in name.lower() else 1,
            0 if "Bx2" not in name else 1,
            name,
        )
    rows.sort(key=lambda r: _sort_key(r.get("scenario","")))

    bounds = _load_bounds_from_yaml(CFG_DIR)

    # Plot estesi
    plot_r_mean(rows)
    plot_x_vs_bound(rows, bounds)
    plot_u_grouped(rows)
    plot_compare_1fa_2fa(rows)
    plot_compare_base_vs_heavy(rows)
    plot_compare_bx2(rows)
    plot_n_sys(rows)

    if show:
        plt.show()


if __name__ == "__main__":
    main(show=True)
