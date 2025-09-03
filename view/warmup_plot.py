# view/warmup_plot.py
from __future__ import annotations
from typing import List, Tuple, Optional
import math
import matplotlib.pyplot as plt

def _estimate_warmup_from_cumulative(series_cum: List[Tuple[float, float]],
                                     window: int = 10,
                                     eps_rel: float = 0.01) -> Optional[float]:
    """
    Stima warmup: primo t in cui la variazione relativa della media cumulativa
    resta < eps_rel per 'window' punti consecutivi.
    """
    if not series_cum or len(series_cum) < window + 1:
        return None
    stable = 0
    prev = series_cum[0][1]
    for t, rbar in series_cum[1:]:
        if not (math.isfinite(rbar) and math.isfinite(prev)):
            stable = 0
        else:
            rel = abs(rbar - prev) / max(abs(rbar), 1e-12)
            stable = stable + 1 if rel < eps_rel else 0
            if stable >= window:
                return t
        prev = rbar
    return None

def plot_warmup_R(R_cum, R_bin=None, *, title=None, outfile=None, show=True):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    # Serie per-bin (rumorosa)
    if R_bin:
        t_bin, y_bin = zip(*R_bin)
        plt.plot(t_bin, y_bin, color="skyblue", alpha=0.6, label="R medio per bin (rumoroso)")

    # Serie cumulativa (più stabile)
    if R_cum:
        t_cum, y_cum = zip(*R_cum)
        plt.plot(t_cum, y_cum, color="darkorange", linewidth=2, label="R medio cumulativo")

    # --- Calcolo teorico ---
    lam = 1/3  # tasso di arrivo
    D_A, D_B, D_P = 0.7, 0.8, 0.4
    U_A, U_B, U_P = lam*D_A, lam*D_B, lam*D_P
    R_A = D_A / (1 - U_A)
    R_B = D_B / (1 - U_B)
    R_P = D_P / (1 - U_P)
    R_theory = R_A + R_B + R_P
    print(f"[INFO] Tempo di risposta teorico R ≈ {R_theory:.3f} s")

    # Linea orizzontale per il valore teorico
    plt.axhline(R_theory, color="red", linestyle="--", linewidth=2,
                label=f"R teorico ≈ {R_theory:.2f}s")

    # Titoli ed etichette
    if title:
        plt.title(title)
    plt.xlabel("Tempo di simulazione [s]")
    plt.ylabel("Tempo di risposta medio R [s]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Salvataggio / show
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
