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

def plot_convergence_R(R_cum, *, lam=None, scn=None, title=None, outfile=None, show=True):
    import math
    import matplotlib.pyplot as plt

    if scn is None:
        raise ValueError("scn mancante")

    # --- helper per leggere attributi da oggetto o dict ---
    def _get(obj, key, default=None):
        return getattr(obj, key, default) if hasattr(obj, key) else (obj.get(key, default) if isinstance(obj, dict) else default)

    # --- verifica che arrivi e servizi siano tutti esponenziali ---
    arr_proc = _get(scn, "arrival_process", None)
    svc_dist = _get(scn, "service_dist", {}) or {}

    def _is_exp(x):
        return isinstance(x, str) and x.strip().lower() == "exp"

    all_exp = _is_exp(arr_proc) # and all(_is_exp(svc_dist.get(n, "")) for n in ("A", "B", "P"))

    # --- prepara figura + serie cumulativa ---
    plt.figure(figsize=(8, 5))
    if R_cum:
        t_cum, y_cum = zip(*R_cum)
        plt.plot(t_cum, y_cum, color="darkorange", linewidth=2, label="R medio cumulativo")

    # --- se NON tutto exp: niente linea teorica ---
    if not all_exp:
        print(f"[INFO] R teorico NON tracciato: arrival_process='{arr_proc}' "
              f"service_dist={{A:{svc_dist.get('A')}, B:{svc_dist.get('B')}, P:{svc_dist.get('P')}}} "
              "(richiesto: tutti 'exp').")
    else:
        # ricava λ se non passato
        if lam is None:
            mean_iat = _get(scn, "interarrival_mean_s", None)
            if not mean_iat or mean_iat <= 0:
                raise ValueError("lam mancante e scn.interarrival_mean_s non valido")
            lam = 1.0 / mean_iat

        # service demand medi dai service_demands dello scenario (somma sulle classi)
        sd = _get(scn, "service_demands", {}) or {}
        D_A = sum(sd.get("A", {}).values()) if "A" in sd else 0.0
        D_B = sum(sd.get("B", {}).values()) if "B" in sd else 0.0
        D_P = sum(sd.get("P", {}).values()) if "P" in sd else 0.0

        # formula M/M/1 per nodo (come nel tuo codice)
        U_A, U_B, U_P = lam * D_A, lam * D_B, lam * D_P

        def R_node(D, U):
            return math.inf if U >= 1 else D / (1 - U)

        R_A = R_node(D_A, U_A)
        R_B = R_node(D_B, U_B)
        R_P = R_node(D_P, U_P)
        R_theory = R_A + R_B + R_P

        if math.isfinite(R_theory):
            print(f"[INFO] Tempo di risposta teorico R (M/M/1) ≈ {R_theory:.3f} s")
            plt.axhline(R_theory, color="red", linestyle="--", linewidth=2,
                        label=f"R teorico ≈ {R_theory:.2f}s")
        else:
            print("[WARN] Almeno un nodo ha U≥1: R teorico = ∞ (non traccio la linea)")

    # titoli/assi
    if title:
        plt.title(title)
    plt.xlabel("Tempo di simulazione [s]")
    plt.ylabel("Tempo di risposta medio R [s]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # salvataggio/show
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

