# view/convergence_plot.py
from __future__ import annotations

import math
import matplotlib.pyplot as plt

STD_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# --- MULTI-SEED OVERLAY PLOTS -------------------------------------------------

def plot_convergence_R_multi(R_cum_list, *, lam=None, scn=None, title=None,
                             labels=None, outfile=None, show=True):
    """
    Disegna più andamenti di R_cum (uno per seed) sullo stesso asse.
    - R_cum_list: lista di serie cumulative, ciascuna come [(t0, R0), (t1, R1), ...]
    - labels: etichette per la legenda (opzionali)
    - scn/lam: per la linea teorica (M/M/1 per nodo) se tutto exp
    """
    if scn is None:
        raise ValueError("scn mancante")

    def _get(obj, key, default=None):
        return getattr(obj, key, default) if hasattr(obj, key) else (
            obj.get(key, default) if isinstance(obj, dict) else default)

    def _is_exp(x):
        return isinstance(x, str) and x.strip().lower() == "exp"

    plt.figure(figsize=(8, 5))

    if labels is None:
        labels = [f"run {i+1}" for i in range(len(R_cum_list))]

    for i, (curve, lab) in enumerate(zip(R_cum_list, labels)):
        if curve:
            t, y = zip(*curve)
            # Seleziona il colore ciclicamente
            c = STD_COLORS[i % len(STD_COLORS)]
            plt.plot(t, y, color=c, linewidth=1.8, alpha=0.9, label=str(lab))

    # Linea teorica (se arrivi/servizi esponenziali)
    arr_proc = _get(scn, "arrival_process", None)
    svc_dist = _get(scn, "service_dist", {}) or {}
    all_exp = _is_exp(arr_proc)

    if not all_exp:
        print(f"[INFO] R teorico NON tracciato: arrival_process='{arr_proc}' "
              f"service_dist={{A:{svc_dist.get('A')}, B:{svc_dist.get('B')}, P:{svc_dist.get('P')}}} "
              "(richiesto: tutti 'exp').")
    else:
        if lam is None:
            mean_iat = _get(scn, "interarrival_mean_s", None)
            if not mean_iat or mean_iat <= 0:
                raise ValueError("lam mancante e scn.interarrival_mean_s non valido")
            lam = 1.0 / mean_iat

        sd = _get(scn, "service_demands", {}) or {}
        D_A = sum(sd.get("A", {}).values()) if "A" in sd else 0.0
        D_B = sum(sd.get("B", {}).values()) if "B" in sd else 0.0
        D_P = sum(sd.get("P", {}).values()) if "P" in sd else 0.0

        U_A, U_B, U_P = lam * D_A, lam * D_B, lam * D_P

        def R_node(D, U):
            return math.inf if U >= 1 else D / (1 - U)

        R_theory = R_node(D_A, U_A) + R_node(D_B, U_B) + R_node(D_P, U_P)

        if math.isfinite(R_theory):
            print(f"[INFO] Tempo di risposta teorico R (M/M/1) ≈ {R_theory:.3f} s")
            plt.axhline(R_theory, color="red", linestyle="--", linewidth=2,
                        label=f"R teorico ≈ {R_theory:.2f}s")
        else:
            print("[WARN] Almeno un nodo ha U≥1: R teorico = ∞ (non traccio la linea)")

    if title:
        plt.title(title)
    plt.xlabel("Tempo di simulazione [s]")
    plt.ylabel("Tempo di risposta medio R [s]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence_N_multi(N_cum_list, *, lam=None, scn=None, title=None,
                             labels=None, outfile=None, show=True):
    """
    Disegna più andamenti di N_cum (uno per seed) sullo stesso asse.
    - N_cum_list: lista di serie cumulative [(t, N), ...]
    - labels: etichette per la legenda (opzionali)
    - scn/lam: per la linea teorica (M/M/1 per nodo) se tutto exp
    """
    if scn is None:
        raise ValueError("scn mancante")

    def _get(obj, key, default=None):
        return getattr(obj, key, default) if hasattr(obj, key) else (
            obj.get(key, default) if isinstance(obj, dict) else default)

    def _is_exp(x):
        return isinstance(x, str) and x.strip().lower() == "exp"

    plt.figure(figsize=(8, 5))

    if labels is None:
        labels = [f"run {i+1}" for i in range(len(N_cum_list))]

    for curve, lab in zip(N_cum_list, labels):
        if curve:
            t, y = zip(*curve)
            plt.plot(t, y, linewidth=1.8, alpha=0.9, label=str(lab))

    arr_proc = _get(scn, "arrival_process", None)
    svc_dist = _get(scn, "service_dist", {}) or {}
    all_exp = _is_exp(arr_proc)

    if not all_exp:
        print(f"[INFO] N teorico NON tracciato: arrival_process='{arr_proc}' "
              f"service_dist={{A:{svc_dist.get('A')}, B:{svc_dist.get('B')}, P:{svc_dist.get('P')}}} "
              "(richiesto: tutti 'exp').")
    else:
        if lam is None:
            mean_iat = _get(scn, "interarrival_mean_s", None)
            if not mean_iat or mean_iat <= 0:
                raise ValueError("lam mancante e scn.interarrival_mean_s non valido")
            lam = 1.0 / mean_iat

        sd = _get(scn, "service_demands", {}) or {}
        D_A = sum(sd.get("A", {}).values()) if "A" in sd else 0.0
        D_B = sum(sd.get("B", {}).values()) if "B" in sd else 0.0
        D_P = sum(sd.get("P", {}).values()) if "P" in sd else 0.0

        U_A, U_B, U_P = lam * D_A, lam * D_B, lam * D_P

        def L_node(U):
            return math.inf if U >= 1 else U / (1 - U)

        N_theory = L_node(U_A) + L_node(U_B) + L_node(U_P)

        if math.isfinite(N_theory):
            print(f"[INFO] Numero medio teorico N (M/M/1) ≈ {N_theory:.3f}")
            plt.axhline(N_theory, color="red", linestyle="--", linewidth=2,
                        label=f"N teorico ≈ {N_theory:.2f}")
        else:
            print("[WARN] Almeno un nodo ha U≥1: N teorico = ∞ (non traccio la linea)")

    if title:
        plt.title(title)
    plt.xlabel("Tempo di simulazione [s]")
    plt.ylabel("Numero medio N [-]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence_R_multi_via_runner(runner, seeds=(101, 102, 103, 104, 105), *,
                                        lam=None, scn=None, title=None,
                                        outfile=None, show=True):
    """
    runner(scn, seed) -> R_cum. Esegue i seed e richiama plot_convergence_R_multi.
    """
    series = [runner(scn, s) for s in seeds]
    labels = [f"seed {s}" for s in seeds]
    return plot_convergence_R_multi(series, lam=lam, scn=scn, title=title,
                                    labels=labels, outfile=outfile, show=show)


def plot_convergence_N_multi_via_runner(runner, seeds=(101, 102, 103, 104, 105), *,
                                        lam=None, scn=None, title=None,
                                        outfile=None, show=True):
    """
    runner(scn, seed) -> N_cum. Esegue i seed e richiama plot_convergence_N_multi.
    """
    series = [runner(scn, s) for s in seeds]
    labels = [f"seed {s}" for s in seeds]
    return plot_convergence_N_multi(series, lam=lam, scn=scn, title=title,
                                    labels=labels, outfile=outfile, show=show)
