# view/convergence_plot.py
from __future__ import annotations

import math
import matplotlib.pyplot as plt


def plot_convergence_R(R_cum, *, lam=None, scn=None, title=None, outfile=None, show=True):

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

def plot_convergence_N(N_cum, *, lam=None, scn=None, title=None, outfile=None, show=True):
    if scn is None:
        raise ValueError("scn mancante")

    # helper lettura attributi
    def _get(obj, key, default=None):
        return getattr(obj, key, default) if hasattr(obj, key) else (
            obj.get(key, default) if isinstance(obj, dict) else default)

    # verifica caso esponenziale
    arr_proc = _get(scn, "arrival_process", None)
    svc_dist = _get(scn, "service_dist", {}) or {}

    def _is_exp(x):
        return isinstance(x, str) and x.strip().lower() == "exp"

    all_exp = _is_exp(arr_proc)  # stessa logica di plot_convergence_R

    # figura + serie cumulativa
    plt.figure(figsize=(8, 5))
    if N_cum:
        t_cum, y_cum = zip(*N_cum)
        plt.plot(t_cum, y_cum, color="darkorange", linewidth=2, label="N medio cumulativo")

    # teorico solo se tutto exp
    if not all_exp:
        print(f"[INFO] N teorico NON tracciato: arrival_process='{arr_proc}' "
              f"service_dist={{A:{svc_dist.get('A')}, B:{svc_dist.get('B')}, P:{svc_dist.get('P')}}} "
              "(richiesto: tutti 'exp').")
    else:
        # ricava λ se non passato
        if lam is None:
            mean_iat = _get(scn, "interarrival_mean_s", None)
            if not mean_iat or mean_iat <= 0:
                raise ValueError("lam mancante e scn.interarrival_mean_s non valido")
            lam = 1.0 / mean_iat

        # service demand medi per nodo
        sd = _get(scn, "service_demands", {}) or {}
        D_A = sum(sd.get("A", {}).values()) if "A" in sd else 0.0
        D_B = sum(sd.get("B", {}).values()) if "B" in sd else 0.0
        D_P = sum(sd.get("P", {}).values()) if "P" in sd else 0.0

        U_A, U_B, U_P = lam * D_A, lam * D_B, lam * D_P

        def L_node(U):
            return math.inf if U >= 1 else U / (1 - U)

        N_A = L_node(U_A)
        N_B = L_node(U_B)
        N_P = L_node(U_P)
        N_theory = N_A + N_B + N_P

        if math.isfinite(N_theory):
            print(f"[INFO] Numero medio teorico N (M/M/1) ≈ {N_theory:.3f}")
            plt.axhline(N_theory, color="red", linestyle="--", linewidth=2,
                        label=f"N teorico ≈ {N_theory:.2f}")
        else:
            print("[WARN] Almeno un nodo ha U≥1: N teorico = ∞ (non traccio la linea)")

    # titoli/assi
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