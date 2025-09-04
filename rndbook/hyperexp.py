"""
hyperexp.py — Generatore iperesponenziale H2 (balanced) per progetti che usano la
libreria rvgs.py (Exponential, Bernoulli).

Uso tipico
----------
from hyperexp import HyperExp2Balanced, scv_from_p, p_from_scv

# Media target degli inter-arrivi (es. 1/lambda = 3)
mean_iat = 3.0

# Scegli p direttamente (alta variabilità se p è vicino a 0 o 1)
x = HyperExp2Balanced(mean_iat, p=0.10)

# Oppure specifica un SCV desiderato (>1) e ricava p
p = p_from_scv(c2=4.0)
x = HyperExp2Balanced(mean_iat, p)

In un contesto di simulazione dove si è già selezionato lo stream RNG
corretto (es. "arrivals"), questo generatore rispetta lo stream corrente
perché sfrutta le primitive Bernoulli ed Exponential della libreria esistente.

Dettagli
--------
Forma H2 "bilanciata":
  - con probabilità p si estrae Exp(m1), con m1 = mean / (2p)
  - con probabilità 1-p si estrae Exp(m2), con m2 = mean / (2(1-p))
La media risultante è mean, indipendentemente da p.
Lo SCV vale c^2 = 1/(2 p (1-p)) - 1, ≥ 1.
"""
from rndbook.rvgs import Exponential, Bernoulli


def HyperExp2Balanced(mean: float, p: float) -> float:
    """
    Estrae un campione da una H2 bilanciata con media 'mean'.

    Parametri
    ---------
    mean : float
        Media target (>0).
    p : float
        Probabilità del ramo 1 (0<p<1). Più p è vicino a 0 o 1, più la
        variabilità è alta.

    Ritorna
    -------
    float
        Un campione casuale.
    """
    if not (mean > 0.0):
        raise ValueError("mean deve essere > 0")
    if not (0.0 < p < 1.0):
        raise ValueError("p deve essere in (0,1)")
    # NOTA: Exponential(m) attende la *media*, non il tasso.
    m1 = mean / (2.0 * p)
    m2 = mean / (2.0 * (1.0 - p))
    return Exponential(m1) if Bernoulli(p) else Exponential(m2)


def scv_from_p(p: float) -> float:
    """
    Coefficiente di variazione al quadrato (SCV) dell'H2 bilanciata dato p.
    c^2 = 1/(2 p (1-p)) - 1
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p deve essere in (0,1)")
    return 1.0 / (2.0 * p * (1.0 - p)) - 1.0


def p_from_scv(c2: float) -> float:
    """
    Ricava p da un SCV desiderato c2 (>1). Ritorna il valore di p ∈ (0, 0.5].
    (Esiste anche l'altro simmetrico 1-p).

    Formula inversa (bilanciata):
        p = (1 - sqrt((c2 - 1)/(c2 + 1))) / 2

    Per ottenere l'altro ramo (>=0.5), usa 1 - p.
    """
    if not (c2 > 1.0):
        raise ValueError("c2 (SCV) deve essere > 1 per un'iperesponenziale più variabile dell'esponenziale")
    import math
    return (1.0 - math.sqrt((c2 - 1.0) / (c2 + 1.0))) / 2.0