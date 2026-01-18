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
from rndbook.rng_setup import use_stream


def hyperexp2(mean: float, p: float) -> float:

    if not (mean > 0.0):
        raise ValueError("mean deve essere > 0")
    if not (0.0 < p < 1.0):
        raise ValueError("p deve essere in (0,1)")
    # NOTA: Exponential(m) attende la *media*, non il tasso.

    m1 = mean / (2.0 * p)
    m2 = mean / (2.0 * (1.0 - p))

    use_stream("arrivals_H2_phase")
    branch_choice  = Bernoulli(p)

    use_stream("arrivals_H2_time")
    if branch_choice:
        return Exponential(m1)
    else:
        return Exponential(m2)