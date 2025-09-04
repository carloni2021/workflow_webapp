"""
service_dists.py — wrapper per domande di servizio realistiche:
- A e P: Erlang-k (SCV < 1) calibrata a media fissata
- B: Lognormale (SCV > 1) calibrata a media fissata

Regole di taratura
------------------
Erlang-k (A,P):   n = k (intero >=2), b = mean/k  → media = mean, SCV = 1/k
Lognormale (B):   sigma^2 = ln(1 + c2), mu = ln(mean) - 0.5*sigma^2 → media = mean, SCV = c2

Uso tipico
----------
from service_dists import demand_A, demand_B, demand_P, lognormal_params_from_mean_scv

x = demand_A(mean=2.5)               # default k=3 (SCV=1/3)
y = demand_P(mean=3.0, k=4)          # k scelto esplicitamente
z = demand_B(mean=2.5, c2=2.0)       # SCV=2 per la stazione B

Puoi anche ottenere i parametri senza estrarre campioni:
(mu, sigma) = lognormal_params_from_mean_scv(mean=2.5, c2=2.0)
"""

from typing import Tuple
import math

# Primitive RNG dal progetto
from rndbook.rvgs import Erlang, Lognormal


# ---------- Helper: Erlang-k ----------
def erlang_params_from_mean(mean: float, k: int) -> Tuple[int, float]:
    """
    Ritorna (n, b) per un'Erlang con media 'mean' e SCV=1/k.
    n = k (intero >=2), b = mean/k.
    """
    if not (mean > 0.0):
        raise ValueError("mean deve essere > 0")
    if not (isinstance(k, int) and k >= 2):
        raise ValueError("k deve essere un intero >= 2")
    n = k
    b = mean / k
    return n, b


def demand_erlang(mean: float, k: int) -> float:
    """
    Estrae un campione Erlang-k con media 'mean' e SCV=1/k.
    """
    n, b = erlang_params_from_mean(mean, k)
    return Erlang(n, b)


# ---------- Helper: Lognormale ----------
def lognormal_params_from_mean_scv(mean: float, c2: float) -> Tuple[float, float]:
    """
    Ritorna (mu, sigma) per una Lognormale con media 'mean' e SCV 'c2' (>1).
    """
    if not (mean > 0.0):
        raise ValueError("mean deve essere > 0")
    if not (c2 > 1.0):
        raise ValueError("c2 (SCV) deve essere > 1 per la Lognormale più variabile dell'esponenziale")
    sigma2 = math.log(1.0 + c2)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - 0.5 * sigma2
    return mu, sigma


def demand_lognormal(mean: float, c2: float) -> float:
    """
    Estrae un campione Lognormale con media 'mean' e SCV 'c2' (>1).
    """
    mu, sigma = lognormal_params_from_mean_scv(mean, c2)
    return Lognormal(mu, sigma)


# ---------- Wrapper per stazioni A, B, P ----------
def demand_A(mean: float, k: int = 3) -> float:
    """Erlang-k per stazione A. Default k=3 → SCV=1/3."""
    return demand_erlang(mean, k)


def demand_P(mean: float, k: int = 2) -> float:
    """Erlang-k per stazione P. Default k=2 → SCV=1/2."""
    return demand_erlang(mean, k)


def demand_B(mean: float, c2: float = 2.0) -> float:
    """Lognormale per stazione B. Default SCV=2.0."""
    return demand_lognormal(mean, c2)
