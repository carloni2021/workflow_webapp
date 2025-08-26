from typing import Dict
from .rngs import plantSeeds, selectStream  # multi-stream RNG 0..255

# NOTA: rvgs.py usa internamente rngs.rndbook(), quindi rispetta lo stream selezionato.

# Mappa simbolica → indice stream (personalizzabile)
STREAMS: Dict[str, int] = {
    "arrivals": 0,
    "srv_S1":   1,
    "srv_S2":   2,
    "routing":  3,
    "setup":    4,
    "failure":  5,
}

def init_rng_for_replication(seed0: int, rep_id: int) -> None:
    """
    Inizializza i 256 stream per una replica.
    Per repliche indipendenti usa seed0 + rep_id.
    Per scenari confrontabili via CRN, tieni lo stesso seed0.
    """
    plantSeeds(seed0 + rep_id)  # inizializza coerentemente tutti gli stream.

def use_stream(name: str) -> None:
    selectStream(STREAMS[name])  # seleziona lo stream corrente.
