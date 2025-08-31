from typing import Dict
from .rngs import plantSeeds, selectStream  # multi-stream RNG 0..255

# NOTA: rvgs.py usa internamente rngs.rndbook(), quindi rispetta lo stream selezionato.

# Mappa simbolica → indice stream (personalizzabile)
STREAMS: Dict[str, int] = {
    "arrivals": 0,
    "service_A": 1,
    "service_B": 2,
    "service_P": 3,
    # Altri stream possono essere aggiunti qui
}

def init_rng_for_replication(seed: int) -> None:
    """
    Inizializza i 256 stream per una replica.
    Per scenari confrontabili via CRN, tenere lo stesso seed0.
    """
    plantSeeds(seed)  # Inizializza coerentemente tutti gli stream.

def use_stream(name: str) -> None:
    selectStream(STREAMS[name])  # Seleziona lo stream corrente.
