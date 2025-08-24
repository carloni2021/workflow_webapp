from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class JobRecord:

    """
    Rappresenta un job che attraversa il sistema di simulazione.

    Ogni job mantiene informazioni sul proprio flusso di servizio
    (tempi di visita, eventuali tempi di attesa) e sugli istanti di
    arrivo/completamento.

    Attributi
    ----------
    id : int
        Identificativo univoco del job (assegnato dal generatore di arrivi).
    class_id : str
        Classe logica del job (es. "Class1", "Class2", ...), usata per
        determinare le domande di servizio nelle varie stazioni.
    arrival_time : float
        Tempo simulato in cui il job è arrivato nel sistema.
    completion_time : Optional[float]
        Tempo simulato in cui il job ha completato il proprio percorso;
        None se non è ancora terminato.
    visit_times : dict[str, float]
        Tempi totali trascorsi dal job in ciascuna stazione
        (dall’ingresso all’uscita, quindi includono anche eventuali rallentamenti PS).
    wait_times : dict[str, float]
        Tempi di attesa per ciascuna stazione. Nel caso di Processor Sharing
        è sempre 0.0, ma il campo è mantenuto per coerenza con altri modelli.

    Note
    ----
    - In questo modello i job riusano lo stesso record attraversando più fasi
      con class_id diversi (Class1 → Class2 → Class3).
    - I campi vengono aggiornati dai metodi `_ps_visit` e `job_flow` in
      `EcommerceModel`.
    """

    id: int
    class_id: str = "Class1"
    arrival_time: float = 0.0
    completion_time: Optional[float] = None
    visit_times: Dict[str, float] = field(default_factory=dict)  # tempo alla stazione (PS)
    wait_times: Dict[str, float] = field(default_factory=dict)   # 0 in PS (placeholder per futuri FCFS)
