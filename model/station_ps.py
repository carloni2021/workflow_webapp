from __future__ import annotations
from typing import List, Optional
import simpy


class ProcessorSharingStation:
    """
    Centro Processor Sharing con capacità 'capacity'.
    Con n job attivi si ha:
        rate_per_job = 1.0           se n <= capacity
        rate_per_job = capacity / n  se n > capacity
    Le domande sono espresse in secondi a rate=1.
    """

    # Tolleranza numerica per trattare residui ~0 dovuti all'aritmetica floating point
    TOL = 1e-9

    class _Token:
        """
        _Token rappresenta lo stato di un singolo job attivo nella stazione PS.

        Attributi:
        - remaining : float
            Domanda di servizio residua espressa in secondi (a rate = 1).
            Viene scalata dallo scheduler (_run_ps_loop) man mano che il job riceve
            servizio Processor Sharing.
        - done_ev : simpy.Event
            Evento SimPy che verrà completato (succeed) dallo scheduler quando
            remaining scende a ~0. È l’handle che consente al processo del job
            (tramite ps_service) di sospendersi fino al termine del servizio.

        In sintesi: un _Token è il “segnaposto” di un job attivo, contenente
        sia il residuo di lavoro da servire, sia l’evento da notificare al job
        al completamento.
        """
        def __init__(self, remaining: float, done_ev: simpy.Event):
            self.remaining = remaining  # Domanda residua
            self.done_ev = done_ev      # Evento SimPy che verrà “sbloccato” al completamento

    def __init__(self, env: simpy.Environment, name: str, capacity: int = 1):
        self.env = env
        self.name = name
        self.capacity = max(1, int(capacity))

        # Stato PS Station
        self.active: List[ProcessorSharingStation._Token] = []
        self._arrival_ev: Optional[simpy.Event] = None
        self._scheduler_running = False

        # Integrazione
        self._last_change = 0.0
        self._busy_servers = 0   # = min(len(active), capacity)
        self.busy_area = 0.0

    def _area_accumulate(self):
        now = self.env.now
        self.busy_area += self._busy_servers * (now - self._last_change)
        self._last_change = now

    def _recompute_busy(self):
        self._busy_servers = min(len(self.active), self.capacity)

    def utilization(self, horizon_end: float) -> float:
        denom = self.capacity * horizon_end
        return (self.busy_area / denom) if denom > 0 else 0.0

    # -------------
    # ---- API ----
    # -------------
    def ps_service(self, demand_s: float) -> simpy.Event:
        """
        Punto di ingresso per un job nella stazione Processor Sharing.
        Crea e restituisce un evento SimPy che verrà completato quando il job avrà
        ricevuto tutta la sua domanda di servizio.
        """
        # Evento banale
        if demand_s <= 0:
            ev = self.env.event()
            ev.succeed()
            return ev

        # Aggiorna (integra) l'area con il livello prima dell'arrivo
        self._area_accumulate()

        # Aggiungi job
        ev = self.env.event()  # Evento di completamento del job
        self.active.append(ProcessorSharingStation._Token(demand_s, ev))  # Il job è subito “attivo”
        self._recompute_busy()  # Aggiorna busy in base a n e capacity

        # Se il loop era in attesa di arrivi, sveglia il planner
        if self._arrival_ev is not None and not self._arrival_ev.triggered:
            self._arrival_ev.succeed()
            self._arrival_ev = None

        # Avvia il planner se non attivo
        if not self._scheduler_running:
            self._scheduler_running = True
            self.env.process(self._run_ps_loop())

        return ev


    def _run_ps_loop(self):
        last_rates_update = self.env.now

        while self.active:
            # Snapshot dei job già presenti PRIMA di attendere il prossimo evento
            cohort = list(self.active)                   # <-- solo questi ricevono servizio nell'intervallo
            n = len(cohort)

            rate_per_job = 1.0 if n <= self.capacity else self.capacity / n
            min_remaining = min(t.remaining for t in cohort)
            dt_to_finish = min_remaining / rate_per_job if rate_per_job > 0 else float("inf")

            timeout_ev = self.env.timeout(dt_to_finish)
            self._arrival_ev = self.env.event()
            res = yield simpy.AnyOf(self.env, [timeout_ev, self._arrival_ev])

            # Aggiorna i residui SOLO dei job che erano già nel cohort
            elapsed = self.env.now - last_rates_update
            for t in cohort:
                t.remaining = max(0.0, t.remaining - elapsed * rate_per_job)
                if t.remaining <= self.TOL:
                    t.remaining = 0.0
            last_rates_update = self.env.now

            if timeout_ev in res.events:
                # Completamento: integra area e completa proprio quel token
                self._area_accumulate()

                try:
                    done_tok = next(t for t in cohort if t.remaining <= self.TOL)
                except StopIteration:
                    done_tok = min(cohort, key=lambda t: t.remaining)
                    done_tok.remaining = 0.0

                # Rimuovi per identità dall'attuale lista active (potrebbe avere nuovi arrivi)
                try:
                    self.active.remove(done_tok)
                except ValueError:
                    # fallback robusto, non dovrebbe servire
                    idx = min(range(len(self.active)), key=lambda i: self.active[i].remaining)
                    done_tok = self.active.pop(idx)

                self._recompute_busy()
                done_tok.done_ev.succeed()
            else:
                # Nuovo arrivo: nessun completamento in questo step
                pass

        self._area_accumulate()
        self._recompute_busy()
        self._scheduler_running = False