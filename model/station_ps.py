from __future__ import annotations
from typing import List, Optional
import simpy


class ProcessorSharingStation:
    """
    Centro Processor Sharing con capacità 'capacity'.
    Se n job attivi:
      rate_per_job = 1.0           se n <= capacity
                   = capacity / n  se n >  capacity
    Le domande sono espresse in secondi a rate=1.
    """
    class _Token:
        def __init__(self, remaining: float, done_ev: simpy.Event):
            self.remaining = remaining
            self.done_ev = done_ev

    def __init__(self, env: simpy.Environment, name: str, capacity: int = 1):
        self.env = env
        self.name = name
        self.capacity = max(1, int(capacity))

        # stato PS
        self.active: List[ProcessorSharingStation._Token] = []
        self._arrival_ev: Optional[simpy.Event] = None
        self._scheduler_running = False

        # integrazione utilizzo
        self._last_change = 0.0
        self._busy_servers = 0   # = min(len(active), capacity)
        self.busy_area = 0.0

    # ---- integrazione utilizzo ----
    def _area_accumulate(self):
        now = self.env.now
        self.busy_area += self._busy_servers * (now - self._last_change)
        self._last_change = now

    def _recompute_busy(self):
        self._busy_servers = min(len(self.active), self.capacity)

    def utilization(self, horizon_end: float) -> float:
        denom = self.capacity * horizon_end
        return (self.busy_area / denom) if denom > 0 else 0.0

    # ---- API ----
    def ps_service(self, demand_s: float) -> simpy.Event:
        if demand_s <= 0:
            ev = self.env.event()
            ev.succeed()
            return ev

        # aggiorna area con il livello pre-arrivo
        self._area_accumulate()

        # aggiungi job
        ev = self.env.event()
        self.active.append(ProcessorSharingStation._Token(demand_s, ev))
        self._recompute_busy()

        # sveglia il planner se in attesa di un arrivo
        if self._arrival_ev is not None and not self._arrival_ev.triggered:
            self._arrival_ev.succeed()
            self._arrival_ev = None

        # avvia il planner se non attivo
        if not self._scheduler_running:
            self._scheduler_running = True
            self.env.process(self._run_ps_loop())

        return ev

    # ---- loop del planner PS ----
    def _run_ps_loop(self):
        last_rates_update = self.env.now
        while self.active:
            n = len(self.active)
            rate_per_job = 1.0 if n <= self.capacity else self.capacity / n

            # tempo fino al primo completamento, a rate costante
            min_remaining = min(t.remaining for t in self.active)
            dt_to_finish = min_remaining / rate_per_job if rate_per_job > 0 else float('inf')

            timeout_ev = self.env.timeout(dt_to_finish)
            self._arrival_ev = self.env.event()
            res = yield simpy.AnyOf(self.env, [timeout_ev, self._arrival_ev])

            # aggiorna le domande residue per il tempo trascorso
            elapsed = self.env.now - last_rates_update
            for t in self.active:
                t.remaining = max(0.0, t.remaining - elapsed * rate_per_job)
            last_rates_update = self.env.now

            if timeout_ev in res.events:
                # completamento
                self._area_accumulate()
                idx = next(i for i, t in enumerate(self.active) if t.remaining <= 1e-12)
                done_tok = self.active.pop(idx)
                self._recompute_busy()
                done_tok.done_ev.succeed()
            else:
                # nuovo arrivo già inserito da ps_service
                pass

        # coda vuota: integra ultimo tratto e termina
        self._area_accumulate()
        self._recompute_busy()
        self._scheduler_running = False
