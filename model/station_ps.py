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
            self.remaining = remaining # Domanda residua
            self.done_ev = done_ev # Evento SimPy che verrà “sbloccato” al completamento

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
        ps_service è il punto di ingresso per un job nella stazione Processor Sharing.
        Crea e restituisce un evento SimPy che verrà completato quando il job avrà
        ricevuto tutta la sua domanda di servizio.

        Funzionamento:
        1. Se la domanda è nulla/negativa, restituisce subito un evento già completato.
        2. Aggiorna l’integrazione dell’utilizzo fino a ORA (prima dell’arrivo).
        3. Crea un token per il nuovo job (domanda residua + evento di completamento)
           e lo inserisce nella lista degli attivi (in PS il job inizia subito).
        4. Aggiorna il numero di server occupati (busy_servers).
        5. Se lo scheduler era in attesa di arrivi (_arrival_ev non ancora scattato),
           lo sveglia: l’arrivo cambia il numero di job attivi e quindi la velocità
           di servizio da riassegnare.
        6. Se lo scheduler non era attivo, lo avvia (_run_ps_loop).
        7. Restituisce l’evento che verrà “sbloccato” dal planner quando la domanda
           del job sarà interamente servita.

        In sintesi: ps_service registra un nuovo job come immediatamente attivo,
        aggiorna le statistiche di utilizzo, sincronizza lo scheduler e ritorna
        un evento che il chiamante deve yield-are per attendere la fine del servizio.
        """

        # Evento banale
        if demand_s <= 0:
            ev = self.env.event()
            ev.succeed()
            return ev

        # Aggiorna (integra) l'area con il livello prima dell'arrivo
        self._area_accumulate()

        # Aggiungi job
        ev = self.env.event() # Evento di completamento del job
        self.active.append(ProcessorSharingStation._Token(demand_s, ev)) # Il job è da subito “attivo” (ricordiamo che PS → niente coda)
        self._recompute_busy() #  Aggiorna busy in base a n e capacity

        # Se il loop era in attesa di arrivi, viene svegliato il planner
        # (cambio di n → cambia la stima del prox completamento)
        if self._arrival_ev is not None and not self._arrival_ev.triggered:
            self._arrival_ev.succeed()
            self._arrival_ev = None

        # Avvia il planner se non attivo
        if not self._scheduler_running:
            self._scheduler_running = True
            self.env.process(self._run_ps_loop())

        return ev

    def _run_ps_loop(self):

        """
         _run_ps_loop è il planner che applica il Processor Sharing aggiornando i “residui” di servizio
         e decidendo quando avviene il prossimo completamento o un cambiamento di regime (arrivo).

         Finché ci sono job attivi (self.active non è vuota), il loop:
         1. calcola il rate per job costante fino al prossimo evento,
         2. prevede quando accadrà il primo completamento (se non succedono arrivi),
         3. va in attesa del minimo tra:
            - quel completamento previsto,
            - un nuovo arrivo (se arriva, il rate cambia),
         4. quando si sveglia, scala le domande residue di tutti i job del servizio effettivamente erogato nel tratto,
         5. gestisce l’evento (completamento o arrivo) e ripete.
        """

        # Inizializzazione del tempo in cui è avvenuto l'ultimo aggiornamento del tempo residuo
        # La prima volta viene impostato a ORA, per capire quanto tempo sarà passato quando andranno aggiornate effettivamente le domande
        last_rates_update = self.env.now

        # ---- Loop ----
        while self.active:

            n = len(self.active)        # Numero di job attivi

            # Regola fondamentale del PS:
            # Se i job attivi sono ≤ capacità, ognuno lavora a 1.0
            # Altrimenti, la capacità si divide: capacity / n.
            rate_per_job = 1.0 if n <= self.capacity else self.capacity / n

            # Tempo fino al primo completamento, a rate costante
            # Se niente cambia, il primo job che finirebbe è quello con remaining minimo.
            min_remaining = min(t.remaining for t in self.active)
            dt_to_finish = min_remaining / rate_per_job if rate_per_job > 0 else float('inf')

            # Si attende tramite AnyOf.
            # AnyOf è un composite event: rappresenta l’attesa finché almeno uno tra un insieme di eventi succede
            # Il processo che fa yield si sospende finché uno qualunque degli eventi della lista è completato.
            # In questo caso:
            # 1. timeout_ev: Il completamento previsto,
            # 2. self._arrival_ev: Un “campanello” che ps_service() farà scattare quando arriva un nuovo job.
            #                      Se scatta l’arrivo prima del timeout, il rate non è più valido e va ricalcolato.
            #
            #
            # env.timeout(delay) crea un evento programmato che verrà completato (→ succeed())
            # dopo che il tempo simulato sarà avanzato di delay unità.
            # Non mette in pausa il programma reale, ma l’orologio di SimPy
            # (il Simulation Clock di cui si parla nel Next Event).
            timeout_ev = self.env.timeout(dt_to_finish)     # Possibile completamento
            # Crea un evento “vuoto” in SimPy,
            # che verrà usato come un campanello per notificare lo scheduler di un nuovo arrivo nella stazione Processor Sharing.
            self._arrival_ev = self.env.event()     # Campanello di arrivo
            res = yield simpy.AnyOf(self.env, [timeout_ev, self._arrival_ev])

            # Aggiorna le domande residue per il tempo trascorso
            elapsed = self.env.now - last_rates_update      # Calcolo del tempo passato dall'ultimo aggiornamento delle domande residue
            for t in self.active:
                t.remaining = max(0.0, t.remaining - elapsed * rate_per_job) # Tra gli eventi il rate per job è costante, quindi in elapsed ciascun job riceve (elapsed * rate_per_job)
            last_rates_update = self.env.now        # Aggiornamento del tempo in cui è avvenuto l'ultimo aggiornamento delle domande residue

            if timeout_ev in res.events:
                # Caso Completamento
                self._area_accumulate() # Integra l'utilizzo fino a ORA
                # Rimuove uno dei job che hanno remaining ≈ 0 (tolleranza 1e-12) e segnala il suo evento di fine.
                # Se più job sono arrivati a 0 nello stesso istante, ne toglie uno per giro;
                # Il successivo verrà tolto al giro seguente senza far scorrere il tempo (sempre env.now).
                idx = next(i for i, t in enumerate(self.active) if t.remaining <= 1e-12)

                done_tok = self.active.pop(idx)
                self._recompute_busy()
                done_tok.done_ev.succeed()
            else:
                # Caso Nuovo arrivo - di almeno un job (già inserito da ps_service)
                # Non c’è completamento in questo step;
                # Al giro successivo ricalcola n e il nuovo rate.
                pass

        # Quando la coda active è vuota: integrazione finale e terminazione
        self._area_accumulate()
        self._recompute_busy()
        self._scheduler_running = False
