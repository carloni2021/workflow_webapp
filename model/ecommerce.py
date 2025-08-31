import simpy
from typing import Dict, List, Optional
import statistics as stats

from rndbook.rng_setup import init_rng_for_replication, use_stream
from rndbook.rvgs import Exponential

from model.scenario import Scenario
from model.entities import JobRecord
from model.station_ps import ProcessorSharingStation

class EcommerceModel:
    """
    Modello e-commerce con tre stazioni PS (A, B, P).
    Supporta:
      - run_finite(...)              : orizzonte finito (tempo), warmup opzionale
      - run_steady(...)              : orizzonte infinito (batch means) con 64×1024 di default
    """

    def __init__(self, scn: Scenario, seed: int = 1234):

        self._arrival_times = [] # log arrivi per debug

        self.scenario = scn
        self.env = simpy.Environment()
        self.seed = int(seed)

        init_rng_for_replication(self.seed) # Inizializza RNG (tutti i 256 stream) per la replica

        self.lambda_req_s = 0.0 # tasso di arrivo (req/s) - inizializzato a 0, va settato con set_arrival_rate() nello sweep

        caps = self.scenario.capacities
        self.A = ProcessorSharingStation(self.env, "A", caps.get("A", 1))
        self.B = ProcessorSharingStation(self.env, "B", caps.get("B", 1))
        self.P = ProcessorSharingStation(self.env, "P", caps.get("P", 1))

        self.jobs_completed: List[JobRecord] = []
        self._batcher: Optional[EcommerceModel._BatchMeans] = None
        self._stop_event: Optional[simpy.Event] = None

    def set_arrival_rate(self, lam: Optional[float]):
        """Se impostato, usa lam (req/s) al posto dello Scenario."""
        """Usata dallo: basta aggiornare il tasso"""
        self.lambda_req_s = max(0.0, float(lam))

    def _arrival_process(self):

        jid = 0
        while True:
            lam = self.lambda_req_s
            if lam <= 0.0:
                # niente arrivi (protezione)
                yield self.env.timeout(1.0)
                continue

            mean_iat = 1.0 / lam
            use_stream("arrivals")  # seleziona lo stream definito in rng_setup.STREAMS["arrivals"]
            iat = Exponential(mean_iat)  # estrae l'inter-arrivo esponenziale con media 1/λ
            yield self.env.timeout(iat)
            self._arrival_times.append(self.env.now)  # LOG dell’arrivo reale

            # avvia il workflow della nuova richiesta
            jid += 1
            self.env.process(self.job_flow(jid))


    def _exp_demand(self, station: str, job_class: str) -> float:

        mean = float(self.scenario.service_demands.get(station, {}).get(job_class, 0.0))
        if mean <= 0.0:
            return 0.0
        use_stream(f"service_{station}")  # es.: service_A/B/P in rng_setup.STREAMS
        return Exponential(mean)

    # ---------- Arrivi e visite ----------
    def _ps_visit(self, station: ProcessorSharingStation, sname: str, job: JobRecord, class_id: str):
        """
        Visita a una stazione PS con domanda di servizio stocastica:
        campiona Exp(mean = D_s) dove D_s è nello Scenario per (stazione, class_id).
        Lo switch di stream RNG è "safe": se lo stream dedicato non esiste, ripiega.
        """
        d = self._exp_demand(sname, class_id)  # campione esponenziale
        if d <= 0.0:
            return

        # Esecuzione PS
        t_in = self.env.now
        done = station.ps_service(d)
        yield done
        t_out = self.env.now

        # Traccia permanenza al centro (PS: include tutto il tempo alla stazione)
        job.visit_times[sname] = job.visit_times.get(sname, 0.0) + (t_out - t_in)
        job.wait_times[sname] = job.wait_times.get(sname, 0.0) + 0.0  # in PS non c'è coda distinta

        # Notifica batcher (se attivo)
        if self._batcher is not None:
            self._batcher.on_visit_complete(sname, t_out - t_in)

    def job_flow(self, jid: int):
        job = JobRecord(id=jid, class_id="Class1", arrival_time=self.env.now)

        # Class1: A -> B
        yield from self._ps_visit(self.A, "A", job, "Class1")
        yield from self._ps_visit(self.B, "B", job, "Class1")

        # Class2: A -> P
        job.class_id = "Class2"
        yield from self._ps_visit(self.A, "A", job, "Class2")
        yield from self._ps_visit(self.P, "P", job, "Class2")

        # Class3: A -> exit
        job.class_id = "Class3"
        yield from self._ps_visit(self.A, "A", job, "Class3")

        job.completion_time = self.env.now
        self.jobs_completed.append(job)

        # Notifica il completamento di un JOB al batcher (se attivo)
        if self._batcher is not None:
            self._batcher.on_job_complete(job)

    # ---------- Run: orizzonte finito ----------
    def run_finite(self, *, horizon_s: float, warmup_s: float = 8000.0) -> Dict[str, float]:
        """
        Esegue una replica a orizzonte finito.
        Se warmup_s > 0, scarta il transitorio iniziale e misura solo su [warmup_s, horizon_s].
        """
        assert horizon_s > 0 and warmup_s >= 0 and horizon_s > warmup_s

        # Avvia il processo di arrivo
        self.env.process(self._arrival_process())

        # 1) Warmup (opzionale): esegui fino a warmup_s e scatta snapshot per aree busy
        areaA0 = areaB0 = areaP0 = t0 = 0.0
        if warmup_s > 0.0:
            self.env.run(until=warmup_s)
            # snapshot aree e tempo inizio finestra
            # 🔹 FLUSH fino a t = warmup_s
            self.A._area_accumulate()
            self.B._area_accumulate()
            self.P._area_accumulate()
            # snapshot aree e tempo inizio finestra
            areaA0, areaB0, areaP0 = self.A.busy_area, self.B.busy_area, self.P.busy_area
            t0 = warmup_s
            # azzera l'elenco dei job completati per misurare solo quelli che terminano dopo il warmup
            self.jobs_completed = []
        else:
            t0 = 0.0

        # 2) Misura: esegui fino all'orizzonte finale
        self.env.run(until=horizon_s)
        self.A._area_accumulate()
        self.B._area_accumulate()
        self.P._area_accumulate()

        # 3) Calcolo metriche SOLO sulla finestra [t0, horizon_s]
        duration = horizon_s - t0

        # Stima del tasso di arrivo (per debug)
        arrivals_in = sum(t0 <= t < horizon_s for t in self._arrival_times)
        lambda_hat = arrivals_in / duration if duration > 0 else float("nan")
        print(f"[CHECK] lambda_set={self.lambda_req_s:.3f}  lambda_hat={lambda_hat:.3f}  "
              f"arrivals_in={arrivals_in}  duration={duration:.0f}s")

        completed = list(self.jobs_completed)

        # Utilizzazioni: integrazione differenziale delle aree busy nella finestra
        if warmup_s > 0.0:
            U_A = (self.A.busy_area - areaA0) / (self.A.capacity * duration) if duration > 0 else float("nan")
            U_B = (self.B.busy_area - areaB0) / (self.B.capacity * duration) if duration > 0 else float("nan")
            U_P = (self.P.busy_area - areaP0) / (self.P.capacity * duration) if duration > 0 else float("nan")
        else:
            U_A = self.A.utilization(horizon_s)
            U_B = self.B.utilization(horizon_s)
            U_P = self.P.utilization(horizon_s)

        # Tempi di risposta e throughput sulla finestra
        completed = [j for j in self.jobs_completed if j.arrival_time >= t0]
        R = [j.completion_time - j.arrival_time for j in completed]
        R_mean = stats.mean(R) if R else float("nan")
        X = (len(completed) / duration) if duration > 0 else float("nan")
        print(f"[DEBUG R] t0={t0}  horizon={horizon_s}  duration={duration}  completed={len(completed)}")
        if completed:
            r0 = completed[0].completion_time - max(completed[0].arrival_time, t0)
            print(
                f"[DEBUG R] first R in window = {r0:.4f}s (comp={completed[0].completion_time:.2f}, arr={completed[0].arrival_time:.2f})")
        print(
            f"[DEBUG U] areas ΔA={self.A.busy_area - areaA0 if warmup_s > 0 else self.A.busy_area:.2f}  ΔB={self.B.busy_area - areaB0 if warmup_s > 0 else self.B.busy_area:.2f}  ΔP={self.P.busy_area - areaP0 if warmup_s > 0 else self.P.busy_area:.2f}")

        # --- sanity check di coerenza carico/utilizzo ---
        lam = getattr(self, "lambda_req_s", float("nan"))
        D = {"A": 0.7, "B": 0.8, "P": 0.4}  # dal tuo YAML 1FA (base)
        U_expected = {s: lam * D[s] for s in D}

        print(f"[CHECK] λ={lam:.3f}  U_atteso: A={U_expected['A']:.3f}  "
              f"B={U_expected['B']:.3f}  P={U_expected['P']:.3f}")
        print(f"[CHECK] U_misurato: A={U_A:.3f}  B={U_B:.3f}  P={U_P:.3f}")

        return {
            "R_mean_s": R_mean,
            "X_jobs_per_s": X,
            "U_A": U_A,
            "U_B": U_B,
            "U_P": U_P,
            "n_completed": len(completed),
        }
    # ---------- Run: steady-state (batch means) ----------
    class _BatchMeans:
        """Raccoglie statistiche per-batch mentre la simulazione gira."""
        def __init__(self, model: "EcommerceModel", n_batches: int, jobs_per_batch: int):
            self.m = model
            self.n_batches = int(n_batches)
            self.jobs_per_batch = int(jobs_per_batch)

            # stato corrente del batch
            self.reset_batch_state()

            # risultati per-batch
            self.R_mean_s_batches: List[float] = []
            self.X_jobs_per_s_batches: List[float] = []
            self.U_A_batches: List[float] = []
            self.U_B_batches: List[float] = []
            self.U_P_batches: List[float] = []

            # evento di stop per chiudere la run
            self.done_ev = self.m.env.event()

        def reset_batch_state(self) -> None:
            self.jobs_in_batch = 0
            self.sum_R = 0.0
            # snapshot tempo e aree per calcolare U su ciascun batch
            self.t0 = self.m.env.now
            # assicura che le aree siano aggiornate prima dello snapshot
            self.m.A._area_accumulate()
            self.m.B._area_accumulate()
            self.m.P._area_accumulate()
            self.areaA0 = self.m.A.busy_area
            self.areaB0 = self.m.B.busy_area
            self.areaP0 = self.m.P.busy_area

        def on_visit_complete(self, sname: str, soj_time: float) -> None:
            # Non serve per questa versione (usiamo le busy_area); lasciato per estensioni future.
            return

        def on_job_complete(self, job) -> None:
            # aggiorna R del job nel batch corrente
            self.jobs_in_batch += 1
            self.sum_R += (job.completion_time - job.arrival_time)

            if self.jobs_in_batch >= self.jobs_per_batch:
                # chiudi il batch e calcola le metriche
                t1 = self.m.env.now
                duration = max(t1 - self.t0, 1e-12)  # protezione numerica

                R_mean = self.sum_R / float(self.jobs_in_batch)
                X = float(self.jobs_in_batch) / duration

                # utilizzi via differenza d'area (normalizzata per capacità)
                self.m.A._area_accumulate()
                self.m.B._area_accumulate()
                self.m.P._area_accumulate()

                U_A = (self.m.A.busy_area - self.areaA0) / (self.m.A.capacity * duration)
                U_B = (self.m.B.busy_area - self.areaB0) / (self.m.B.capacity * duration)
                U_P = (self.m.P.busy_area - self.areaP0) / (self.m.P.capacity * duration)

                # salva risultati
                self.R_mean_s_batches.append(R_mean)
                self.X_jobs_per_s_batches.append(X)
                self.U_A_batches.append(U_A)
                self.U_B_batches.append(U_B)
                self.U_P_batches.append(U_P)

                # prepara il prossimo batch o termina
                if len(self.R_mean_s_batches) >= self.n_batches:
                    if not self.done_ev.triggered:
                        self.done_ev.succeed()
                else:
                    self.reset_batch_state()

    def run_batch_means(self, *, n_batches: int, jobs_per_batch: int) -> Dict[str, List[float]]:
        """
        Esegue la run a regime (batch means) e restituisce le serie per-batch:
        - R_mean_s_batches:   media dei tempi di risposta per batch
        - X_jobs_per_s_batches: throughput per batch
        - U_A/B/P_batches:    utilizzazioni per batch
        """
        assert n_batches > 0 and jobs_per_batch > 0

        # Imposta il tasso di arrivo dal tuo Scenario (il controller non lo setta)
        lam = 1.0 / float(self.scenario.get_interarrival_mean())
        self.set_arrival_rate(lam)

        # batcher e avvio sorgente di arrivi
        self._batcher = EcommerceModel._BatchMeans(self, n_batches, jobs_per_batch)
        self.env.process(self._arrival_process())

        # Esegui finché non si completano n_batches * jobs_per_batch job
        self.env.run(until=self._batcher.done_ev)

        # Confeziona il risultato atteso dal controller steady_state.py
        series = {
            "R_mean_s_batches": self._batcher.R_mean_s_batches,
            "X_jobs_per_s_batches": self._batcher.X_jobs_per_s_batches,
            "U_A_batches": self._batcher.U_A_batches,
            "U_B_batches": self._batcher.U_B_batches,
            "U_P_batches": self._batcher.U_P_batches,
        }

        # pulizia
        self._batcher = None
        return series
