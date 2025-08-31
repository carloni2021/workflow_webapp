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