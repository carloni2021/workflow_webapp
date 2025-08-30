import rndbook.rvgs as rvgs
import rndbook.rng_setup as rng_setup
import simpy
from typing import Dict, List, Optional
import statistics as stats

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

    def __init__(self, scenario: Scenario, seed: int = 42):
        self.scenario = scenario
        self.env = simpy.Environment()
        rng_setup.init_rng_for_replication(seed)

        self.arrival_rate_override: Optional[float] = None  # <<< NEW

        caps = scenario.capacities
        self.A = ProcessorSharingStation(self.env, "A", caps.get("A", 1))
        self.B = ProcessorSharingStation(self.env, "B", caps.get("B", 1))
        self.P = ProcessorSharingStation(self.env, "P", caps.get("P", 1))

        self.jobs_completed: List[JobRecord] = []
        self._batcher: Optional[EcommerceModel._BatchMeans] = None
        self._stop_event: Optional[simpy.Event] = None

    def set_arrival_rate(self, lam: Optional[float]):
        """Se impostato, usa lam (req/s) al posto dello Scenario."""
        self.arrival_rate_override = lam

    # ---------- Arrivi e visite ----------
    def _ps_visit(self, station: ProcessorSharingStation, sname: str, job: JobRecord, class_id: str):
        """
        Visita a una stazione PS con domanda di servizio stocastica:
        campiona Exp(mean = D_s) dove D_s è nello Scenario per (stazione, class_id).
        Lo switch di stream RNG è "safe": se lo stream dedicato non esiste, ripiega.
        """
        mean_D = self.scenario.service_demands[sname][class_id]
        if mean_D <= 0.0:
            return

        # --- Stream RNG: prova con uno dedicato, altrimenti ripiega su "services", altrimenti lascia com'è ---
        stream_name = f"svc_{sname}_{class_id}"
        try:
            rng_setup.use_stream(stream_name)
        except Exception:
            # ripiego comune a tutti i servizi (se definito nel tuo rng_setup)
            try:
                rng_setup.use_stream("services")
            except Exception:
                # opzionale: ultimo tentativo con "service"
                try:
                    rng_setup.use_stream("service")
                except Exception:
                    # nessuno stream disponibile: non cambiamo stream per evitare KeyError
                    pass

        # Campionamento domanda di servizio (Exp con media D_s)
        demand = rvgs.Exponential(mean_D)

        # Esecuzione PS
        t_in = self.env.now
        done = station.ps_service(demand)
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

    def arrival_process(self):
        # Se è presente un override, usalo. Altrimenti usa lo Scenario.
        if self.arrival_rate_override is not None and self.arrival_rate_override > 0:
            mean_ia = 1.0 / self.arrival_rate_override
        else:
            mean_ia = self.scenario.get_interarrival_mean()

        print("arrival_rate_override: " + str(self.arrival_rate_override) + ", mean_ia:  " + str(mean_ia))

        jid = 0
        while True:
            rng_setup.use_stream("arrivals")
            ia = rvgs.Exponential(mean_ia) if mean_ia > 0 else float("inf")
            yield self.env.timeout(ia)
            jid += 1
            self.env.process(self.job_flow(jid))

    # ---------- Run: orizzonte finito ----------
    def run_finite(self, *, horizon_s: float, warmup_s: float = 8000.0) -> Dict[str, float]:
        """
        Esegue una replica a orizzonte finito.
        Se warmup_s > 0, scarta il transitorio iniziale e misura solo su [warmup_s, horizon_s].
        """
        assert horizon_s > 0 and warmup_s >= 0 and horizon_s > warmup_s

        # Avvia il processo di arrivo
        self.env.process(self.arrival_process())

        # 1) Warmup (opzionale): esegui fino a warmup_s e scatta snapshot per aree busy
        areaA0 = areaB0 = areaP0 = t0 = 0.0
        if warmup_s > 0.0:
            self.env.run(until=warmup_s)
            # snapshot aree e tempo inizio finestra
            areaA0, areaB0, areaP0 = self.A.busy_area, self.B.busy_area, self.P.busy_area
            t0 = warmup_s
            # azzera l'elenco dei job completati per misurare solo quelli che terminano dopo il warmup
            self.jobs_completed = []
        else:
            t0 = 0.0

        # 2) Misura: esegui fino all'orizzonte finale
        self.env.run(until=horizon_s)

        # 3) Calcolo metriche SOLO sulla finestra [t0, horizon_s]
        duration = horizon_s - t0
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

        return {
            "R_mean_s": R_mean,
            "X_jobs_per_s": X,
            "U_A": U_A,
            "U_B": U_B,
            "U_P": U_P,
            "n_completed": len(completed),
        }

    # ---------- Run: orizzonte inffinito (batch means) ----------
    def run_batch_means(self, *, n_batches: int = 64, jobs_per_batch: int = 1024) -> Dict[str, List[float]]:
        """
        Esegue UNA run lunga, divisa in batch per NUMERO DI JOB (o visite),
        e ritorna le SERIE per-batch delle metriche:
          - R_mean_s_batches, X_jobs_per_s_batches  (globali per job)
          - U_A_batches, U_B_batches, U_P_batches    (per stazione, via integrazione)
        Lo stop avviene quando TUTTE le serie hanno riempito n_batches batch.
        """
        # Inizializza stop-event
        self._stop_event = self.env.event()
        # Inizializza il batcher
        self._batcher = EcommerceModel._BatchMeans(self, n_batches, jobs_per_batch, self._stop_event)

        # Avvia processo degli arrivi
        self.env.process(self.arrival_process())
        # Esegue fino allo stop-event
        self.env.run(until=self._stop_event)

        # Serie per-batch
        series = self._batcher.collect_series()

        # Cleanup
        self._batcher = None
        self._stop_event = None
        return series

    # ---------- Recorder interno per i batch means ----------
    class _BatchMeans:
        def __init__(self, model: "EcommerceModel", n_batches: int, jobs_per_batch: int, stop_event: simpy.Event):
            self.m = model
            self.env = model.env
            self.n_batches = int(n_batches)
            self.batch_size = int(jobs_per_batch)
            self.stop_event = stop_event

            # Stato per le stazioni
            self.sta_names = ["A", "B", "P"]
            self.sta = {}
            for s in self.sta_names:
                st = getattr(self.m, s)
                self.sta[s] = {
                    "idx": 0,
                    "count": 0,
                    "t0": self.env.now,
                    "area0": st.busy_area,
                    "sum_visit": 0.0,
                    "U_batches": [],
                    "visit_mean_batches": [],
                    "X_visits_batches": [],
                }
            # Stato globale (job completi)
            self.glob = {
                "idx": 0, "count": 0, "t0": self.env.now,
                "sum_R": 0.0,
                "R_mean_batches": [], "X_jobs_batches": [],
            }

        def on_visit_complete(self, sname: str, visit_time: float):
            s = self.sta[sname]
            st_obj: ProcessorSharingStation = getattr(self.m, sname)

            s["sum_visit"] += visit_time
            s["count"] += 1

            if s["count"] >= self.batch_size and s["idx"] < self.n_batches:
                t1 = self.env.now
                dt = max(1e-9, t1 - s["t0"])
                util = (st_obj.busy_area - s["area0"]) / (st_obj.capacity * dt)
                mean_visit = s["sum_visit"] / s["count"]
                x_vis = s["count"] / dt

                s["U_batches"].append(util)
                s["visit_mean_batches"].append(mean_visit)
                s["X_visits_batches"].append(x_vis)

                s["idx"] += 1
                s["count"] = 0
                s["t0"] = t1
                s["area0"] = st_obj.busy_area
                s["sum_visit"] = 0.0

                self._check_done()

        def on_job_complete(self, job: JobRecord):
            g = self.glob
            if g["idx"] >= self.n_batches:
                return
            R = job.completion_time - job.arrival_time
            g["sum_R"] += R
            g["count"] += 1

            if g["count"] >= self.batch_size:
                t1 = self.env.now
                dt = max(1e-9, t1 - g["t0"])
                R_mean = g["sum_R"] / g["count"]
                X_jobs = g["count"] / dt

                g["R_mean_batches"].append(R_mean)
                g["X_jobs_batches"].append(X_jobs)

                g["idx"] += 1
                g["count"] = 0
                g["t0"] = t1
                g["sum_R"] = 0.0

                self._check_done()

        def _check_done(self):
            sta_done = all(self.sta[s]["idx"] >= self.n_batches for s in self.sta_names)
            glob_done = (self.glob["idx"] >= self.n_batches)
            if sta_done and glob_done and not self.stop_event.triggered:
                self.stop_event.succeed()

        def collect_series(self) -> Dict[str, List[float]]:
            out: Dict[str, List[float]] = {
                "R_mean_s_batches": self.glob["R_mean_batches"],
                "X_jobs_per_s_batches": self.glob["X_jobs_batches"],
                "U_A_batches": self.sta["A"]["U_batches"],
                "U_B_batches": self.sta["B"]["U_batches"],
                "U_P_batches": self.sta["P"]["U_batches"],
            }
            return out
