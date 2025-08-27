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
      - run_steady(...)  : orizzonte infinito (batch means) con 64×1024 di default
    """

    def __init__(self, scenario: Scenario, seed: int = 42):
        self.scenario = scenario
        self.env = simpy.Environment()
        rng_setup.init_rng_for_replication(seed)

        caps = scenario.capacities
        self.A = ProcessorSharingStation(self.env, "A", caps.get("A", 1))
        self.B = ProcessorSharingStation(self.env, "B", caps.get("B", 1))
        self.P = ProcessorSharingStation(self.env, "P", caps.get("P", 1))

        self.jobs_completed: List[JobRecord] = []
        self._batcher: Optional[EcommerceModel._BatchMeans] = None
        self._stop_event: Optional[simpy.Event] = None

    # ---------- Arrivi e visite ----------
    def _ps_visit(self, station: ProcessorSharingStation, sname: str, job: JobRecord, class_id: str):
        demand = self.scenario.service_demands[sname][class_id]
        if demand <= 0:
            return
        t_in = self.env.now
        done = station.ps_service(demand)
        yield done
        t_out = self.env.now

        job.visit_times[sname] = job.visit_times.get(sname, 0.0) + (t_out - t_in)
        job.wait_times[sname] = job.wait_times.get(sname, 0.0) + 0.0

        # Notifica il completamento di una visita al batcher (se attivo)
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
        mean_ia = self.scenario.get_interarrival_mean()
        jid = 0
        while True:
            rng_setup.use_stream("arrivals")
            ia = rvgs.Exponential(mean_ia) if mean_ia > 0 else float("inf")
            yield self.env.timeout(ia)
            jid += 1
            self.env.process(self.job_flow(jid))

    # ---------- Run: orizzonte finito ----------
    def run_finite(self, *, horizon_s: float) -> Dict[str, float]:
        """
        Esegue una replica a orizzonte finito (misura su [0, horizon]).
        """
        self.env.process(self.arrival_process())
        self.env.run(until=horizon_s)

        duration = horizon_s
        completed = list(self.jobs_completed)

        U_A = self.A.utilization(horizon_s)
        U_B = self.B.utilization(horizon_s)
        U_P = self.P.utilization(horizon_s)

        R = [j.completion_time - j.arrival_time for j in completed]
        R_mean = stats.mean(R) if R else float("nan")
        X = len(completed) / duration

        return {
            "R_mean_s": R_mean,
            "X_jobs_per_s": X,
            "U_A": U_A,
            "U_B": U_B,
            "U_P": U_P,
            "n_completed": len(completed),
        }

    # ---------- Run: orizzonte infinito (batch means) ----------
    def run_batch_means(self, *, n_batches: int = 64, jobs_per_batch: int = 1024) -> Dict[str, List[float]]:
        """
        Esegue UNA run lunga, divisa in batch per NUMERO DI JOB (o visite),
        e ritorna le SERIE per-batch delle metriche:
          - R_mean_s_batches, X_jobs_per_s_batches  (globali per job)
          - U_A_batches, U_B_batches, U_P_batches    (per stazione, via integrazione)
        Lo stop avviene quando TUTTE le serie hanno riempito n_batches batch.
        """
        # inizializza batcher e stop-event
        self._stop_event = self.env.event()
        self._batcher = EcommerceModel._BatchMeans(self, n_batches, jobs_per_batch, self._stop_event)

        # avvia arrivi e corri finché il batcher non dice "basta"
        self.env.process(self.arrival_process())
        self.env.run(until=self._stop_event)

        # produce le serie per-batch
        series = self._batcher.collect_series()
        # cleanup
        self._batcher = None
        self._stop_event = None
        return series

    # ---------- Recorder interno per i batch means ----------
    class _BatchMeans:
        """
        Gestisce i batch per:
          - GLOBAL: job completi (R_mean, X)
          - STAZIONI: A, B, P (utilizzazione, mean visit time, throughput visite)
        Ogni stazione fa batch per numero di VISITE completate alla stazione.
        """
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
                    "idx": 0, "count": 0, "t0": self.env.now, "area0": st.busy_area,
                    "sum_visit": 0.0,
                    "U_batches": [], "visit_mean_batches": [], "X_visits_batches": [],
                }

            # Stato globale (job completi)
            self.glob = {
                "idx": 0, "count": 0, "t0": self.env.now,
                "sum_R": 0.0,
                "R_mean_batches": [], "X_jobs_batches": [],
            }

        # --- Hook: una VISITA a una stazione è completa ---
        def on_visit_complete(self, sname: str, visit_time: float):
            s = self.sta[sname]
            st_obj: ProcessorSharingStation = getattr(self.m, sname)

            # Accumula per la batch corrente
            s["sum_visit"] += visit_time
            s["count"] += 1

            # Se la batch è piena (per numero di visite/job al centro), chiudi la batch
            if s["count"] >= self.batch_size and s["idx"] < self.n_batches:
                t1 = self.env.now
                dt = max(1e-9, t1 - s["t0"])
                util = (st_obj.busy_area - s["area0"]) / (st_obj.capacity * dt)
                mean_visit = s["sum_visit"] / s["count"]
                x_vis = s["count"] / dt

                s["U_batches"].append(util)
                s["visit_mean_batches"].append(mean_visit)
                s["X_visits_batches"].append(x_vis)

                # reset per prossima batch
                s["idx"] += 1
                s["count"] = 0
                s["t0"] = t1
                s["area0"] = st_obj.busy_area
                s["sum_visit"] = 0.0

                self._check_done()

        # --- Hook: un JOB (sistema) è completo ---
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
            # Stop quando TUTTE le serie hanno raggiunto n_batches
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
                # Se ti interessano anche le medie dei tempi di visita per stazione:
                # "V_A_mean_s_batches": self.sta["A"]["visit_mean_batches"],
                # "V_B_mean_s_batches": self.sta["B"]["visit_mean_batches"],
                # "V_P_mean_s_batches": self.sta["P"]["visit_mean_batches"],
            }
            return out
