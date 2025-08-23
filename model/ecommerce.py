
import random
import simpy
from typing import Dict, List, Optional
import statistics as stats

from model.scenario import Scenario
from model.entities import JobRecord
from model.station_ps import ProcessorSharingStation

class EcommerceModel:
    def __init__(self, scenario: Scenario, seed: int = 42):
        self.scenario = scenario
        self.env = simpy.Environment()
        random.seed(seed)
        self.rng = random.Random(seed)

        caps = scenario.capacities
        self.A = ProcessorSharingStation(self.env, "A", caps.get("A", 1))
        self.B = ProcessorSharingStation(self.env, "B", caps.get("B", 1))
        self.P = ProcessorSharingStation(self.env, "P", caps.get("P", 1))

        self.jobs_completed: List[JobRecord] = []

    def _ps_visit(self, station: ProcessorSharingStation, sname: str, job: JobRecord, class_id: str):
        demand = self.scenario.service_demands[sname][class_id]
        if demand <= 0:
            return
        t_in = self.env.now
        done = station.ps_service(demand)
        yield done
        t_out = self.env.now
        job.visit_times[sname] = job.visit_times.get(sname, 0.0) + (t_out - t_in)
        job.wait_times[sname] = job.wait_times.get(sname, 0.0) + 0.0  # PS

    def job_flow(self, jid: int):
        job = JobRecord(id=jid, class_id="Class1", arrival_time=self.env.now)

        # 1) A(Class1) → B
        yield from self._ps_visit(self.A, "A", job, "Class1")
        yield from self._ps_visit(self.B, "B", job, "Class1")

        # 2) A(Class2) → P
        job.class_id = "Class2"
        yield from self._ps_visit(self.A, "A", job, "Class2")
        yield from self._ps_visit(self.P, "P", job, "Class2")

        # 3) A(Class3) → exit
        job.class_id = "Class3"
        yield from self._ps_visit(self.A, "A", job, "Class3")

        job.completion_time = self.env.now
        self.jobs_completed.append(job)

    def arrival_process(self):
        ia_mean = self.scenario.get_interarrival_mean()
        jid = 0
        while True:
            ia = self.rng.expovariate(1.0 / ia_mean) if ia_mean > 0 else float("inf")
            yield self.env.timeout(ia)
            jid += 1
            self.env.process(self.job_flow(jid))

    def run(self) -> Dict[str, float]:
        self.env.process(self.arrival_process())
        self.env.run(until=self.scenario.run_s)

        warmup = self.scenario.warmup_s
        horizon = self.scenario.run_s
        completed = [j for j in self.jobs_completed if j.completion_time and j.completion_time >= warmup]
        duration = max(1e-9, horizon - warmup)

        R = [j.completion_time - j.arrival_time for j in completed]
        R_mean = stats.mean(R) if R else float("nan")
        X = len(completed) / duration

        U_A = self.A.utilization(horizon)
        U_B = self.B.utilization(horizon)
        U_P = self.P.utilization(horizon)

        return {
            "R_mean_s": R_mean,
            "X_jobs_per_s": X,
            "U_A": U_A,
            "U_B": U_B,
            "U_P": U_P,
            "n_completed": len(completed),
        }
