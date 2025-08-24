
import random
import simpy
from typing import Dict, List, Optional
import statistics as stats

from model.scenario import Scenario
from model.entities import JobRecord
from model.station_ps import ProcessorSharingStation
"""
    Crea un ambiente SimPy e tre stazioni Processor Sharing (A, B, P) con capacità da Scenario.
    Genera arrivi Poisson (expovariate con media get_interarrival_mean()).
    Ogni job percorre tre fasi nello stesso JobRecord cambiando class_id:
    Class1: A → B
    Class2: A → P
    Class3: A → (exit)
    Registra i tempi di visita (in PS non c’è coda separata → wait_times[*] = 0).
    A fine run calcola: R_mean_s, X_jobs_per_s, U_A/B/P, n_completed, filtrando i job completati dopo il warmup.
"""

class EcommerceModel:
    def __init__(self, scenario: Scenario, seed: int = 42):
        self.scenario = scenario                 # Scenario con parametri (domande, capacità, tempi, ecc.)
        self.env = simpy.Environment()           # Ambiente di simulazione discreta SimPy (gestisce il tempo simulato)

        random.seed(seed)                        # Imposta il PRNG globale di Python (influenza anche altro codice globale)
        self.rng = random.Random(seed)           # PRNG locale al modello (usato per gli interarrivi)

        capacities = scenario.capacities               # Dizionario capacità per stazione (es. {"A":1,"B":1,"P":1})
        self.A = ProcessorSharingStation(self.env, "A", capacities.get("A", 1))  # Stazione PS A con capacità
        self.B = ProcessorSharingStation(self.env, "B", capacities.get("B", 1))  # Stazione PS B con capacità
        self.P = ProcessorSharingStation(self.env, "P", capacities.get("P", 1))  # Stazione PS P con capacità

        self.jobs_completed: List[JobRecord] = []  # Collezione dei job completati (per calcolo metriche)

    def _ps_visit(self, station: ProcessorSharingStation, sname: str, job: JobRecord, class_id: str):
        demand = self.scenario.service_demands[sname][class_id]  # Domanda di servizio (secondi) per stazione/classe
        if demand <= 0:
            return                              # Se domanda nulla/negativa, salta la visita
        t_in = self.env.now                     # Istante di ingresso alla stazione
        done = station.ps_service(demand)       # Evento SimPy: completa quando la quota PS eroga 'demand' secondi
        yield done                              # Attende il completamento del servizio
        t_out = self.env.now                    # Istante di uscita dalla stazione
        job.visit_times[sname] = job.visit_times.get(sname, 0.0) + (t_out - t_in)  # Tempo trascorso alla stazione
        job.wait_times[sname] = job.wait_times.get(sname, 0.0) + 0.0  # In PS non c'è attesa separata (coda condivisa)

    def job_flow(self, jid: int):
        job = JobRecord(id=jid, class_id="Class1", arrival_time=self.env.now)  # Crea un job con arrivo “ora”

        # 1) Fase Class1: A → B
        yield from self._ps_visit(self.A, "A", job, "Class1")
        yield from self._ps_visit(self.B, "B", job, "Class1")

        # 2) Fase Class2: A → P (riusa lo stesso job cambiando class_id)
        job.class_id = "Class2"
        yield from self._ps_visit(self.A, "A", job, "Class2")
        yield from self._ps_visit(self.P, "P", job, "Class2")

        # 3) Fase Class3: A → uscita
        job.class_id = "Class3"
        yield from self._ps_visit(self.A, "A", job, "Class3")

        job.completion_time = self.env.now      # Timestamp di completamento dell’intero flusso (Class1+2+3)
        self.jobs_completed.append(job)         # Registra per metriche

    def arrival_process(self):
        ia_mean = self.scenario.get_interarrival_mean()  # Interarrivo medio (ridotto del 15% se heavy_load=True)
        jid = 0
        while True:
            # Genera interarrivo ~ Exp(1/ia_mean). Se ia_mean <= 0 → nessun nuovo arrivo (timeout infinito)
            ia = self.rng.expovariate(1.0 / ia_mean) if ia_mean > 0 else float("inf")
            yield self.env.timeout(ia)          # Attende l’interarrivo
            jid += 1
            self.env.process(self.job_flow(jid))# Spawna un nuovo processo job (flusso completo)

    def run(self) -> Dict[str, float]:

        self.env.process(self.arrival_process())         # Avvia il generatore di arrivi
        self.env.run(until=self.scenario.run_s)          # Esegue fino al tempo simulato run_s

        warmup = self.scenario.warmup_s
        horizon = self.scenario.run_s

        # Considera nei calcoli solo i job completati dopo il warmup
        completed = [j for j in self.jobs_completed if j.completion_time and j.completion_time >= warmup]
        duration = max(1e-9, horizon - warmup)           # Durata finestra di misura (evita divisione per 0)

        # Tempo di risposta R = completion - arrival (anche se l'arrivo può essere prima del warmup)
        R = [j.completion_time - j.arrival_time for j in completed]
        R_mean = stats.mean(R) if R else float("nan")    # Se nessun job → NaN
        X = len(completed) / duration                    # Throughput medio (job/s) nella finestra post-warmup

        U_A = self.A.utilization(horizon)                # Utilizzazione stazione A misurata fino a horizon
        U_B = self.B.utilization(horizon)                # Utilizzazione stazione B
        U_P = self.P.utilization(horizon)                # Utilizzazione stazione P

        return {
            "R_mean_s": R_mean,                          # Tempo medio di risposta
            "X_jobs_per_s": X,                           # Throughput medio
            "U_A": U_A, "U_B": U_B, "U_P": U_P,          # Utilizzazioni per stazione
            "n_completed": len(completed),               # Numero job completati nella finestra
        }