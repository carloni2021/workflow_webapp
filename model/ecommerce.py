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
        Crea un ambiente SimPy e tre stazioni Processor Sharing (A, B, P) con capacità da Scenario.
        Genera arrivi Poisson (expovariate con media get_interarrival_mean()).
        Ogni job percorre tre fasi nello stesso JobRecord cambiando class_id:
        Class1: A → B
        Class2: A → P
        Class3: A → (exit)
        Registra i tempi di visita (in PS non c’è coda separata → wait_times[*] = 0).
        A fine run calcola: R_mean_s, X_jobs_per_s, U_A/B/P, n_completed, filtrando i job completati dopo il warmup.
    """

    def __init__(self, scenario: Scenario, seed: int = 42):
        self.scenario = scenario                 # Scenario con parametri (domande, capacità, tempi, ecc.)
        self.env = simpy.Environment()           # Ambiente di simulazione discreta SimPy (gestisce il tempo simulato)

        # rndbook.seed(seed)                        # Imposta il PRNG globale di Python (influenza anche altro codice globale)
        # self.rng = rndbook.Random(seed)           # PRNG locale al modello (usato per gli interarrivi)
        rng_setup.init_rng_for_replication(seed, 0)

        # Ogni ProcessorSharingStation è una risorsa PS:
        # se ci sono n job attivi e capacità c, ciascuno riceve c/n di servizio in parallelo (niente coda separata).
        capacities = scenario.capacities               # Dizionario capacità per stazione (es. {"A":1,"B":1,"P":1})
        self.A = ProcessorSharingStation(self.env, "A", capacities.get("A", 1))  # Stazione PS A con capacità
        self.B = ProcessorSharingStation(self.env, "B", capacities.get("B", 1))  # Stazione PS B con capacità
        self.P = ProcessorSharingStation(self.env, "P", capacities.get("P", 1))  # Stazione PS P con capacità

        self.jobs_completed: List[JobRecord] = []  # Collezione dei job completati (per calcolo metriche)

    def _ps_visit(self, station: ProcessorSharingStation, sname: str, job: JobRecord, class_id: str):
        demand = self.scenario.service_demands[sname][class_id]  # Domanda di servizio (secondi) per stazione/classe. Questo è il tempo di lavoro richiesto dal job.
        if demand <= 0:
            return                              # Se domanda nulla/negativa, salta la visita
        t_in = self.env.now                     # Istante di ingresso del job alla stazione
        done = station.ps_service(demand)       # Evento SimPy: completa quando la quota PS eroga 'demand' secondi
        yield done                              # Attende il completamento del servizio
        t_out = self.env.now                    # Istante di uscita dalla stazione

        job.visit_times[sname] = job.visit_times.get(sname, 0.0) + (t_out - t_in)  # Tempo trascorso alla stazione
        job.wait_times[sname] = job.wait_times.get(sname, 0.0) + 0.0  # In PS non c'è attesa separata (coda condivisa)

    def job_flow(self, jid: int):

        # Creazione di un nuovo record (JobRecord) che memorizzerà i tempi di visita e attesa.
        # Importante notare che:
        # arrival_time = self.env.now → l’arrivo avviene esattamente nel momento in cui questo processo è stato generato da arrival_process
        job = JobRecord(id=jid, class_id="Class1", arrival_time=self.env.now)

        # la funzione ,_ps_visit() gestisce il servizio PS: attende finché il job non riceve tutto il suo “demand” e registra i tempi spesi.

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
        # Registra il job completato alla lista dei job completati per metriche
        self.jobs_completed.append(job)

    # Processo degli arrivi (il generatore degli arrivi durante l'esecuzione)
    def arrival_process(self):
        interarrival_mean = self.scenario.get_interarrival_mean()       # Interarrivo medio preso dallo scenario (ridotto del 15% se heavy_load=True)
        jid = 0     # Job id, per dare un id progressivo ai job
        # Questo processo non si ferma da solo: genera job per tutta la durata della simulazione.
        # Infatti nella funzione .run() si ha l'istruzione env.run(until=...) che tronca quando si raggiunge l’orizzonte.
        while True:
            # Genera interarrivo ~ Exp(1/interarrival_mean). Se interarrival_mean <= 0 → nessun nuovo arrivo (timeout infinito)
            #
            # ia = self.rng.expovariate(1.0 / interarrival_mean) if interarrival_mean > 0 else float("inf")
            #
            rng_setup.use_stream("arrivals")
            ia = rvgs.Exponential(interarrival_mean)
            # Attende l’interarrivo
            # Questo è il cuore della logica SimPy: il processo aspetta per ia unità di tempo simulato prima di proseguire.
            # Così i job arrivano in tempi casuali esponenziali (un processo di Poisson).
            yield self.env.timeout(ia)
            jid += 1        # Viene incrementato l'id dei job
            # Si avvia un nuovo processo SimPy che segue il flusso del job completo (A→B, A→P, A→uscita)
            self.env.process(self.job_flow(jid))

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