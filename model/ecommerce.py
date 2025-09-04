import simpy
from typing import Dict, List, Optional
import statistics as stats
from rndbook import service_dist

from rndbook.rng_setup import init_rng_for_replication, use_stream
from rndbook.rvgs import Exponential
from rndbook.hyperexp import HyperExp2Balanced

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
            # iat = Exponential(mean_iat)  # estrae l'inter-arrivo esponenziale con media 1/λ
            if self.scenario.arrival_process == "exp":
                iat = Exponential(mean_iat)
            elif self.scenario.arrival_process == "hyperexp":
                iat = HyperExp2Balanced(mean_iat, p=0.10)
            else:
                raise ValueError(f"arrival_kind non supportato: {self.arrival_process}")
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
        kindA=self.scenario.service_dist.get("A")
        kindB=self.scenario.service_dist.get("B")
        kindP=self.scenario.service_dist.get("P")
        d=0.0;

        if sname=="A":
            if kindA == "exp":
                d = self._exp_demand(sname, class_id)
            elif kindA == "erlang":
                mean = float(self.scenario.service_demands.get(station, {}).get(class_id, 0.0))
                d = service_dist.demand_erlang(mean,3)

        if sname=="B":
            if kindB == "exp":
                d = self._exp_demand(sname, class_id)
            elif kindB == "lognorm":
                mean = float(self.scenario.service_demands.get(station, {}).get(class_id, 0.0))
                d = service_dist.demand_lognormal(mean,3)
        if sname=="P":
            if kindP == "exp":
                d = self._exp_demand(sname, class_id)
            elif kindP == "erlang":
                mean = float(self.scenario.service_demands.get(station, {}).get(class_id, 0.0))
                d = service_dist.demand_erlang(mean,3)
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
    def run_finite(self, *, horizon_s: float, warmup_s: float = 0.0,
                   verbose: bool = True) -> Dict:
        """
        Esegue una run a orizzonte finito (transiente) e misura le metriche nella finestra [warmup_s, horizon_s].
        Ritorna:
            - R_mean_s        : tempo di risposta medio nella finestra
            - X_jobs_per_s    : throughput (completamenti / durata finestra)
            - U_A, U_B, U_P   : utilizzazioni medie per stazione nella finestra
            - n_completed     : # job completati nella finestra
            - N_mean          : stima del numero medio di utenti nel sistema via Little (X * R)
            - N_series        : opzionale, lista di (t_centro_finestra, N_i) per studiare la convergenza
        """
        assert horizon_s > 0.0 and horizon_s >= warmup_s

        # reset log locali per la run
        self.jobs_completed = []
        self._arrival_times = []

        # Se non è stato impostato da fuori, usa il valore dallo Scenario
        if self.lambda_req_s <= 0.0:
            try:
                lam_from_scn = 1.0 / float(self.scenario.get_interarrival_mean())
            except Exception:
                lam_from_scn = 1.0 / float(getattr(self.scenario, "interarrival_mean_s", 1.0))
            self.set_arrival_rate(lam_from_scn)

        # --- snapshot delle aree busy all'inizio della finestra di misura ---
        t0 = float(warmup_s)
        snapshot = {"t0": 0.0, "A": 0.0, "B": 0.0, "P": 0.0}

        def _take_snapshot():
            # attende il warmup e poi fotografa le aree
            if t0 > 0.0:
                yield self.env.timeout(t0)
            # aggiorna le aree prima di leggerle
            self.A._area_accumulate();
            self.B._area_accumulate();
            self.P._area_accumulate()
            snapshot["t0"] = self.env.now
            snapshot["A"] = self.A.busy_area
            snapshot["B"] = self.B.busy_area
            snapshot["P"] = self.P.busy_area

        self.env.process(_take_snapshot())

        # --- avvio processi: arrivi + (le visite sono lanciate dal flow alla creazione job) ---
        self.env.process(self._arrival_process())

        # --- run fino all'orizzonte ---
        self.env.run(until=horizon_s)

        # --- misure nella finestra [t0, horizon] ---
        # finalizza integrazione delle aree al tempo corrente
        self.A._area_accumulate();
        self.B._area_accumulate();
        self.P._area_accumulate()

        t_start = snapshot["t0"]
        t_end = float(horizon_s)
        duration = max(t_end - t_start, 1e-12)

        # Completamenti nella finestra
        completed = [j for j in self.jobs_completed if (t_start <= j.completion_time <= t_end)]
        R = [(j.completion_time - j.arrival_time) for j in completed if j.arrival_time >= t_start]
        R_mean = (stats.mean(R) if R else float("nan"))
        X = (len(completed) / duration) if duration > 0 else float("nan")

        # Stima del tasso d'arrivo osservato nella finestra
        arrivals_in = sum((t_start <= t <= t_end) for t in self._arrival_times)
        lambda_hat = arrivals_in / duration if duration > 0 else float("nan")
        lambda_set = float(self.lambda_req_s)

        if verbose:
            print(
                f"[CHECK] lambda_set={lambda_set:.3f}  lambda_hat={lambda_hat:.3f}  arrivals_in={arrivals_in}  duration={int(duration)}s")
            print(f"[DEBUG R] t0={t_start}  horizon={t_end}  duration={duration}  completed={len(completed)}")
            if R:
                j0 = min(completed, key=lambda j: j.completion_time)
                print(
                    f"[DEBUG R] first R in window = {j0.completion_time - j0.arrival_time:.4f}s (comp={j0.completion_time:.2f}, arr={j0.arrival_time:.2f})")

        # Utilizzazioni: differenza d'area / (capacity * durata) nella finestra
        dA = self.A.busy_area - snapshot["A"]
        dB = self.B.busy_area - snapshot["B"]
        dP = self.P.busy_area - snapshot["P"]
        U_A = dA / (self.A.capacity * duration)
        U_B = dB / (self.B.capacity * duration)
        U_P = dP / (self.P.capacity * duration)

        if verbose:
            print(f"[DEBUG U] areas ΔA={dA:.2f}  ΔB={dB:.2f}  ΔP={dP:.2f}")
            # Check teorico (hard-coded per il tuo 1FA base; cambia se usi altri YAML)
            D = {"A": 0.7, "B": 0.8, "P": 0.4}
            U_expected = {s: lambda_set * D[s] for s in D}
            print(
                f"[CHECK] λ={lambda_set:.3f}  U_atteso: A={U_expected['A']:.3f}  B={U_expected['B']:.3f}  P={U_expected['P']:.3f}")
            print(f"[CHECK] U_misurato: A={U_A:.3f}  B={U_B:.3f}  P={U_P:.3f}")

        # Stima L = λ W (numero medio di utenti nel sistema)
        N_mean = (X * R_mean) if (R_mean == R_mean and X == X) else float("nan")

        out = {
            "R_mean_s": R_mean,
            "X_jobs_per_s": X,
            "U_A": U_A, "U_B": U_B, "U_P": U_P,
            "n_completed": len(completed),
            "N_mean": N_mean,
        }

        # --- Serie R(t) cumulativa per analizzare il warmup (nessuna logica per-bin) ---
        jobs_puri = [
            j for j in self.jobs_completed
            if (t_start <= j.completion_time <= t_end) and (j.arrival_time >= t_start)
        ]
        jobs_sorted = sorted(jobs_puri, key=lambda j: j.completion_time)

        R_series_cum = []
        s = 0.0
        for i, j in enumerate(jobs_sorted, start=1):
            s += (j.completion_time - j.arrival_time)
            R_series_cum.append((j.completion_time, s / i))
        out["R_series_cum"] = R_series_cum

        return out

        # out = {
        #     "R_mean_s": R_mean,
        #     "X_jobs_per_s": X,
        #     "U_A": U_A, "U_B": U_B, "U_P": U_P,
        #     "n_completed": len(completed),
        #     "N_mean": N_mean,
        # }

        # # --- Serie R(t): cumulativa e per-bin per analizzare il warmup ---
        # # 1) Media cumulativa del tempo di risposta in funzione del tempo di completamento
        # jobs_sorted = sorted(self.jobs_completed, key=lambda j: j.completion_time)
        # R_series_cum = []
        # s = 0.0
        # n = 0
        # for j in jobs_sorted:
        #     # consideriamo solo i job "puri" della finestra di misura
        #     if not (t_start <= j.completion_time <= t_end):
        #         continue
        #     if j.arrival_time < t_start:
        #         continue
        #     n += 1
        #     s += (j.completion_time - j.arrival_time)
        #     R_series_cum.append((j.completion_time, s / n))
        # out["R_series_cum"] = R_series_cum

        return out

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
