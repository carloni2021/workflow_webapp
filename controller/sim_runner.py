from __future__ import annotations
from typing import Dict
from model.scenario import Scenario
from controller.replications import run_experiment
from view.reporters import write_csv_row


# Tipo di ritorno: un dizionario Dict[str, float] con le metriche aggregate (es. tempi di risposta medi, throughput, utilizzo…).
def run_and_save(scn: Scenario, out_csv: str, seed0: int = 1234) -> Dict[str, float]:
    """
    Esegue una simulazione multi-replica e salva i risultati su file.
    Si tratta di unaFunzione “orchestratrice"

    Parametri
    ----------
    scenario_file : str
      Percorso al file JSON (o altro formato supportato) che descrive lo scenario.
    output_file : str
      Percorso del file di output (CSV/JSON) in cui salvare i risultati delle repliche.
    n_reps : int, opzionale
      Numero di repliche Monte Carlo da eseguire (default = 10).
    seed0 : int, opzionale
      Seed iniziale per il generatore pseudo-casuale (default = 42).

    Effetti
    -------
    - Carica lo scenario da `scenario_file`.
    - Esegue `run_experiment` con il numero di repliche richiesto.
    - Salva il DataFrame risultante in `output_file`.

    Note
    ----
    - È il punto di ingresso tipico da linea di comando (`python sim_runner.py ...`).
    - Consente di mantenere una pipeline completa: carica → simula → salva.
    """

    # run_experiment chiama la simulazione vera e propria
    # Fa girare il modello un numero di volte pari a "replications"
    # Per ogni replica crea un EcommerceModel (in model/ecommerce.py), lo esegue, raccoglie i risultati.
    # Poi calcola le medie e le stdev delle metriche e restituisce un dizionario.
    agg = run_experiment(scn, seed0=seed0)

    # Scrive una riga con i valori del dict agg.
    # Così, se si hanno più scenari o più run, ogni risultato finisce in una riga del CSV.
    write_csv_row(out_csv, agg, header_if_new=True)

    return agg