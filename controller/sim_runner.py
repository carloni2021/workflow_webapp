from __future__ import annotations
from typing import Dict
from model.scenario import Scenario
from controller.replications import run_experiment
from view.reporters import write_csv_row

# Funzione “orchestratrice"
# Tipo di ritorno: un dizionario Dict[str, float] con le metriche aggregate (es. tempi di risposta medi, throughput, utilizzo…).
def run_and_save(scn: Scenario, out_csv: str, seed0: int = 1234) -> Dict[str, float]:
    # run_experiment chiama la simulazione vera e propria
    # Fa girare il modello un numero di volte pari a "replications"
    # Per ogni replica crea un EcommerceModel (in model/ecommerce.py), lo esegue, raccoglie i risultati.
    # Poi calcola le medie e le stdev delle metriche e restituisce un dizionario.
    agg = run_experiment(scn, seed0=seed0)
    # Scrive una riga con i valori del dict agg.
    # Così, se si hanno più scenari o più run, ogni risultato finisce in una riga del CSV.
    write_csv_row(out_csv, agg, header_if_new=True)

    return agg