from __future__ import annotations
import math
import statistics as stats
from typing import Dict, List
from model.ecommerce import EcommerceModel
from model.scenario import Scenario

def run_experiment(scn: Scenario, seed0: int = 1234) -> Dict[str, float]:
    results: List[Dict[str, float]] = []
    # Esegue N repliche (N = scn.replications).
    for r in range(scn.replications):
        # Per ogni replica costruisce un EcommerceModel con seed diverso (seed0 + r)
        # per rendere le repliche indipendenti ma riproducibili.
        model = EcommerceModel(scn, seed=seed0 + r)
        # Chiama model.run() che restituisce un dict di metriche
        # Dopodiché, aggiunge il dict alla lista results.
        results.append(model.run())

    # agg_metrics conterrà le metriche aggregate e verrà restituito alla fine.
    agg_metrics: Dict[str, float] = {}

    # Se ci sono risultati, prende l’insieme delle chiavi dalla prima replica.
    keys = results[0].keys() if results else []

    #Per ogni metrica:
    for key in keys:
        # Si costruisce vals -> la lista dei valori non-NaN per quella metrica, raccolti da tutte le repliche.
        # Si prendono le repliche che hanno un valore non-NaN.
        # Il filtro isinstance(..., float) and math.isnan(...) evita di chiamare isnan su numeri non-float:
        # quindi, se il valore non è float, a prescindere non viene scartato (ok per int/bool).
        vals = [res[key] for res in results if not(isinstance(res[key], float) and math.isnan(res[key]))]

        # Calcola la media dei valori se c’è almeno un valore; altrimenti NaN.
        agg_metrics[f"{key}_mean"] = stats.mean(vals) if vals else float("nan")
        # deviazione standard campionaria (statistics.stdev, usa n−1) se ci sono >1 valori; altrimenti 0.0 (scelta pragmatica per “nessuna variabilità osservabile”).
        agg_metrics[f"{key}_stdev"] = stats.stdev(vals) if len(vals) > 1 else 0.0

    # Aggiunge una colonna descrittiva con il nome scenario (e il suffisso se heavy_load=True)
    agg_metrics["scenario"] = scn.name + (" +15% load" if scn.heavy_load else "")
    return agg_metrics
