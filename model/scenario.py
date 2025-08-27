from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import yaml

"""
    Questo file è il ponte tra i file .yaml e l’oggetto Python Scenario,
    che racchiude tutti i parametri di simulazione (routing indiretto, service demand, risorse, repliche, ecc.).

    Definisce la struttura dati che rappresenta uno scenario di simulazione.
    Gli attributi diventano automaticamente parametri del costruttore.
"""

@dataclass
class Scenario:
    name: str
    service_demands: Dict[str, Dict[str, float]]
    capacities: Dict[str, int] = field(default_factory=lambda: {"A": 1, "B": 1, "P": 1})
    interarrival_mean_s: float = 3.0
    run_s: float = 10000.0
    replications: int = 5
    heavy_load: bool = False

    def get_interarrival_mean(self) -> float:
        # Se heavy_load=True, riduce l’interarrivo medio del 15% → più job al secondo.
        return self.interarrival_mean_s / 1.15 if self.heavy_load else self.interarrival_mean_s

    # Legge un file .yaml e costruisce direttamente l’oggetto Scenario.
    @staticmethod
    def from_yaml(path: str) -> "Scenario":
        with open(path, "r", encoding="utf-8") as f:
            # Usa yaml.safe_load per ottenere un dizionario Python.
            # Espande quel dizionario nei parametri della dataclass (**data).
            # NOTA: L’operatore ** espande quel dizionario come argomenti keyword al costruttore.
            # Quindi il file .yaml DEVE avere chiavi uguali ai nomi degli attributi (name, service_demands, capacities, …).
            data = yaml.safe_load(f)

        # Ritorna l'oggetto scenario creato invocando il costruttore di default
        return Scenario(**data)
