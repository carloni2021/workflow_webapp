from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import yaml

@dataclass
class Scenario:
    name: str
    service_demands: Dict[str, Dict[str, float]]
    capacities: Dict[str, int] = field(default_factory=lambda: {"A": 1, "B": 1, "P": 1})
    interarrival_mean_s: float = 3.0
    warmup_s: float = 1000.0
    run_s: float = 10000.0
    replications: int = 5
    heavy_load: bool = False

    def get_interarrival_mean(self) -> float:
        # +15% carico ⇒ interarrivo medio ridotto
        return self.interarrival_mean_s / 1.15 if self.heavy_load else self.interarrival_mean_s

    @staticmethod
    def from_yaml(path: str) -> "Scenario":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Scenario(**data)
