from __future__ import annotations
from typing import Dict
from model.scenario import Scenario
from controller.replications import run_experiment
from view.reporters import write_csv_row

def run_and_save(scn: Scenario, out_csv: str, seed0: int = 1234) -> Dict[str, float]:
    agg = run_experiment(scn, seed0=seed0)
    write_csv_row(out_csv, agg, header_if_new=True)
    return agg