from __future__ import annotations
from pathlib import Path
from model.scenario import Scenario
from controller.sim_runner import run_and_save

# Seed iniziale per ogni scenario
SEED0 = 1000

def main():
    config_dir = Path("config")
    out_csv = "out/summary.csv"
    yaml_files = sorted(config_dir.glob("*.yaml"))
    if not yaml_files:
        raise SystemExit("Nessun file .yaml trovato in ./config")

    # Per ogni YAML crea un Scenario.
    # Output: un oggetto Scenario con parametri (domande di servizio, capacità, warmup, durata, repliche, ecc.).
    for i, path in enumerate(yaml_files):

        # Il metodo from_yaml si occupa di fare il parsing/validazione
        scn = Scenario.from_yaml(str(path))

        # Esegue l’esperimento e salva i risultati al CSV.
        # Utilizziamo seed0=1000 + 100*i perché:
        # rende riproducibili gli scenari e
        # cambia il seed tra scenari diversi (1000, 1100, 1200, …).
        # Quindi lascia “spazio” tra seed di base se
        # replications.py fa seed0 + r per le repliche (evita collisioni casuali).
        # L'output res è un dictionary con metriche aggregate (media/stdev di R, X, U, ecc.)
        res = run_and_save(scn, out_csv, seed0=SEED0)

        print(f"{scn.name}: {res}")

if __name__ == "__main__":
    main()
