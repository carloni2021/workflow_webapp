from __future__ import annotations
from pathlib import Path
from model.scenario import Scenario
from controller.sim_runner import run_and_save
from view import plots_extended  # 👈 nuovo import

def main():
    config_dir = Path("config")
    out_csv = "out/summary.csv"
    yaml_files = sorted(config_dir.glob("*.yaml"))

    if not yaml_files:
        raise SystemExit("Nessun file .yaml trovato in ./config")

    for i, path in enumerate(yaml_files):
        scn = Scenario.from_yaml(str(path))
        res = run_and_save(scn, out_csv, seed0=1000 + 100*i)
        print(f"{scn.name}: {res}")

    # genera e mostra i grafici estesi
    plots_extended.main(show=True)

if __name__ == "__main__":
    main()
