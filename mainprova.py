# run_batch_R.py
from pathlib import Path
from model.scenario import Scenario
from model.ecommerce import EcommerceModel
from engineering.stats import aggregate_vals

def main():
    # --- 1. Carica scenario ---
    config_file = Path("config/scenario_1fa.yaml")   # <-- cambia con il tuo YAML
    scn = Scenario.from_yaml(str(config_file))

    # --- 2. Crea modello ---
    model = EcommerceModel(scn, seed=1234)

    # --- 3. Esegui batch means (solo R) ---
    results = model.run_batch_means_R(n_batches=64, jobs_per_batch=1024)
    # avrai: {"R_mean_s_batches": [R1, R2, ..., R8]}

    # --- 4. Statistiche aggregate ---
    R_batches = results["R_mean_s_batches"]
    agg = aggregate_vals("R_mean_s_batches", R_batches, conf_level=0.95)

    # --- 5. Stampa ---
    print(f"[SCENARIO] {scn.name}")
    print(f"[BATCHES]  {len(R_batches)} valori raccolti")
    print("[RISULTATI]")
    for k, v in agg.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
