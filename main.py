# main.py
from __future__ import annotations

from model.validation.stresstest import plot_stress_transient
from running_simulator.running_validation import run_phase_validation
from running_simulator.running_convergence import run_phase_convergence
from running_simulator.running_steady_state import run_single_lambda_batch_means

DEFAULT_CONFIG_DIR = "config"

# ------------------------------- ENTRYPOINT ----------------------------------

def main() -> None:
    config_dir = DEFAULT_CONFIG_DIR

    #print(f"[INFO] finite horizon run - validation study")
    #run_phase_validation(config_dir=config_dir)

    #print(f"[INFO] finite horizon run - convergence study")
    #run_phase_convergence(config_dir=config_dir)

    #print(f"[INFO] batch-means one-λ | λ=0.33")
    #run_single_lambda_batch_means(config_dir=config_dir, n_batches=64)

    #la validazione va svolta sullo stato stazionario, eseguiamo uno stress test di quest'ultimo
    plot_stress_transient(config_path="config/scenario_1fa_base.yaml")

if __name__ == "__main__":
    main()
