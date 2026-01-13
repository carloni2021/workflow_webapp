# main.py
from __future__ import annotations

from matplotlib import pyplot as plt

from model.validation.stresstest import plot_stress_transient, plot_lambda_sweep_explosion, run_batch_means_sweepST
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
    plot_lambda_sweep_explosion(config_path="config/scenario_1fa_base.yaml")

    # Generazione del grafico
    lambdas, R, hw = run_batch_means_sweepST("config/scenario_1fa_base.yaml")
    plt.errorbar(lambdas, R, yerr=hw, fmt='-o', capsize=5, label="Batch Means R ± CI95")
    plt.xlabel("λ (Arrival Rate)")
    plt.ylabel("R (Response Time)")
    plt.title("Studio Steady-State al variare di λ")
    plt.grid(True)
    plt.show()
    #verifica della distribuzione iper-esponenziale

if __name__ == "__main__":
    main()
