# main.py
from __future__ import annotations

from matplotlib import pyplot as plt

from model.validation.stresstest import plot_stress_transient, plot_step_stress_test, \
    run_lambda_validation_continuous_rng, \
    run_batch_means_sweepST
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
    run_phase_convergence(config_dir=config_dir)

    #print(f"[INFO] batch-means one-λ | λ=0.33")
    #run_single_lambda_batch_means(config_dir=config_dir, n_batches=64)

    #la validazione va svolta sullo stato stazionario, eseguiamo uno stress test di quest'ultimo
    #plot_stress_transient(config_path="config/scenario_1fa_base.yaml")
    #plot_step_stress_test(config_path="config/scenario_1fa_base.yaml")

    #run_batch_means_sweepST(config_path="config/scenario_1fa_base.yaml")
    # Carichiamo lo scenario base (1FA)
    config_path = "config/scenario_1fa_base.yaml"

    # Eseguiamo lo sweep: da lambda molto basso (0.1) a oltre la saturazione (1.4)

    """
    lambdas, r_means, r_hws = run_batch_means_sweepST(config_path=config_path)
    # Visualizzazione dei risultati
    plt.figure(figsize=(10, 6))
    # Disegniamo la curva con le barre d'errore (intervalli di confidenza)
    plt.errorbar(lambdas, r_means, yerr=r_hws, fmt='-o', capsize=5, label='R medio (Batch Means)')
    # Aggiungiamo il limite asintotico inferiore (1.9s)
    plt.axhline(y=1.9, color='g', linestyle='--', label='Limite teorico R_min (1.9s)')
    # Aggiungiamo il limite di saturazione teorico (lambda = 1.25)
    plt.axvline(x=1.25, color='r', linestyle=':', label='Saturazione teorica (lambda=1.25)')
    plt.title("Sweep di Validazione: Tempo di Risposta vs Carico Arrivi")
    plt.xlabel("Tasso di Arrivo (lambda - req/s)")
    plt.ylabel("Tempo di Risposta Medio (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
"""

    # Generazione del grafico
    #run_lambda_validation_continuous_rng(config_path="config/scenario_1fa_base.yaml")

    #verifica della distribuzione iper-esponenziale

if __name__ == "__main__":
    main()
