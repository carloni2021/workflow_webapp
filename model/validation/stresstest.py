import matplotlib.pyplot as plt
import numpy as np

from model.scenario import Scenario
from model.ecommerce import EcommerceModel


def plot_stress_transient(config_path):
    # 1. Caricamento dello scenario di stress
    scn = Scenario.from_yaml(config_path)
    scn.interarrival_mean_s=100  # Riduciamo l'interarrivo per stressare il sistema
    model = EcommerceModel(scn, seed=42)

    # 2. Esecuzione con tracciamento della convergenza
    # Usiamo un warmup_s = 0 per vedere l'intera evoluzione dall'istante zero
    print(f"Esecuzione simulazione di stress: {scn.name}...")
    results = model.run_finite(horizon_s=scn.run_s, warmup_s=0.0, trace_convergence=True)

    trace = results["R_convergence_trace"]  # [(time, mean, hw), ...]

    times = [t[0] for t in trace]
    means = [t[1] for t in trace]
    upper_bound = [t[1] + t[2] for t in trace]
    lower_bound = [t[1] - t[2] for t in trace]

    # 3. Generazione del Grafico
    plt.figure(figsize=(12, 6))
    plt.plot(times, means, label='Media Progressiva R (Welford)', color='blue')
    plt.fill_between(times, lower_bound, upper_bound, color='blue', alpha=0.2, label='Confidenza 95%')

    plt.title(f"Validazione Transiente: Evoluzione di R - {scn.name}")
    plt.xlabel("Tempo simulato (s)")
    plt.ylabel("Tempo di risposta medio (s)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Annotazione teorica per la validazione
    if scn.heavy_load:
        plt.annotate('Fase Transiente: Crescita rapida', xy=(times[len(times) // 10], means[len(means) // 10]),
                     xytext=(times[len(times) // 10], means[len(means) // 10] * 1.5),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.show()

def plot_lambda_sweep_explosion(config_path):
    """
    Produce un grafico che mostra l'esplosione del tempo di risposta
    all'aumentare del tasso di arrivo lambda (Stress Test Asintotico).
    """
    # 1. Caricamento scenario base per estrarre le domande di servizio
    scn = Scenario.from_yaml(config_path)

    # Calcolo teorico della soglia di saturazione (1 / D_max)
    # Basato sulle tue medie: A=0.2+0.4+0.1=0.7, B=0.8, P=0.4 (dai tuoi log precedenti)
    # Se B è il collo di bottiglia con D_B = 0.8s, allora lambda_critico = 1.25
    d_max = 0.8  # Domanda di servizio massima (collo di bottiglia)
    lambda_critico = 1.0 / d_max

    # Definiamo un range di lambda che attraversi la soglia critica
    # Es: da 0.2 fino a poco oltre il limite critico
    lambdas = np.linspace(0.2, lambda_critico + 0.2, 15)
    results_R = []
    results_CI = []

    print(f"Inizio sweep di lambda per scenario: {scn.name}")
    print(f"Soglia critica teorica prevista: {lambda_critico:.3f} req/s")

    for lam in lambdas:
        # Istanza del modello per ogni lambda
        model = EcommerceModel(scn, seed=42)
        model.set_arrival_rate(lam)

        # Eseguiamo run_finite con orizzonte fisso (es. 2000s)
        # trace_convergence=True ci serve per ottenere l'ultimo intervallo di confidenza
        res = model.run_finite(horizon_s=2000, warmup_s=200, trace_convergence=True)

        # Prendiamo l'ultima media e l'ultimo half-width calcolati
        if res["R_convergence_trace"]:
            last_t, last_mean, last_hw = res["R_convergence_trace"][-1]
            results_R.append(last_mean)
            results_CI.append(last_hw)
        else:
            results_R.append(float('nan'))
            results_CI.append(0)

    # 3. Generazione del Grafico "a impennata"
    plt.figure(figsize=(10, 6))

    # Plot dei risultati con barre di errore (CI95)
    plt.errorbar(lambdas, results_R, yerr=results_CI, fmt='o-', color='red',
                 ecolor='gray', capsize=5, label='R medio (Simulazione)')

    # Linea verticale per il limite teorico
    plt.axvline(x=lambda_critico, color='black', linestyle='--', alpha=0.7,
                label=f'Limite teorico (1/Dmax = {lambda_critico:.2f})')

    plt.title(f"Stress Test: Esplosione di R al variare di lambda - {scn.name}")
    plt.xlabel("Tasso di arrivo lambda (req/s)")
    plt.ylabel("Tempo di risposta medio R (s)")
    plt.yscale('log')  # Scala logaritmica utile per vedere bene l'impennata
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend()

    # Annotazione del regime di instabilità
    plt.fill_betweenx([min(results_R), max(results_R)], lambda_critico, max(lambdas),
                      color='orange', alpha=0.1, label='Zona Instabilità')

    plt.show()

def run_batch_means_sweepST(config_path, lambda_min=0.2, lambda_max=2, steps=10):
    scn = Scenario.from_yaml(config_path)
    lambdas = np.linspace(lambda_min, lambda_max, steps)

    results_R = []
    results_HW = []  # Half-width per intervallo di confidenza

    for lam in lambdas:
        print(f"Esecuzione Batch Means per λ = {lam:.3f}")
        model = EcommerceModel(scn, seed=1234)

        # Usiamo la tua funzione automatica che stima b via ACF
        #
        series, diag = model.run_batch_means_auto_single_lambda(
            lam=lam,
            n_batches=64,
            n_jobs_calib=50000
        )

        # Calcolo media e intervallo di confidenza dai batch
        Rb = series["R_mean_s_batches"]
        mean_R = np.mean(Rb)
        # Calcolo semi-ampiezza dell'intervallo (Standard Error * t-student)
        hw = 1.96 * (np.std(Rb, ddof=1) / np.sqrt(len(Rb)))

        results_R.append(mean_R)
        results_HW.append(hw)

    return lambdas, results_R, results_HW
# Esempio di utilizzo:
# plot_lambda_sweep_explosion("config/1FA_base.yaml")
# plot_stress_transient("config/1FA_stress.yaml")