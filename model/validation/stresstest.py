import matplotlib.pyplot as plt
from model.scenario import Scenario
from model.ecommerce import EcommerceModel


def plot_stress_transient(config_path):
    # 1. Caricamento dello scenario di stress
    scn = Scenario.from_yaml(config_path)
    scn.interarrival_mean_s=10  # Riduciamo l'interarrivo per stressare il sistema
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

# Esempio di utilizzo:
# plot_stress_transient("config/1FA_stress.yaml")