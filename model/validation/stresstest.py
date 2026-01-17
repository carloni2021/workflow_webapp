import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from model.ecommerce import EcommerceModel
from model.scenario import Scenario
from rndbook.rngs import plantSeeds


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

def plot_step_stress_test(config_path):
    #ATTENZIONE fase transiente diverse lambda, non vengono smaltite le richieste quindi cresce fortissimo
    scn = Scenario.from_yaml(config_path)

    # 1. Setup Lambda e Modello Unico
    lambda_critico = 1.25
    lambdas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.3]  # Esempio di rampa

    step_duration = 2000  # Tempo trascorso per ogni livello di carico

    model = EcommerceModel(scn, seed=42)

    results_R = []
    time_axis = []

    print(f"--- Inizio Simulazione Step-Stress Unica ---")

    current_time = 0
    for i, lam in enumerate(lambdas):
        # Cambiamo il tasso di arrivo "al volo"
        model.set_arrival_rate(lam)

        # Definiamo la fine di questo step
        current_time += step_duration

        # Eseguiamo la simulazione continuando dalla posizione attuale
        # Usiamo un warmup interno per ignorare il "colpo" del cambio di carico
        res = model.run_finite(horizon_s=current_time, warmup_s=current_time - 1000)

        r_val = res.get("R_mean_s", float('nan'))
        results_R.append(r_val)
        time_axis.append(lam)  # Usiamo lambda come asse X per vedere la curva

        status = "STABILE" if lam < lambda_critico else "INSTABILE"
        print(f"Time {current_time}s | Lambda {lam:.2f} | R {r_val:.2f}s | {status}")

    # --- Grafico della Curva di Risposta ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Curva di risposta
    ax1.plot(lambdas, results_R, 'o-', color='tab:red', linewidth=2, label='R medio nel gradino')
    ax1.axvline(x=lambda_critico, color='black', linestyle='--', label='Soglia Saturazione')

    # Formattazione
    ax1.set_yscale('log')
    ax1.set_xlabel('Arrival Rate $lambda$ (req/s)')
    ax1.set_ylabel('Response Time $R$ (s) - Log Scale')
    ax1.set_title('Step-Stress Test: Evoluzione del Sistema in Simulazione Unica')

    # Griglia e Legenda
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    plt.show()

def run_batch_means_sweepST(config_path, lambda_min=0.2, lambda_max=1.4, steps=10):
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

def run_lambda_validation_continuous_rng(config_path):
    scn = Scenario.from_yaml(config_path)

    lambdas = np.linspace(0.20, 1.25, 20)
    results_R = []
    WARMUP = 0.0
    MEASURE = 40000.0

    plantSeeds(1234)
    model = EcommerceModel(scn)

    for lam in lambdas:
        print(f"Simulazione per lambda = {lam:.3f}...", end=" ", flush=True)
        model.set_arrival_rate(lam)
        res = model.run_finite(horizon_s=MEASURE, warmup_s=WARMUP, verbose=False, trace_convergence=True)

        trace_R = res.get("R_convergence_trace", [])
        if trace_R:
            r_mean = trace_R[-1][1]
            results_R.append(r_mean)
            print(f"R medio: {r_mean:.4f}")
        else:
            results_R.append(np.nan)
        MEASURE=MEASURE+40000

    # --- 2. GENERAZIONE GRAFICO ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lambdas, results_R, 'o-', color='tab:green', label='Steady-state R (Continuous RNG)')

    # Granularità asse Y: un segno ogni 10 unità
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Zoom: impostiamo il limite massimo poco sopra il valore massimo trovato
    # (es. se R_max è 192, il limite sarà 200)
    if not np.isnan(results_R).all():
        plt.ylim(0, max(results_R) + 10)

    # Riga critica e ticks asse X
    plt.axvline(x=1.25, color='red', linestyle='--', linewidth=2, label='$\lambda_{crit} = 1.25$')
    current_ticks = list(plt.xticks()[0])
    if 1.25 not in current_ticks:
        plt.xticks(sorted(current_ticks + [1.25]))

    plt.title(f"Validazione: $R$ vs $\lambda$ - Seed Unico Iniziale\nScenario: {scn.name}")
    plt.xlabel("Arrival Rate $\lambda$ (jobs/s)")
    plt.ylabel("Mean Response Time $R$ (s)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    print("\n[INFO] Generazione grafico completata.")
    plt.show()
# Esempio di utilizzo:
# plot_lambda_sweep_explosion("config/1FA_base.yaml")
# plot_stress_transient("config/1FA_stress.yaml")