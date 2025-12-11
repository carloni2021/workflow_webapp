import re
from pathlib import Path
from model.ecommerce import EcommerceModel
from model.scenario import Scenario
from view.convergence_plot import plot_convergence_R_multi, plot_convergence_N_multi
from rndbook.rngs import selectStream, getSeed, putSeed, plantSeeds
from rndbook.rng_setup import STREAMS
import numpy as np
import matplotlib.pyplot as plt
DEFAULT_CONFIG_DIR = "config"
RUNS = 5           # quante repliche vuoi sovrapporre
INIT_SEED = 1234   # un solo seed iniziale
STD_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# ---------------- utils ----------------
def _slug(txt: str | None) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '-', str(txt or "scenario")).strip('-').lower()

# ------------- snapshot/restore RNG streams ----------------
def _snapshot_streams(stream_names=None) -> dict[str, int]:
    stream_names = list(stream_names) if stream_names is not None else list(STREAMS.keys())
    snap = {}
    for name in stream_names:
        selectStream(STREAMS[name])
        snap[name] = int(getSeed())
    return snap

def _restore_streams(state: dict[str, int]) -> None:
    for name, seed in state.items():
        selectStream(STREAMS[name])
        putSeed(int(seed))

# ---------------- core: run "in continuità" sui RNG streams ----------------
# In running_convergence.py (assicurati di avere gli import: numpy, matplotlib.pyplot)

def _plot_ci_run(times, means, hws, title, ylabel, filename, color_line, color_fill):
    if len(times) == 0: return
    t = np.array(times)
    m = np.array(means)
    h = np.array(hws)

    plt.figure(figsize=(8, 5))
    plt.plot(t, m, color=color_line, linewidth=1.5, label='Mean')
    plt.fill_between(t, m - h, m + h, color=color_fill, alpha=0.25, label='95% CI')

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[OK] {filename}")


def _run_overlay_continued_rng(
        scn: Scenario, *, lam: float, measure_s: float, warmup_s: float,
        runs: int = RUNS, init_seed: int = INIT_SEED, outroot: str | Path = "out",
) -> None:
    outdir = Path(outroot) / _slug(scn.name)
    outdir.mkdir(parents=True, exist_ok=True)
    scn_slug = _slug(scn.name)

    # 1. Setup Sotto-cartelle (R e N)
    single_R_dir = outdir / "single_run_R"
    single_R_dir.mkdir(parents=True, exist_ok=True)

    single_N_dir = outdir / "single_run_N"  # <--- NUOVO
    single_N_dir.mkdir(parents=True, exist_ok=True)  # <--- NUOVO

    # 2. Init Seed
    plantSeeds(init_seed)
    last_state = _snapshot_streams()

    # Liste per i dati aggregati (R e N)
    all_runs_R_means = []
    all_runs_N_means = []  # <--- NUOVO

    for k in range(1, runs + 1):
        # Setup Modello
        model = EcommerceModel(scn)
        model.set_arrival_rate(lam)
        _restore_streams(last_state)

        # --- RUN ---
        res = model.run_finite(horizon_s=measure_s, warmup_s=warmup_s,
                               verbose=False, trace_convergence=True)

        last_state = _snapshot_streams()

        # Determina il colore per questa Run (lo usiamo per entrambi i grafici)
        color_idx = (k - 1) % len(STD_COLORS)
        this_run_color = STD_COLORS[color_idx]

        # ---------------- GESTIONE R (Tempo di Risposta) ----------------
        trace_R = res.get("R_convergence_trace", [])
        if trace_R:
            times = [x[0] for x in trace_R]
            means = [x[1] for x in trace_R]
            hws = [x[2] for x in trace_R]

            all_runs_R_means.append(list(zip(times, means)))

            fname_R = single_R_dir / f"Run{k}_R_CI_{scn_slug}.png"
            _plot_ci_run(times, means, hws,
                         title=f"Run {k} - R(t) ± Internal CI",
                         ylabel="Response Time (s)",
                         filename=str(fname_R),
                         color_line=this_run_color,
                         color_fill=this_run_color)

        # ---------------- GESTIONE N (Utenti nel Sistema) ----------------
        trace_N = res.get("N_convergence_trace", [])  # <--- NUOVO
        if trace_N:
            times_n = [x[0] for x in trace_N]
            means_n = [x[1] for x in trace_N]
            hws_n = [x[2] for x in trace_N]

            all_runs_N_means.append(list(zip(times_n, means_n)))

            fname_N = single_N_dir / f"Run{k}_N_CI_{scn_slug}.png"
            _plot_ci_run(times_n, means_n, hws_n,
                         title=f"Run {k} - N(t) ± Internal CI",
                         ylabel="Avg Users (N)",
                         filename=str(fname_N),
                         color_line=this_run_color,
                         color_fill=this_run_color)

    # 3. Plot Aggregati (Overlay)

    # Overlay R
    png_overlay_R = outdir / f"Overlay_R_Multi_{scn_slug}.png"
    plot_convergence_R_multi(all_runs_R_means, lam=lam, scn=scn,
                             title=f"Overlay {runs} Runs - R(t)",
                             outfile=str(png_overlay_R), show=False)

    # Overlay N
    png_overlay_N = outdir / f"Overlay_N_Multi_{scn_slug}.png"  # <--- NUOVO
    plot_convergence_N_multi(all_runs_N_means, lam=lam, scn=scn,  # <--- NUOVO
                             title=f"Overlay {runs} Runs - N(t)",
                             outfile=str(png_overlay_N), show=False)
# ---------------- entrypoint ----------------
def run_phase_convergence(config_dir: str = DEFAULT_CONFIG_DIR) -> None:
    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(Path(config_dir).glob("*.y*ml"))
    if not yaml_files:
        print(f"[WARN] Nessun YAML trovato in '{config_dir}'.")
        return

    for path in yaml_files:
        scn = Scenario.from_yaml(str(path))
        print(f"[SCN]\n\n {scn.get_heavy_load()}\n\n")
        lam = 1.0 / (float(scn.get_interarrival_mean()))
        print(f"[SCN]\n\n {lam}\n\n")
        measure_s = 20_000.0  # 1 giorno
        warmup_s = 0.0

        print(f"[USO] warmup — overlay R(t), N(t) con continuità RNG (runs={RUNS}, seed0={INIT_SEED})")
        _run_overlay_continued_rng(
            scn,
            lam=lam,
            measure_s=measure_s,
            warmup_s=warmup_s,
            runs=RUNS,
            init_seed=INIT_SEED,
            outroot="out",
        )
