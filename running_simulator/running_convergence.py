import re
from pathlib import Path
from model.ecommerce import EcommerceModel
from model.scenario import Scenario
from view.convergence_plot import (
    plot_convergence_R_multi,
    plot_convergence_N_multi,
)
from rndbook.rngs import selectStream, getSeed, putSeed, plantSeeds
from rndbook.rng_setup import STREAMS

DEFAULT_CONFIG_DIR = "config"
RUNS = 5           # quante repliche vuoi sovrapporre
INIT_SEED = 1234   # un solo seed iniziale

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
def _run_overlay_continued_rng(
    scn: Scenario,
    *,
    lam: float,
    measure_s: float,
    warmup_s: float,
    runs: int = RUNS,
    init_seed: int = INIT_SEED,
    outroot: str | Path = "out",
) -> None:
    """
    Esegue 'runs' repliche in sequenza: la run k+1 riparte dai RNG streams
    fermi alla fine della run k. Produce overlay di R(t) e N(t).
    """
    outdir = Path(outroot) / _slug(scn.name)
    outdir.mkdir(parents=True, exist_ok=True)
    scn_slug = _slug(scn.name)

    R_list, N_list = [], []

    # --- RUN 1: parte dal seed iniziale (semina gestita dal modello)
    print(f"[INFO] Seed iniziale usato: {init_seed}")
    model = EcommerceModel(scn, seed=init_seed)
    model.set_arrival_rate(lam)
    res = model.run_finite(horizon_s=measure_s, warmup_s=warmup_s, verbose=False)

    R_cum = res.get("R_series_cum", [])
    N_cum = res.get("N_series_cum", [])
    if not N_cum and R_cum:
        N_cum = [(t, lam * y) for (t, y) in R_cum]
    R_list.append(R_cum)
    N_list.append(N_cum)

    plantSeeds(init_seed)  # per sicurezza, reimposto il seed iniziale

    # Snapshot stato RNG al termine della prima run
    last_state = _snapshot_streams()
    print(f"[INFO] Stato RNG fine-run 1: {last_state}")

    # --- RUN 2..runs: riprendi esattamente da dove si erano fermati i RNG
    for k in range(2, runs + 1):
        # 1) crea il modello della prossima run (eventuali init interni non ti disturbano)
        model = EcommerceModel(scn)
        model.set_arrival_rate(lam)
        # 2) ripristina lo stato RNG *dopo* la costruzione ma *prima* della run
        _restore_streams(last_state)

        res = model.run_finite(horizon_s=measure_s, warmup_s=warmup_s, verbose=False)

        R_cum = res.get("R_series_cum", [])
        N_cum = res.get("N_series_cum", [])
        if not N_cum and R_cum:
            N_cum = [(t, lam * y) for (t, y) in R_cum]
        R_list.append(R_cum)
        N_list.append(N_cum)

        # 3) salva il nuovo stato (avanzato) per la run successiva
        last_state = _snapshot_streams()
        print(f"[INFO] Stato RNG fine-run {k}: {last_state}")

    # --- plot overlay (R e N)
    labels = [f"run {i+1}" for i in range(runs)]

    title_R = f"{scn.name} — R(t) overlay  λ={lam:.3f}"
    png_R = outdir / f"warmup_R_multi_{scn_slug}_lam{lam:.2f}_W{int(warmup_s)}_M{int(measure_s)}.png"
    plot_convergence_R_multi(R_list, lam=lam, scn=scn, title=title_R,
                             labels=labels, outfile=str(png_R), show=False)
    print(f"[OK] Overlay R salvato: {png_R}")

    title_N = f"{scn.name} — N(t) overlay  λ={lam:.3f}"
    png_N = outdir / f"warmup_N_multi_{scn_slug}_lam{lam:.2f}_W{int(warmup_s)}_M{int(measure_s)}.png"
    plot_convergence_N_multi(N_list, lam=lam, scn=scn, title=title_N,
                             labels=labels, outfile=str(png_N), show=False)
    print(f"[OK] Overlay N salvato: {png_N}")

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
        measure_s = 86_400.0  # 1 giorno
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
