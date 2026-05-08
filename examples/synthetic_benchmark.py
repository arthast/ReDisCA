#!/usr/bin/env python3
"""Compact synthetic ReDisCA benchmark.

The synthetic data follow the structure used in the ReDisCA simulations:

1. For every planted source, condition time series are generated as ``S = M @ Z``.
   ``Z`` is Gaussian noise low-pass filtered at 2 Hz and ``M`` is a fixed random
   condition-mixing matrix for that source.
2. The source time series define the true source RDM.
3. The target RDM is the true RDM plus symmetric noise.
4. Sensor data are a mixture of several source topographies plus 1/f-like
   sensor noise averaged over repeated trials.

The script saves one CSV with recovery metrics and two figures:
``source_recovery_metrics.png`` and ``source_recovery_example.png``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

from redisca import fit_redisca
from redisca.summary import best_component_by_pearson


# =============================================================================
# User settings
# =============================================================================

OUTPUT_ROOT = ROOT / "examples" / "repro_outputs" / "synthetic_benchmark"

N_ITER = 30
SNR_LEVELS = [0.2, 1.0]

N_CONDITIONS = 6
N_SOURCES = 4
N_CHANNELS = 32
N_TIMEPOINTS = 250
N_TRIALS = 20
SFREQ = 100.0

SOURCE_CUTOFF_HZ = 2.0
RDM_NOISE_SCALE = 0.25
TOPOGRAPHY_NOISE_SCALE = 0.15
RANDOM_STATE = 0


METRICS = [
    ("true_rdm_corr", "RDM corr"),
    ("pattern_corr", "Pattern corr"),
    ("filter_corr", "Filter corr"),
]


def upper_triangle_vector(matrix: np.ndarray) -> np.ndarray:
    """Return the upper-triangular off-diagonal entries of a square matrix."""
    return np.asarray(matrix[np.triu_indices_from(matrix, k=1)], dtype=float)


def normalize_rdm(rdm: np.ndarray) -> np.ndarray:
    """Make an RDM non-negative with zero diagonal and max distance equal to 1."""
    rdm = np.asarray(rdm, dtype=float).copy()
    np.fill_diagonal(rdm, 0.0)
    off_diag = upper_triangle_vector(rdm)
    min_value = float(off_diag.min())
    if min_value < 0.0:
        rows, cols = np.triu_indices_from(rdm, k=1)
        rdm[rows, cols] -= min_value
        rdm[cols, rows] -= min_value
    max_value = float(upper_triangle_vector(rdm).max())
    if max_value > 0.0:
        rdm /= max_value
    np.fill_diagonal(rdm, 0.0)
    return rdm


def rdm_pearson(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Pearson correlation between the upper triangles of two RDMs."""
    lhs_vec = upper_triangle_vector(lhs)
    rhs_vec = upper_triangle_vector(rhs)
    if min(float(lhs_vec.std()), float(rhs_vec.std())) < 1e-12:
        return 0.0
    return float(np.corrcoef(lhs_vec, rhs_vec)[0, 1])


def abs_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Absolute Pearson correlation between two vectors."""
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if min(float(lhs.std()), float(rhs.std())) < 1e-12:
        return 0.0
    return abs(float(np.corrcoef(lhs, rhs)[0, 1]))


def sample_source_timeseries(
    n_conditions: int,
    n_timepoints: int,
    rng: np.random.Generator,
    *,
    mixing: np.ndarray | None = None,
    sfreq: float = SFREQ,
    cutoff_hz: float = SOURCE_CUTOFF_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate source activations with the article's ``S = M @ Z`` recipe."""
    if mixing is None:
        mixing = rng.standard_normal((n_conditions, n_conditions))

    sos = butter(6, cutoff_hz, btype="lowpass", fs=sfreq, output="sos")
    latent = rng.standard_normal((n_conditions, n_timepoints))
    latent = sosfiltfilt(sos, latent, axis=-1)

    source = mixing @ latent
    source -= source.mean(axis=1, keepdims=True)
    source /= source.std(axis=1, keepdims=True) + 1e-12
    return source, mixing


def source_rdm_from_timeseries(source_timeseries: np.ndarray) -> np.ndarray:
    """Build a source RDM from condition-wise mean squared differences."""
    source_timeseries = np.asarray(source_timeseries, dtype=float)
    n_conditions = source_timeseries.shape[0]
    rdm = np.zeros((n_conditions, n_conditions), dtype=float)
    for first in range(n_conditions):
        for second in range(first + 1, n_conditions):
            distance = float(
                np.mean((source_timeseries[first] - source_timeseries[second]) ** 2)
            )
            rdm[first, second] = distance
            rdm[second, first] = distance
    return normalize_rdm(rdm)


def make_noisy_target_rdm(
    true_rdm: np.ndarray,
    rng: np.random.Generator,
    *,
    noise_scale: float = RDM_NOISE_SCALE,
) -> np.ndarray:
    """Add symmetric target-RDM noise while preserving valid RDM structure."""
    noise = rng.standard_normal(true_rdm.shape)
    noise = 0.5 * (noise + noise.T)
    np.fill_diagonal(noise, 0.0)
    scale = noise_scale * (float(upper_triangle_vector(true_rdm).std()) + 1e-12)
    return normalize_rdm(true_rdm + scale * noise)


def sample_topographies(
    n_channels: int,
    n_sources: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample separated unit-norm sensor topographies."""
    raw = rng.standard_normal((n_channels, n_sources))
    topographies, _ = np.linalg.qr(raw)
    return topographies[:, :n_sources]


def colored_sensor_noise(
    n_channels: int,
    n_timepoints: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate simple 1/f-like sensor noise."""
    white = rng.standard_normal((n_channels, n_timepoints))
    spectrum = np.fft.rfft(white, axis=-1)
    freqs = np.fft.rfftfreq(n_timepoints, d=1.0 / SFREQ)
    weights = 1.0 / np.sqrt(np.maximum(freqs, 0.5))
    noise = np.fft.irfft(spectrum * weights, n=n_timepoints, axis=-1)
    noise -= noise.mean(axis=-1, keepdims=True)
    noise /= noise.std(axis=-1, keepdims=True) + 1e-12
    return noise


def rms(values: np.ndarray) -> float:
    """Root mean square of an array."""
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def make_synthetic_dataset(
    rng: np.random.Generator,
    *,
    snr: float,
    n_conditions: int = N_CONDITIONS,
    n_sources: int = N_SOURCES,
    n_channels: int = N_CHANNELS,
    n_timepoints: int = N_TIMEPOINTS,
    n_trials: int = N_TRIALS,
    rdm_noise_scale: float = RDM_NOISE_SCALE,
    topography_noise_scale: float = TOPOGRAPHY_NOISE_SCALE,
    mixings: list[np.ndarray] | None = None,
) -> dict[str, object]:
    """Create one multi-source synthetic sensor dataset."""
    topographies = sample_topographies(n_channels, n_sources, rng)
    noisy_topographies = topographies.copy()
    for src in range(n_sources):
        noisy_topographies[:, src] += topography_noise_scale * rng.standard_normal(
            n_channels
        )

    if mixings is None:
        mixings = [None] * n_sources

    source_series = []
    true_rdms = []
    target_rdms = []
    source_mixings = []
    for src in range(n_sources):
        series, mixing = sample_source_timeseries(
            n_conditions,
            n_timepoints,
            rng,
            mixing=mixings[src],
        )
        true_rdm = source_rdm_from_timeseries(series)
        source_series.append(series)
        source_mixings.append(mixing)
        true_rdms.append(true_rdm)
        target_rdms.append(
            make_noisy_target_rdm(true_rdm, rng, noise_scale=rdm_noise_scale)
        )

    source_series_arr = np.stack(source_series, axis=0)
    signal = np.zeros((n_conditions, n_channels, n_timepoints), dtype=float)
    for condition in range(n_conditions):
        for src in range(n_sources):
            signal[condition] += np.outer(
                noisy_topographies[:, src],
                source_series_arr[src, condition],
            )

    signal_rms = rms(signal)
    X = signal.copy()
    for _ in range(n_trials):
        noise = np.stack(
            [
                colored_sensor_noise(n_channels, n_timepoints, rng)
                for _ in range(n_conditions)
            ],
            axis=0,
        )
        X += (signal_rms / max(float(snr), 1e-12)) * noise / rms(noise) / n_trials

    return {
        "X": X,
        "true_rdms": true_rdms,
        "target_rdms": target_rdms,
        "topographies": topographies,
        "source_series": source_series_arr,
        "mixings": source_mixings,
    }


def evaluate_source(
    dataset: dict[str, object],
    source_index: int,
) -> dict[str, float | int]:
    """Fit ReDisCA for one target RDM and measure source recovery."""
    X = dataset["X"]
    target_rdm = dataset["target_rdms"][source_index]
    true_rdm = dataset["true_rdms"][source_index]
    true_topography = dataset["topographies"][:, source_index]

    result = fit_redisca(X, target_rdm)
    component = best_component_by_pearson(result)
    recovered_rdm = result.component_rdms[component]

    return {
        "source_index": int(source_index),
        "component": int(component),
        "lambda": float(result.lambdas[component]),
        "target_rdm_corr": float(result.pearson_scores[component]),
        "true_rdm_corr": rdm_pearson(recovered_rdm, true_rdm),
        "pattern_corr": abs_corr(result.A[:, component], true_topography),
        "filter_corr": abs_corr(result.W[:, component], true_topography),
    }


def run_synthetic_benchmark(
    *,
    n_iter: int = N_ITER,
    snr_levels: list[float] | tuple[float, ...] = tuple(SNR_LEVELS),
    n_channels: int = N_CHANNELS,
    n_timepoints: int = N_TIMEPOINTS,
    n_sources: int = N_SOURCES,
    n_conditions: int = N_CONDITIONS,
    n_trials: int = N_TRIALS,
    rdm_noise_scale: float = RDM_NOISE_SCALE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run Monte Carlo recovery trials."""
    rng = np.random.default_rng(random_state)
    mixings = [
        rng.standard_normal((n_conditions, n_conditions))
        for _ in range(n_sources)
    ]
    records: list[dict[str, float | int]] = []

    for trial in range(n_iter):
        for snr in snr_levels:
            dataset = make_synthetic_dataset(
                rng,
                snr=snr,
                n_conditions=n_conditions,
                n_sources=n_sources,
                n_channels=n_channels,
                n_timepoints=n_timepoints,
                n_trials=n_trials,
                rdm_noise_scale=rdm_noise_scale,
                mixings=mixings,
            )
            for source_index in range(n_sources):
                row = evaluate_source(dataset, source_index)
                row["trial"] = int(trial)
                row["snr"] = float(snr)
                records.append(row)

    return pd.DataFrame(records)


def plot_metric_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Plot recovery metrics across SNR levels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snr_levels = list(dict.fromkeys(float(value) for value in df["snr"]))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(13, 4), sharey=True)

    for ax, (metric, title) in zip(axes, METRICS):
        samples = [
            df.loc[df["snr"] == snr, metric].to_numpy(dtype=float)
            for snr in snr_levels
        ]
        ax.boxplot(samples, showfliers=False)
        ax.set_xticks(
            np.arange(1, len(snr_levels) + 1),
            [f"{snr:.1f}" for snr in snr_levels],
        )
        ax.set_xlabel("SNR")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Correlation")
    fig.suptitle("Synthetic source recovery", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_rdm_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    *,
    labels: list[str],
) -> plt.AxesImage:
    """Plot one normalized RDM panel."""
    image = ax.imshow(normalize_rdm(matrix), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(labelsize=8)
    return image


def normalized(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to max absolute value 1 for display."""
    vector = np.asarray(vector, dtype=float)
    scale = float(np.max(np.abs(vector)))
    if scale < 1e-12:
        return np.zeros_like(vector)
    return vector / scale


def plot_recovery_example(output_path: Path, *, random_state: int = RANDOM_STATE) -> None:
    """Save a compact example showing all planted sources."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    dataset = make_synthetic_dataset(rng, snr=max(SNR_LEVELS))
    labels = [f"C{i + 1}" for i in range(N_CONDITIONS)]

    fig, axes = plt.subplots(
        N_SOURCES,
        4,
        figsize=(14, 3 * N_SOURCES),
        layout="constrained",
    )
    last_image = None
    for src in range(N_SOURCES):
        result = fit_redisca(dataset["X"], dataset["target_rdms"][src])
        component = best_component_by_pearson(result)
        true_topography = dataset["topographies"][:, src]
        recovered_pattern = result.A[:, component]
        if float(np.dot(true_topography, recovered_pattern)) < 0.0:
            recovered_pattern = -recovered_pattern

        last_image = plot_rdm_panel(
            axes[src, 0],
            dataset["true_rdms"][src],
            f"Source {src + 1}: true RDM",
            labels=labels,
        )
        plot_rdm_panel(
            axes[src, 1],
            dataset["target_rdms"][src],
            "Target RDM",
            labels=labels,
        )
        plot_rdm_panel(
            axes[src, 2],
            result.component_rdms[component],
            f"Recovered RDM\nr={result.pearson_scores[component]:.2f}",
            labels=labels,
        )

        axes[src, 3].plot(normalized(true_topography), label="True")
        axes[src, 3].plot(normalized(recovered_pattern), "--", label="Recovered")
        axes[src, 3].set_ylim(-1.05, 1.05)
        axes[src, 3].set_title(
            f"Pattern corr={abs_corr(recovered_pattern, true_topography):.2f}"
        )
        axes[src, 3].set_xlabel("Sensor")
        axes[src, 3].grid(alpha=0.25)
        axes[src, 3].legend(frameon=False, fontsize=8)

    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes[:, :3].ravel().tolist(),
            fraction=0.025,
            pad=0.02,
        )
    fig.suptitle("Synthetic multi-source recovery example", fontsize=14)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the benchmark using the settings above."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = run_synthetic_benchmark()
    df.to_csv(OUTPUT_ROOT / "source_recovery_trials.csv", index=False)
    plot_metric_summary(df, OUTPUT_ROOT / "source_recovery_metrics.png")
    plot_recovery_example(
        OUTPUT_ROOT / "source_recovery_example.png",
        random_state=RANDOM_STATE + 1000,
    )

    print(f"\nSaved outputs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
