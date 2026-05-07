#!/usr/bin/env python3
"""Run synthetic ReDisCA simulations inspired by the article.

This file is a normal Python example: change the settings below and run it.
There are no command-line arguments.

The simulation mirrors two article-style scenarios in a lightweight,
fully reproducible sensor-space form:

1. A single representational source mixed into multichannel data.
2. Multiple simultaneously active sources, each with its own RDM.

For each Monte Carlo trial the script measures:
- how well the best ReDisCA component matches the noisy target RDM;
- how well it matches the true planted source RDM;
- how close the recovered pattern is to the planted source topography.

The source waveforms are generated from low-pass-smoothed Gaussian latents,
which follows the spirit of the article's simulation recipe without requiring
forward/inverse modeling or cortical meshes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from redisca import fit_redisca


# =============================================================================
# User settings
# =============================================================================

OUTPUT_ROOT = ROOT / "examples" / "repro_outputs" / "synthetic_article"

# Monte Carlo iterations per SNR level. Increase this for smoother summaries.
N_ITER = 30

# Larger values make the task-relevant source stronger relative to background
# sources and sensor noise.
SNR_LEVELS = [0.35, 0.70, 1.05]

# Simulated sensor-space data shape.
N_CHANNELS = 32
N_TIMEPOINTS = 200

# Smoothness of generated source waveforms. Larger values make slower signals.
SMOOTH_SIGMA = 10.0

# Noise added to the true source RDM before it is given to ReDisCA as target.
RDM_NOISE_SCALE = 0.25

RANDOM_STATE = 0


METRICS = [
    ("target_rdm_corr", "Corr(best RDM, noisy target)"),
    ("true_rdm_corr", "Corr(best RDM, true source RDM)"),
    ("pattern_cosine", "Abs cosine(best pattern, true topography)"),
]


def upper_triangle_vector(matrix: np.ndarray) -> np.ndarray:
    """Return the upper-triangular off-diagonal entries of a square matrix."""
    idx = np.triu_indices_from(matrix, k=1)
    return np.asarray(matrix[idx], dtype=float)


def normalize_rdm(rdm: np.ndarray) -> np.ndarray:
    """Rescale an RDM to have zero diagonal and max off-diagonal equal to 1."""
    rdm = np.asarray(rdm, dtype=float).copy()
    np.fill_diagonal(rdm, 0.0)
    max_value = float(np.max(upper_triangle_vector(rdm)))
    if max_value > 0:
        rdm /= max_value
    return rdm


def source_rdm_from_timeseries(source_timeseries: np.ndarray) -> np.ndarray:
    """Build a non-negative RDM from condition-specific source time courses."""
    source_timeseries = np.asarray(source_timeseries, dtype=float)
    n_conditions = source_timeseries.shape[0]
    rdm = np.zeros((n_conditions, n_conditions), dtype=float)

    for i in range(n_conditions):
        for j in range(i + 1, n_conditions):
            distance = float(np.mean((source_timeseries[i] - source_timeseries[j]) ** 2))
            rdm[i, j] = distance
            rdm[j, i] = distance

    return normalize_rdm(rdm)


def rdm_pearson(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Pearson correlation between the upper triangles of two RDMs."""
    lhs_vec = upper_triangle_vector(lhs)
    rhs_vec = upper_triangle_vector(rhs)
    lhs_std = float(np.std(lhs_vec))
    rhs_std = float(np.std(rhs_vec))
    if lhs_std < 1e-12 or rhs_std < 1e-12:
        return 0.0
    return float(np.corrcoef(lhs_vec, rhs_vec)[0, 1])


def abs_cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Absolute cosine similarity between two vectors."""
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denom < 1e-12:
        return 0.0
    return abs(float(np.dot(lhs, rhs) / denom))


def sample_unit_topographies(
    n_channels: int,
    n_sources: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample random sensor-space topographies with unit column norm."""
    topographies = rng.standard_normal((n_channels, n_sources))
    topographies /= np.linalg.norm(topographies, axis=0, keepdims=True) + 1e-12
    return topographies


def sample_condition_timeseries(
    n_conditions: int,
    n_timepoints: int,
    rng: np.random.Generator,
    *,
    smooth_sigma: float,
) -> np.ndarray:
    """Sample smooth condition-specific source waveforms."""
    mixing = rng.standard_normal((n_conditions, n_conditions))
    latents = rng.standard_normal((n_conditions, n_timepoints))
    latents = gaussian_filter1d(latents, sigma=smooth_sigma, axis=-1, mode="nearest")
    signals = mixing @ latents
    signals -= signals.mean(axis=1, keepdims=True)
    signals /= signals.std(axis=1, keepdims=True) + 1e-12
    return signals


def make_noisy_target_rdm(
    true_rdm: np.ndarray,
    rng: np.random.Generator,
    *,
    noise_scale: float,
) -> np.ndarray:
    """Perturb a source RDM while keeping it symmetric and non-negative."""
    noise = rng.standard_normal(true_rdm.shape)
    noise = 0.5 * (noise + noise.T)
    np.fill_diagonal(noise, 0.0)

    scale = noise_scale * (float(np.std(upper_triangle_vector(true_rdm))) + 1e-12)
    target_rdm = np.asarray(true_rdm, dtype=float) + scale * noise

    min_offdiag = float(np.min(upper_triangle_vector(target_rdm)))
    if min_offdiag < 0:
        target_rdm -= min_offdiag

    return normalize_rdm(target_rdm)


def sample_background_sources(
    n_conditions: int,
    n_channels: int,
    n_timepoints: int,
    n_sources: int,
    rng: np.random.Generator,
    *,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample nuisance sources that do not define the benchmark target RDM."""
    topographies = sample_unit_topographies(n_channels, n_sources, rng)
    source_timeseries = np.stack(
        [
            sample_condition_timeseries(
                n_conditions,
                n_timepoints,
                rng,
                smooth_sigma=smooth_sigma,
            )
            for _ in range(n_sources)
        ],
        axis=0,
    )
    return topographies, source_timeseries


def choose_best_component(result) -> int:
    """Choose the component with the strongest target alignment."""
    return int(np.nanargmax(result.pearson_scores))


def build_sensor_tensor(
    source_topographies: np.ndarray,
    source_timeseries: np.ndarray,
    background_topographies: np.ndarray,
    background_timeseries: np.ndarray,
    rng: np.random.Generator,
    *,
    source_scale: float,
    background_scale: float,
    sensor_noise_scale: float,
) -> np.ndarray:
    """Mix task-related and nuisance sources into a ``(C, N, T)`` tensor."""
    n_sources, n_conditions, n_timepoints = source_timeseries.shape
    n_channels = source_topographies.shape[0]
    n_background = background_timeseries.shape[0]

    X = np.zeros((n_conditions, n_channels, n_timepoints), dtype=float)

    for cond in range(n_conditions):
        for src_idx in range(n_sources):
            X[cond] += source_scale * np.outer(
                source_topographies[:, src_idx],
                source_timeseries[src_idx, cond],
            )
        for bg_idx in range(n_background):
            X[cond] += background_scale * np.outer(
                background_topographies[:, bg_idx],
                background_timeseries[bg_idx, cond],
            )
        X[cond] += sensor_noise_scale * rng.standard_normal((n_channels, n_timepoints))

    return X


def make_single_source_dataset(
    rng: np.random.Generator,
    *,
    snr: float,
    n_channels: int,
    n_timepoints: int,
    smooth_sigma: float,
    rdm_noise_scale: float,
) -> dict[str, object]:
    """Create one synthetic dataset with a single target representational source."""
    n_conditions = 5
    true_topography = sample_unit_topographies(n_channels, 1, rng)[:, 0]
    true_timeseries = sample_condition_timeseries(
        n_conditions,
        n_timepoints,
        rng,
        smooth_sigma=smooth_sigma,
    )
    true_rdm = source_rdm_from_timeseries(true_timeseries)
    target_rdm = make_noisy_target_rdm(true_rdm, rng, noise_scale=rdm_noise_scale)

    background_topographies, background_timeseries = sample_background_sources(
        n_conditions,
        n_channels,
        n_timepoints,
        n_sources=6,
        rng=rng,
        smooth_sigma=smooth_sigma,
    )
    X = build_sensor_tensor(
        source_topographies=true_topography[:, None],
        source_timeseries=true_timeseries[None, :, :],
        background_topographies=background_topographies,
        background_timeseries=background_timeseries,
        rng=rng,
        source_scale=snr,
        background_scale=0.45,
        sensor_noise_scale=0.35,
    )

    return {
        "X": X,
        "true_rdm": true_rdm,
        "target_rdm": target_rdm,
        "true_topography": true_topography,
        "n_conditions": n_conditions,
    }


def make_multi_source_dataset(
    rng: np.random.Generator,
    *,
    snr: float,
    n_channels: int,
    n_timepoints: int,
    smooth_sigma: float,
    rdm_noise_scale: float,
) -> dict[str, object]:
    """Create one synthetic dataset with several simultaneous target sources."""
    n_conditions = 6
    n_sources = 4

    true_topographies = sample_unit_topographies(n_channels, n_sources, rng)
    source_timeseries = np.stack(
        [
            sample_condition_timeseries(
                n_conditions,
                n_timepoints,
                rng,
                smooth_sigma=smooth_sigma,
            )
            for _ in range(n_sources)
        ],
        axis=0,
    )
    true_rdms = [source_rdm_from_timeseries(source_timeseries[src_idx]) for src_idx in range(n_sources)]
    target_rdms = [
        make_noisy_target_rdm(true_rdm, rng, noise_scale=rdm_noise_scale)
        for true_rdm in true_rdms
    ]

    background_topographies, background_timeseries = sample_background_sources(
        n_conditions,
        n_channels,
        n_timepoints,
        n_sources=8,
        rng=rng,
        smooth_sigma=smooth_sigma,
    )
    X = build_sensor_tensor(
        source_topographies=true_topographies,
        source_timeseries=source_timeseries,
        background_topographies=background_topographies,
        background_timeseries=background_timeseries,
        rng=rng,
        source_scale=snr,
        background_scale=0.35,
        sensor_noise_scale=0.35,
    )

    return {
        "X": X,
        "true_rdms": true_rdms,
        "target_rdms": target_rdms,
        "true_topographies": true_topographies,
        "n_conditions": n_conditions,
        "n_sources": n_sources,
    }


def evaluate_fit(
    X: np.ndarray,
    target_rdm: np.ndarray,
    true_rdm: np.ndarray,
    true_topography: np.ndarray,
) -> dict[str, float | int]:
    """Fit ReDisCA and extract article-style recovery metrics."""
    result = fit_redisca(X, target_rdm)
    best_component = choose_best_component(result)
    empirical_rdm = result.component_rdms[best_component]
    recovered_pattern = result.A[:, best_component]

    return {
        "best_component": best_component,
        "n_components": int(result.n_components),
        "best_lambda": float(result.lambdas[best_component]),
        "target_rdm_corr": float(result.pearson_scores[best_component]),
        "true_rdm_corr": rdm_pearson(empirical_rdm, true_rdm),
        "pattern_cosine": abs_cosine(recovered_pattern, true_topography),
    }


def simulate_single_source_trial(
    rng: np.random.Generator,
    *,
    snr: float,
    n_channels: int,
    n_timepoints: int,
    smooth_sigma: float,
    rdm_noise_scale: float,
) -> dict[str, float | int]:
    """Run one single-source Monte Carlo trial."""
    dataset = make_single_source_dataset(
        rng=rng,
        snr=snr,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        smooth_sigma=smooth_sigma,
        rdm_noise_scale=rdm_noise_scale,
    )

    metrics = evaluate_fit(
        dataset["X"],
        dataset["target_rdm"],
        dataset["true_rdm"],
        dataset["true_topography"],
    )
    metrics["n_conditions"] = int(dataset["n_conditions"])
    return metrics


def simulate_multi_source_trial(
    rng: np.random.Generator,
    *,
    snr: float,
    n_channels: int,
    n_timepoints: int,
    smooth_sigma: float,
    rdm_noise_scale: float,
) -> list[dict[str, float | int]]:
    """Run one multi-source Monte Carlo trial and evaluate each source RDM."""
    dataset = make_multi_source_dataset(
        rng=rng,
        snr=snr,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        smooth_sigma=smooth_sigma,
        rdm_noise_scale=rdm_noise_scale,
    )

    records: list[dict[str, float | int]] = []
    true_rdms = dataset["true_rdms"]
    target_rdms = dataset["target_rdms"]
    true_topographies = dataset["true_topographies"]

    for src_idx, (true_rdm, target_rdm) in enumerate(zip(true_rdms, target_rdms)):
        metrics = evaluate_fit(
            dataset["X"],
            target_rdm,
            true_rdm,
            true_topographies[:, src_idx],
        )
        metrics["source_index"] = src_idx
        metrics["n_conditions"] = int(dataset["n_conditions"])
        records.append(metrics)

    return records


def run_synthetic_article(
    *,
    n_iter: int = N_ITER,
    snr_levels: list[float] | tuple[float, ...] = tuple(SNR_LEVELS),
    n_channels: int = N_CHANNELS,
    n_timepoints: int = N_TIMEPOINTS,
    smooth_sigma: float = SMOOTH_SIGMA,
    rdm_noise_scale: float = RDM_NOISE_SCALE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all Monte Carlo trials for both synthetic scenarios."""
    rng = np.random.default_rng(random_state)
    single_records: list[dict[str, float | int]] = []
    multi_records: list[dict[str, float | int]] = []

    for trial_index in range(n_iter):
        for snr in snr_levels:
            single_metrics = simulate_single_source_trial(
                rng,
                snr=snr,
                n_channels=n_channels,
                n_timepoints=n_timepoints,
                smooth_sigma=smooth_sigma,
                rdm_noise_scale=rdm_noise_scale,
            )
            single_metrics["trial"] = trial_index
            single_metrics["snr"] = float(snr)
            single_records.append(single_metrics)

            for multi_metrics in simulate_multi_source_trial(
                rng,
                snr=snr,
                n_channels=n_channels,
                n_timepoints=n_timepoints,
                smooth_sigma=smooth_sigma,
                rdm_noise_scale=rdm_noise_scale,
            ):
                multi_metrics["trial"] = trial_index
                multi_metrics["snr"] = float(snr)
                multi_records.append(multi_metrics)

    return pd.DataFrame(single_records), pd.DataFrame(multi_records)


def summarize_dataframe(df: pd.DataFrame) -> list[dict[str, object]]:
    """Aggregate mean and median benchmark metrics by SNR."""
    summary: list[dict[str, object]] = []
    for snr, group in df.groupby("snr", sort=False):
        row: dict[str, object] = {
            "snr": float(snr),
            "n_rows": int(len(group)),
        }
        for metric, _ in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        summary.append(row)
    return summary


def plot_metric_grid(
    single_df: pd.DataFrame,
    multi_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot metric distributions for both simulation scenarios."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = [
        ("Single-source", single_df),
        ("Multi-source", multi_df),
    ]

    fig, axes = plt.subplots(
        len(scenarios),
        len(METRICS),
        figsize=(15, 8),
        sharex=False,
        sharey="col",
    )

    for row_idx, (scenario_name, df) in enumerate(scenarios):
        snr_levels = list(dict.fromkeys(float(value) for value in df["snr"]))
        for col_idx, (metric, label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            samples = [
                df.loc[df["snr"] == snr, metric].to_numpy(dtype=float)
                for snr in snr_levels
            ]
            ax.boxplot(samples, showfliers=False)
            ax.set_xticks(
                np.arange(1, len(snr_levels) + 1),
                [f"{snr:.2f}" for snr in snr_levels],
            )
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis="y", alpha=0.25)
            ax.set_title(label if row_idx == 0 else "")
            ax.set_xlabel("SNR")
            if col_idx == 0:
                ax.set_ylabel(f"{scenario_name}\nmetric value")

    fig.suptitle("Article-style synthetic ReDisCA benchmark", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_rdm_panel(
    ax: plt.Axes,
    rdm: np.ndarray,
    title: str,
    *,
    labels: list[str],
) -> plt.AxesImage:
    """Draw one compact RDM heatmap panel."""
    image = ax.imshow(normalize_rdm(rdm), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(labelsize=8)
    return image


def _normalized_for_display(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector for visual comparison while preserving its shape."""
    vector = np.asarray(vector, dtype=float)
    max_abs = float(np.max(np.abs(vector)))
    if max_abs < 1e-12:
        return np.zeros_like(vector)
    return vector / max_abs


def _plot_pattern_comparison_panel(
    ax: plt.Axes,
    *,
    true_topography: np.ndarray,
    recovered_topography: np.ndarray,
    pattern_similarity: float,
) -> None:
    """Draw planted and recovered sensor patterns on the same axes."""
    true_topography = np.asarray(true_topography, dtype=float)
    recovered_topography = np.asarray(recovered_topography, dtype=float)

    # ReDisCA components have arbitrary sign, so flip only for easier plotting.
    if float(np.dot(true_topography, recovered_topography)) < 0:
        recovered_topography = -recovered_topography

    ax.plot(_normalized_for_display(true_topography), label="True")
    ax.plot(
        _normalized_for_display(recovered_topography),
        label="Recovered",
        linestyle="--",
    )
    ax.set_title(f"Pattern match = {pattern_similarity:.2f}")
    ax.set_xlabel("Synthetic channel")
    ax.set_ylabel("Weight")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def plot_single_source_example(
    output_path: Path,
    *,
    snr: float,
    n_channels: int,
    n_timepoints: int,
    smooth_sigma: float,
    rdm_noise_scale: float,
    random_state: int,
) -> None:
    """Save an explanatory plot for one single-source synthetic run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    dataset = make_single_source_dataset(
        rng,
        snr=snr,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        smooth_sigma=smooth_sigma,
        rdm_noise_scale=rdm_noise_scale,
    )

    result = fit_redisca(dataset["X"], dataset["target_rdm"])
    best_component = choose_best_component(result)
    recovered_rdm = result.component_rdms[best_component]

    true_topography = np.asarray(dataset["true_topography"], dtype=float)
    recovered_topography = np.asarray(result.A[:, best_component], dtype=float)
    if float(np.dot(true_topography, recovered_topography)) < 0:
        recovered_topography *= -1.0

    labels = [f"C{i + 1}" for i in range(int(dataset["n_conditions"]))]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")

    _plot_rdm_panel(axes[0, 0], dataset["true_rdm"], "True source RDM", labels=labels)
    _plot_rdm_panel(axes[0, 1], dataset["target_rdm"], "Noisy target RDM", labels=labels)
    image = _plot_rdm_panel(
        axes[1, 0],
        recovered_rdm,
        f"Recovered component RDM #{best_component}",
        labels=labels,
    )

    axes[1, 1].plot(_normalized_for_display(true_topography), label="True pattern")
    axes[1, 1].plot(
        _normalized_for_display(recovered_topography),
        label="Recovered pattern",
        linestyle="--",
    )
    axes[1, 1].set_title("True vs recovered sensor pattern")
    axes[1, 1].set_xlabel("Synthetic channel")
    axes[1, 1].set_ylabel("Normalized weight")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False)

    fig.colorbar(
        image,
        ax=[axes[0, 0], axes[0, 1], axes[1, 0]],
        fraction=0.04,
        pad=0.04,
    )
    fig.suptitle("Single-source synthetic example", fontsize=14)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_multi_source_example(
    output_path: Path,
    *,
    snr: float,
    n_channels: int,
    n_timepoints: int,
    smooth_sigma: float,
    rdm_noise_scale: float,
    random_state: int,
) -> None:
    """Save RDM panels for one multi-source synthetic run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    dataset = make_multi_source_dataset(
        rng,
        snr=snr,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        smooth_sigma=smooth_sigma,
        rdm_noise_scale=rdm_noise_scale,
    )

    n_sources = int(dataset["n_sources"])
    labels = [f"C{i + 1}" for i in range(int(dataset["n_conditions"]))]
    fig, axes = plt.subplots(
        n_sources,
        4,
        figsize=(15, 3.1 * n_sources),
        layout="constrained",
    )

    last_image = None
    for src_idx in range(n_sources):
        true_rdm = dataset["true_rdms"][src_idx]
        target_rdm = dataset["target_rdms"][src_idx]
        result = fit_redisca(dataset["X"], target_rdm)
        best_component = choose_best_component(result)
        recovered_rdm = result.component_rdms[best_component]
        recovered_topography = result.A[:, best_component]
        true_topography = dataset["true_topographies"][:, src_idx]

        rdm_similarity = rdm_pearson(recovered_rdm, true_rdm)
        pattern_similarity = abs_cosine(recovered_topography, true_topography)

        last_image = _plot_rdm_panel(
            axes[src_idx, 0],
            true_rdm,
            f"Source {src_idx + 1}: true RDM",
            labels=labels,
        )
        _plot_rdm_panel(
            axes[src_idx, 1],
            target_rdm,
            "Target RDM",
            labels=labels,
        )
        _plot_rdm_panel(
            axes[src_idx, 2],
            recovered_rdm,
            f"Recovered RDM, match = {rdm_similarity:.2f}",
            labels=labels,
        )
        _plot_pattern_comparison_panel(
            axes[src_idx, 3],
            true_topography=true_topography,
            recovered_topography=recovered_topography,
            pattern_similarity=pattern_similarity,
        )

    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes[:, :3].ravel().tolist(),
            fraction=0.025,
            pad=0.02,
        )
    fig.suptitle("Multi-source synthetic example", fontsize=14)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the example using the settings at the top of this file."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    single_df, multi_df = run_synthetic_article(
        n_iter=N_ITER,
        snr_levels=SNR_LEVELS,
        n_channels=N_CHANNELS,
        n_timepoints=N_TIMEPOINTS,
        smooth_sigma=SMOOTH_SIGMA,
        rdm_noise_scale=RDM_NOISE_SCALE,
        random_state=RANDOM_STATE,
    )
    single_path = OUTPUT_ROOT / "single_source_trials.csv"
    multi_path = OUTPUT_ROOT / "multi_source_trials.csv"
    single_df.to_csv(single_path, index=False)
    multi_df.to_csv(multi_path, index=False)

    plot_metric_grid(single_df, multi_df, OUTPUT_ROOT / "benchmark_overview.png")
    example_snr = float(max(SNR_LEVELS))
    plot_single_source_example(
        OUTPUT_ROOT / "single_source_example.png",
        snr=example_snr,
        n_channels=N_CHANNELS,
        n_timepoints=N_TIMEPOINTS,
        smooth_sigma=SMOOTH_SIGMA,
        rdm_noise_scale=RDM_NOISE_SCALE,
        random_state=RANDOM_STATE + 1000,
    )
    plot_multi_source_example(
        OUTPUT_ROOT / "multi_source_example.png",
        snr=example_snr,
        n_channels=N_CHANNELS,
        n_timepoints=N_TIMEPOINTS,
        smooth_sigma=SMOOTH_SIGMA,
        rdm_noise_scale=RDM_NOISE_SCALE,
        random_state=RANDOM_STATE + 2000,
    )

    summary = {
        "config": {
            "n_iter": int(N_ITER),
            "snr_levels": [float(value) for value in SNR_LEVELS],
            "n_channels": int(N_CHANNELS),
            "n_timepoints": int(N_TIMEPOINTS),
            "smooth_sigma": float(SMOOTH_SIGMA),
            "rdm_noise_scale": float(RDM_NOISE_SCALE),
            "random_state": int(RANDOM_STATE),
        },
        "single_source": summarize_dataframe(single_df),
        "multi_source": summarize_dataframe(multi_df),
        "artifacts": {
            "single_source_trials": single_path.name,
            "multi_source_trials": multi_path.name,
            "overview_plot": "benchmark_overview.png",
            "single_source_plot": "single_source_example.png",
            "multi_source_plot": "multi_source_example.png",
        },
    }
    with (OUTPUT_ROOT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
