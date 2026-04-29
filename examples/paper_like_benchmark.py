#!/usr/bin/env python3
"""Run paper-like ReDisCA simulations inspired by the article.

This benchmark mirrors two scenarios from the paper in a lightweight,
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

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/redisca-mpl-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from redisca import fit_redisca


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

    metrics = evaluate_fit(X, target_rdm, true_rdm, true_topography)
    metrics["n_conditions"] = n_conditions
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

    records: list[dict[str, float | int]] = []
    for src_idx, (true_rdm, target_rdm) in enumerate(zip(true_rdms, target_rdms)):
        metrics = evaluate_fit(X, target_rdm, true_rdm, true_topographies[:, src_idx])
        metrics["source_index"] = src_idx
        metrics["n_conditions"] = n_conditions
        records.append(metrics)

    return records


def run_benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all Monte Carlo trials for both scenarios."""
    rng = np.random.default_rng(args.random_state)
    single_records: list[dict[str, float | int]] = []
    multi_records: list[dict[str, float | int]] = []

    for trial_index in range(args.n_iter):
        for snr in args.snr_levels:
            single_metrics = simulate_single_source_trial(
                rng,
                snr=snr,
                n_channels=args.n_channels,
                n_timepoints=args.n_timepoints,
                smooth_sigma=args.smooth_sigma,
                rdm_noise_scale=args.rdm_noise_scale,
            )
            single_metrics["trial"] = trial_index
            single_metrics["snr"] = float(snr)
            single_records.append(single_metrics)

            for multi_metrics in simulate_multi_source_trial(
                rng,
                snr=snr,
                n_channels=args.n_channels,
                n_timepoints=args.n_timepoints,
                smooth_sigma=args.smooth_sigma,
                rdm_noise_scale=args.rdm_noise_scale,
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

    fig.suptitle("Paper-like ReDisCA benchmark", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "examples" / "repro_outputs" / "paper_like_benchmark",
        help="Directory where benchmark tables, figures, and JSON summaries are saved.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Monte Carlo iterations per SNR level.",
    )
    parser.add_argument(
        "--snr-levels",
        nargs="+",
        type=float,
        default=[0.35, 0.70, 1.05],
        help="Signal amplitudes for the task-relevant sources.",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=32,
        help="Number of simulated sensors.",
    )
    parser.add_argument(
        "--n-timepoints",
        type=int,
        default=200,
        help="Number of samples in each condition waveform.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=10.0,
        help="Gaussian smoothing sigma for source waveforms.",
    )
    parser.add_argument(
        "--rdm-noise-scale",
        type=float,
        default=0.25,
        help="Noise level used when perturbing the planted source RDMs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    single_df, multi_df = run_benchmark(args)
    single_path = args.output_root / "single_source_trials.csv"
    multi_path = args.output_root / "multi_source_trials.csv"
    single_df.to_csv(single_path, index=False)
    multi_df.to_csv(multi_path, index=False)

    plot_metric_grid(single_df, multi_df, args.output_root / "benchmark_overview.png")

    summary = {
        "config": {
            "n_iter": int(args.n_iter),
            "snr_levels": [float(value) for value in args.snr_levels],
            "n_channels": int(args.n_channels),
            "n_timepoints": int(args.n_timepoints),
            "smooth_sigma": float(args.smooth_sigma),
            "rdm_noise_scale": float(args.rdm_noise_scale),
            "random_state": int(args.random_state),
        },
        "single_source": summarize_dataframe(single_df),
        "multi_source": summarize_dataframe(multi_df),
        "artifacts": {
            "single_source_trials": single_path.name,
            "multi_source_trials": multi_path.name,
            "overview_plot": "benchmark_overview.png",
        },
    }
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs to {args.output_root}")


if __name__ == "__main__":
    main()
