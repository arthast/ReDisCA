"""Report helpers for common ReDisCA analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .export import export_result
from .types import ReDisCAResult, SlidingWindowReDisCAResult
from .viz import (
    plot_component_lambdas,
    plot_component_scores,
    plot_component_timeseries,
    plot_rdm,
    plot_top_component_rdms,
)
from .viz_mne import plot_pattern_topomaps
from .windowed import best_window_index


def save_figure(
    fig,
    output_path: str | Path,
    *,
    dpi: int = 160,
    pad_inches: float = 0.2,
) -> None:
    """Save a Matplotlib/MNE figure and close it to keep batch runs light."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(fig, list):
        if len(fig) == 1:
            fig[0].savefig(
                output_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=pad_inches,
            )
            plt.close(fig[0])
            return

        stem = output_path.stem
        suffix = output_path.suffix or ".png"
        for idx, item in enumerate(fig):
            item.savefig(
                output_path.with_name(f"{stem}_{idx}{suffix}"),
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=pad_inches,
            )
            plt.close(item)
        return

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)


def save_target_rdm_figure(
    rdm: NDArray[np.floating],
    output_path: str | Path,
    *,
    title: str,
    cmap: str = "viridis",
    condition_names: Sequence[str] | None = None,
) -> None:
    """Save a standalone target RDM figure for reports and sanity checks."""
    fig, _ = plot_rdm(
        rdm,
        title=title,
        show_values=True,
        cmap=cmap,
        condition_names=condition_names,
    )
    save_figure(fig, output_path)


def save_evoked_overview(
    evokeds: dict[str, Any],
    output_path: str | Path,
    *,
    condition_order: Sequence[str],
    picks: str | Sequence[str] | None = "eeg",
    combine: str | None = "gfp",
    title: str = "Condition evoked responses",
) -> None:
    """Save a compact overview of condition-averaged ERP/ERF traces.

    ``combine="gfp"`` plots global field power and avoids the near-zero trace
    that average-referenced EEG can produce when channels are simply averaged.
    """
    if len(condition_order) == 0:
        raise ValueError("condition_order must contain at least one condition")

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    first = evokeds[condition_order[0]]
    times_ms = 1000.0 * np.asarray(first.times, dtype=np.float64)
    for name in condition_order:
        evoked = evokeds[name]
        if picks is not None and hasattr(evoked, "copy"):
            evoked = evoked.copy().pick(picks)
        data = np.asarray(evoked.data, dtype=np.float64)

        if combine == "gfp":
            y = 1e6 * np.sqrt(np.mean(data**2, axis=0))
            ylabel = "Global field power (uV)"
        elif combine == "mean":
            y = 1e6 * np.mean(data, axis=0)
            ylabel = "Mean amplitude (uV)"
        else:
            raise ValueError("combine must be 'gfp' or 'mean'")

        ax.plot(times_ms, y, linewidth=1.6, label=name)

    if times_ms[0] <= 0.0 <= times_ms[-1]:
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    save_figure(fig, output_path)


def plot_window_metric_heatmap(
    matrix: NDArray[np.floating],
    centers_ms: NDArray[np.floating],
    *,
    title: str,
    colorbar_label: str,
    output_path: str | Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    highlight_center_ms: float | None = None,
    threshold: float | None = None,
    threshold_label: str | None = None,
    empty_label: str | None = None,
    x_label: str = "Window center (ms)",
) -> None:
    """Plot a component-by-window heatmap."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[centers_ms[0], centers_ms[-1], 0.5, matrix.shape[0] + 0.5],
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Component")
    ax.set_title(title)
    if highlight_center_ms is not None:
        ax.axvline(
            highlight_center_ms,
            color="white",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
        )
    if threshold is not None:
        finite_mask = np.isfinite(matrix)
        threshold_mask = finite_mask & (matrix < threshold)
        if np.any(threshold_mask):
            rows, cols = np.where(threshold_mask)
            ax.scatter(
                centers_ms[cols],
                rows + 1,
                marker="*",
                s=90,
                color="white",
                edgecolors="black",
                linewidths=0.4,
            )
        else:
            label = threshold_label or f"No cells below {threshold:g}"
            ax.text(
                0.99,
                0.03,
                label,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="white",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "black",
                    "alpha": 0.35,
                    "edgecolor": "none",
                },
            )
    if empty_label is not None:
        nonzero = np.isfinite(matrix) & (matrix > 0.0)
        if not np.any(nonzero):
            ax.text(
                0.99,
                0.03,
                empty_label,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="black",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "0.8",
                },
            )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=colorbar_label)
    save_figure(fig, output_path)


def save_window_metrics_csv(
    scan: SlidingWindowReDisCAResult,
    output_path: str | Path,
    *,
    max_components: int = 6,
    alpha: float = 0.05,
) -> None:
    """Save per-window component metrics in a tidy CSV table."""
    pearson = scan.component_metric_matrix("pearson_scores", max_components=max_components)
    lambdas = scan.component_metric_matrix("lambdas", max_components=max_components)
    p_values = scan.component_metric_matrix("p_values", max_components=max_components)

    rows: list[dict[str, float | int]] = []
    for window_idx in range(scan.n_windows):
        for component_idx in range(max_components):
            rows.append(
                {
                    "window": window_idx,
                    "component": component_idx,
                    "start_sample": int(scan.window_starts[window_idx]),
                    "stop_sample": int(scan.window_stops[window_idx]),
                    "center": float(scan.window_centers[window_idx]),
                    "pearson_score": float(pearson[component_idx, window_idx]),
                    "lambda": float(lambdas[component_idx, window_idx]),
                    "p_value": float(p_values[component_idx, window_idx]),
                    "significant": bool(
                        np.isfinite(p_values[component_idx, window_idx])
                        and p_values[component_idx, window_idx] < alpha
                    ),
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_sliding_window_report(
    scan: SlidingWindowReDisCAResult,
    output_dir: str | Path,
    *,
    prefix: str = "sliding_window",
    title_prefix: str = "ReDisCA sliding-window scan",
    max_components: int = 6,
    center_scale: float = 1.0,
    alpha: float = 0.05,
    highlight_window_index: int | None = None,
    include_significance: bool = False,
) -> dict[str, str]:
    """Save standard heatmaps and CSV metrics for a sliding-window scan.

    Args:
        scan: Output of ``sliding_window_fit_redisca``.
        output_dir: Directory where report files should be written.
        prefix: File-name prefix for generated artifacts.
        title_prefix: Plot title prefix.
        max_components: Number of component rows to include.
        center_scale: Multiplier for ``scan.window_centers`` in heatmaps.
            Use ``1000.0`` when window centers are stored in seconds and plots
            should be displayed in milliseconds.
        alpha: Significance threshold for the optional binary significance
            heatmap.
        highlight_window_index: Optional index to mark on all heatmaps.
        include_significance: If True, also save a binary p-value threshold
            heatmap. The p-value heatmap is usually more informative.

    Returns:
        Mapping of artifact labels to file names.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    centers = center_scale * scan.window_centers
    x_label = "Window center (ms)" if float(center_scale) == 1000.0 else "Window center"
    highlight_center = (
        None
        if highlight_window_index is None
        else float(centers[highlight_window_index])
    )

    pearson = scan.component_metric_matrix("pearson_scores", max_components=max_components)
    p_values = scan.component_metric_matrix("p_values", max_components=max_components)
    lambdas = scan.component_metric_matrix("lambdas", max_components=max_components)

    artifacts = {
        "pearson_heatmap": f"{prefix}_pearson.png",
        "pvalue_heatmap": f"{prefix}_pvalues.png",
        "lambda_heatmap": f"{prefix}_lambdas.png",
        "window_metrics": f"{prefix}_window_metrics.csv",
    }
    if include_significance:
        artifacts["significance_heatmap"] = f"{prefix}_significant.png"

    save_window_metrics_csv(
        scan,
        output_dir / artifacts["window_metrics"],
        max_components=max_components,
        alpha=alpha,
    )
    plot_window_metric_heatmap(
        pearson,
        centers,
        title=f"{title_prefix} (Pearson r)",
        colorbar_label="Pearson r",
        output_path=output_dir / artifacts["pearson_heatmap"],
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
        highlight_center_ms=highlight_center,
        x_label=x_label,
    )
    plot_window_metric_heatmap(
        p_values,
        centers,
        title=f"{title_prefix} (p-values)",
        colorbar_label="p-value",
        output_path=output_dir / artifacts["pvalue_heatmap"],
        cmap="magma_r",
        vmin=0.0,
        vmax=1.0,
        highlight_center_ms=highlight_center,
        threshold=alpha,
        threshold_label=f"No p < {alpha:g}",
        x_label=x_label,
    )
    plot_window_metric_heatmap(
        lambdas,
        centers,
        title=f"{title_prefix} (lambda)",
        colorbar_label="lambda",
        output_path=output_dir / artifacts["lambda_heatmap"],
        cmap="coolwarm",
        highlight_center_ms=highlight_center,
        x_label=x_label,
    )
    if include_significance:
        plot_window_metric_heatmap(
            (p_values < alpha).astype(float),
            centers,
            title=f"{title_prefix} significant windows (p < {alpha:g})",
            colorbar_label="significant",
            output_path=output_dir / artifacts["significance_heatmap"],
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            highlight_center_ms=highlight_center,
            empty_label=f"No p < {alpha:g}",
            x_label=x_label,
        )

    return artifacts


def save_result_diagnostics(
    result: ReDisCAResult,
    output_dir: str | Path,
    *,
    times: NDArray[np.floating],
    condition_names: Sequence[str],
    target_title: str,
    info: Any | None = None,
    top_k: int = 3,
    time_unit: str = "s",
) -> None:
    """Save standard report figures and export files for one ReDisCA fit."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_target_rdm_figure(
        result.target_rdm,
        output_dir / "target_rdm.png",
        title=target_title,
        condition_names=condition_names,
    )

    fig, _ = plot_top_component_rdms(
        result,
        k=top_k,
        order="pearson",
        include_target=True,
        condition_names=condition_names,
        save_path=output_dir / "top_component_rdms.png",
    )
    plt.close(fig)

    fig, _ = plot_component_scores(
        result,
        order="pearson",
        show_p=True,
        save_path=output_dir / "component_scores.png",
    )
    plt.close(fig)

    fig, _ = plot_component_lambdas(
        result,
        save_path=output_dir / "component_lambdas.png",
    )
    plt.close(fig)

    fig, _ = plot_component_timeseries(
        result,
        idxs=[0],
        time=times,
        time_unit=time_unit,
        condition_names=condition_names,
        legend="axes",
        condition_layout="overlay",
        highlight_interval=(
            (float(1000.0 * times[0]), float(1000.0 * times[-1]))
            if time_unit == "ms"
            else (float(times[0]), float(times[-1]))
        ),
        save_path=output_dir / "component_timeseries_by_condition.png",
    )
    plt.close(fig)

    if info is not None:
        fig, _ = plot_pattern_topomaps(
            result,
            info,
            idxs=list(range(min(top_k, result.n_components))),
            save_path=output_dir / "pattern_topomaps.png",
        )
        plt.close(fig)

    export_result(result, output_dir / "export")


def summarize_component(
    result: ReDisCAResult,
    *,
    component: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int | bool | None]:
    """Summarize one component with JSON-friendly scalar values."""
    if component >= result.n_components:
        return {
            "component": int(component),
            "lambda": None,
            "pearson_score": None,
            "p_value": None,
            "significant": False,
        }

    p_value = None if result.p_values is None else float(result.p_values[component])
    return {
        "component": int(component),
        "lambda": float(result.lambdas[component]),
        "pearson_score": float(result.pearson_scores[component]),
        "p_value": p_value,
        "significant": bool(
            p_value is not None and np.isfinite(p_value) and p_value < alpha
        ),
    }


def best_component_by_pearson(result: ReDisCAResult) -> int:
    """Return the component with the highest finite target-RDM correlation."""
    if not np.isfinite(result.pearson_scores).any():
        raise ValueError("No finite Pearson scores were found.")
    return int(np.nanargmax(result.pearson_scores))


def best_component_by_p_value(result: ReDisCAResult) -> int:
    """Return the component with the lowest p-value, falling back to Pearson."""
    if result.p_values is not None and np.isfinite(result.p_values).any():
        return int(np.nanargmin(result.p_values))
    return best_component_by_pearson(result)


def best_window_by_pearson(
    scan: SlidingWindowReDisCAResult,
    *,
    component: int = 0,
) -> int:
    """Return the window with the highest finite Pearson score."""
    pearson = scan.component_metric_matrix(
        "pearson_scores",
        max_components=component + 1,
    )[component]
    if not np.isfinite(pearson).any():
        raise ValueError("No finite Pearson scores were found.")
    return int(np.nanargmax(pearson))


def _time_suffix(time_unit: str) -> str:
    return f"_{time_unit}" if time_unit else ""


def _scaled_time(value: float, *, time_scale: float) -> float:
    return float(time_scale * value)


def summarize_window(
    scan: SlidingWindowReDisCAResult,
    window_index: int,
    *,
    times: NDArray[np.floating] | None = None,
    component: int = 0,
    alpha: float = 0.05,
    time_scale: float = 1.0,
    time_unit: str = "",
) -> dict[str, float | int | bool | None]:
    """Summarize timing and component metrics for one sliding window."""
    if times is None:
        times = scan.sample_times
    if times is not None:
        times = np.asarray(times, dtype=np.float64)

    suffix = _time_suffix(time_unit)
    result = scan.results[window_index]
    row: dict[str, float | int | bool | None] = {
        "window_index": int(window_index),
        "start_sample": int(scan.window_starts[window_index]),
        "stop_sample_exclusive": int(scan.window_stops[window_index]),
        **summarize_component(result, component=component, alpha=alpha),
    }

    if times is None:
        row["window_start_sample"] = int(scan.window_starts[window_index])
        row["window_stop_sample"] = int(scan.window_stops[window_index] - 1)
        row["window_center_sample"] = float(scan.window_centers[window_index])
    else:
        start = times[scan.window_starts[window_index]]
        stop = times[scan.window_stops[window_index] - 1]
        center = scan.window_centers[window_index]
        row[f"window_start{suffix}"] = _scaled_time(start, time_scale=time_scale)
        row[f"window_stop{suffix}"] = _scaled_time(stop, time_scale=time_scale)
        row[f"window_center{suffix}"] = _scaled_time(center, time_scale=time_scale)

    return row


def significant_window_segments(
    scan: SlidingWindowReDisCAResult,
    *,
    times: NDArray[np.floating] | None = None,
    component: int = 0,
    alpha: float = 0.05,
    time_scale: float = 1.0,
    time_unit: str = "",
) -> list[dict[str, float | int]]:
    """Find contiguous runs of windows where one component has ``p < alpha``."""
    if times is None:
        times = scan.sample_times
    if times is not None:
        times = np.asarray(times, dtype=np.float64)

    p_values = scan.component_metric_matrix(
        "p_values",
        max_components=component + 1,
    )[component]
    pearson = scan.component_metric_matrix(
        "pearson_scores",
        max_components=component + 1,
    )[component]
    significant = np.isfinite(p_values) & (p_values < alpha)
    significant_idxs = np.flatnonzero(significant)
    if significant_idxs.size == 0:
        return []

    segments: list[dict[str, float | int]] = []
    start = int(significant_idxs[0])
    previous = start
    for idx in significant_idxs[1:]:
        idx = int(idx)
        if idx == previous + 1:
            previous = idx
            continue

        segments.append(
            _summarize_window_segment(
                scan,
                start,
                previous,
                p_values,
                pearson,
                times=times,
                time_scale=time_scale,
                time_unit=time_unit,
            )
        )
        start = previous = idx

    segments.append(
        _summarize_window_segment(
            scan,
            start,
            previous,
            p_values,
            pearson,
            times=times,
            time_scale=time_scale,
            time_unit=time_unit,
        )
    )
    return segments


def _summarize_window_segment(
    scan: SlidingWindowReDisCAResult,
    start_idx: int,
    stop_idx: int,
    p_values: NDArray[np.floating],
    pearson: NDArray[np.floating],
    *,
    times: NDArray[np.floating] | None,
    time_scale: float,
    time_unit: str,
) -> dict[str, float | int]:
    suffix = _time_suffix(time_unit)
    segment_slice = slice(start_idx, stop_idx + 1)
    row: dict[str, float | int] = {
        "first_window_index": int(start_idx),
        "last_window_index": int(stop_idx),
        "n_windows": int(stop_idx - start_idx + 1),
        "min_p_value": float(np.nanmin(p_values[segment_slice])),
        "max_pearson_score": float(np.nanmax(pearson[segment_slice])),
    }

    if times is None:
        row["analysis_start_sample"] = int(scan.window_starts[start_idx])
        row["analysis_stop_sample"] = int(scan.window_stops[stop_idx] - 1)
        row["first_center_sample"] = float(scan.window_centers[start_idx])
        row["last_center_sample"] = float(scan.window_centers[stop_idx])
    else:
        row[f"analysis_start{suffix}"] = _scaled_time(
            times[scan.window_starts[start_idx]],
            time_scale=time_scale,
        )
        row[f"analysis_stop{suffix}"] = _scaled_time(
            times[scan.window_stops[stop_idx] - 1],
            time_scale=time_scale,
        )
        row[f"first_center{suffix}"] = _scaled_time(
            scan.window_centers[start_idx],
            time_scale=time_scale,
        )
        row[f"last_center{suffix}"] = _scaled_time(
            scan.window_centers[stop_idx],
            time_scale=time_scale,
        )

    return row


def summarize_sliding_window_scan(
    scan: SlidingWindowReDisCAResult,
    *,
    times: NDArray[np.floating] | None = None,
    component: int = 0,
    alpha: float = 0.05,
    n_perm: int | None = None,
    time_scale: float = 1.0,
    time_unit: str = "",
    reference_center: float | None = None,
    reference_label: str | None = None,
) -> dict[str, object]:
    """Summarize a sliding-window scan for reports."""
    best_p_idx = best_window_index(scan, component=component)
    best_r_idx = best_window_by_pearson(scan, component=component)
    segments = significant_window_segments(
        scan,
        times=times,
        component=component,
        alpha=alpha,
        time_scale=time_scale,
        time_unit=time_unit,
    )

    summary: dict[str, object] = {
        "alpha": float(alpha),
        "component": int(component),
        "best_window_index": int(best_p_idx),
        "best_by_p_value": summarize_window(
            scan,
            best_p_idx,
            times=times,
            component=component,
            alpha=alpha,
            time_scale=time_scale,
            time_unit=time_unit,
        ),
        "best_by_pearson": summarize_window(
            scan,
            best_r_idx,
            times=times,
            component=component,
            alpha=alpha,
            time_scale=time_scale,
            time_unit=time_unit,
        ),
        "significant_segments": segments,
        "has_significant_segment": bool(segments),
        "best_window_summary": summarize_result(scan.results[best_p_idx]),
    }
    if n_perm is not None:
        summary["n_perm"] = int(n_perm)

    if reference_center is not None:
        centers = time_scale * scan.window_centers
        reference_idx = int(np.argmin(np.abs(centers - reference_center)))
        label = reference_label or f"window_around_{reference_center:g}{time_unit}"
        summary[label] = summarize_window(
            scan,
            reference_idx,
            times=times,
            component=component,
            alpha=alpha,
            time_scale=time_scale,
            time_unit=time_unit,
        )

    return summary


def summarize_fixed_window_result(
    result: ReDisCAResult,
    *,
    times: NDArray[np.floating] | None = None,
    alpha: float = 0.05,
    time_scale: float = 1.0,
    time_unit: str = "",
) -> dict[str, object]:
    """Summarize one fixed-window ReDisCA fit for reports."""
    best_r_component = best_component_by_pearson(result)
    best_p_component = best_component_by_p_value(result)

    summary: dict[str, object] = {
        "best_by_pearson": summarize_component(
            result,
            component=best_r_component,
            alpha=alpha,
        ),
        "best_by_p_value": summarize_component(
            result,
            component=best_p_component,
            alpha=alpha,
        ),
        "has_significant_component": bool(
            result.p_values is not None and np.any(result.p_values < alpha)
        ),
        "summary": summarize_result(result),
    }

    if times is not None:
        times = np.asarray(times, dtype=np.float64)
        suffix = _time_suffix(time_unit)
        summary[f"window_start{suffix}"] = _scaled_time(
            times[0],
            time_scale=time_scale,
        )
        summary[f"window_stop{suffix}"] = _scaled_time(
            times[-1],
            time_scale=time_scale,
        )

    return summary


def save_window_result_diagnostics(
    scan: SlidingWindowReDisCAResult,
    window_index: int,
    output_dir: str | Path,
    *,
    times: NDArray[np.floating] | None = None,
    condition_names: Sequence[str],
    target_title: str,
    info: Any | None = None,
    top_k: int = 3,
    time_unit: str = "s",
) -> None:
    """Save standard diagnostics for one selected sliding-window result."""
    if times is None:
        times = scan.sample_times
    if times is None:
        selected_times = np.arange(
            scan.window_stops[window_index] - scan.window_starts[window_index],
            dtype=np.float64,
        )
        selected_time_unit = "sample"
    else:
        selected_times = np.asarray(times, dtype=np.float64)[
            scan.window_starts[window_index]:scan.window_stops[window_index]
        ]
        selected_time_unit = time_unit

    save_result_diagnostics(
        scan.results[window_index],
        output_dir,
        times=selected_times,
        time_unit=selected_time_unit,
        info=info,
        condition_names=condition_names,
        target_title=target_title,
        top_k=top_k,
    )


def summarize_result(result: ReDisCAResult) -> dict[str, int | list[float] | None]:
    """Convert a ReDisCA result into a JSON-friendly summary."""
    return {
        "n_components": int(result.n_components),
        "top_lambdas": [float(x) for x in result.lambdas[:5]],
        "top_pearson_scores": [float(x) for x in result.pearson_scores[:5]],
        "top_p_values": (
            None
            if result.p_values is None
            else [float(x) for x in result.p_values[:5]]
        ),
    }


__all__ = [
    "best_component_by_p_value",
    "best_component_by_pearson",
    "best_window_by_pearson",
    "plot_window_metric_heatmap",
    "save_evoked_overview",
    "save_figure",
    "save_result_diagnostics",
    "save_sliding_window_report",
    "save_target_rdm_figure",
    "save_window_result_diagnostics",
    "save_window_metrics_csv",
    "significant_window_segments",
    "summarize_component",
    "summarize_fixed_window_result",
    "summarize_result",
    "summarize_sliding_window_scan",
    "summarize_window",
]
