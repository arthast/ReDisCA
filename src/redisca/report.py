"""Report helpers for common ReDisCA analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .core import compute_component_timeseries
from .export import export_result
from .types import ReDisCAResult, SlidingWindowReDisCAResult
from .viz import (
    plot_component_timeseries_panel,
    plot_component_lambdas,
    plot_component_scores,
    plot_component_timeseries,
    plot_rdm_panel,
    plot_rdm,
    plot_top_component_rdms,
)
from .viz_mne import plot_pattern_topomap_panel, plot_pattern_topomaps
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


def _short_condition_names(condition_names: Sequence[str]) -> list[str]:
    replacements = {
        "scrambled_face": "scr. face",
        "scrambled_car": "scr. car",
    }
    return [replacements.get(str(name), str(name)) for name in condition_names]


def _component_timeseries_for_plot(
    result: ReDisCAResult,
    X_full: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    if X_full is None:
        return result.component_timeseries

    X_full = np.asarray(X_full, dtype=np.float64)
    if X_full.shape[:2] != (result.n_conditions, result.n_channels):
        raise ValueError(
            "X_full must have shape (conditions, channels, timepoints) aligned "
            f"with the fitted result, got {X_full.shape}."
        )
    return compute_component_timeseries(result.W, X_full)


def _component_stats_title(
    result: ReDisCAResult,
    component: int,
    *,
    alpha: float,
) -> str:
    lines = [
        f"comp {component}",
        f"lambda={result.lambdas[component]:.3f}, r={result.pearson_scores[component]:.3f}",
    ]
    if result.p_values is not None:
        p_value = result.p_values[component]
        marker = "*" if np.isfinite(p_value) and p_value < alpha else "n.s."
        lines.append(f"p={p_value:.3g} {marker}")
    return "\n".join(lines)


def save_scan_overview_figure(
    scan: SlidingWindowReDisCAResult,
    output_path: str | Path,
    *,
    info: Any,
    condition_names: Sequence[str],
    title: str,
    component: int = 0,
    max_components: int = 6,
    alpha: float = 0.05,
    selected_window_index: int | None = None,
    center_scale: float = 1000.0,
    time_unit: str = "ms",
) -> None:
    """Save target RDM, p-value map, selected topomap, and p-value trace."""
    if selected_window_index is None:
        selected_window_index = best_window_index(scan, component=component)

    display_names = _short_condition_names(condition_names)
    centers = center_scale * scan.window_centers
    selected_center = float(centers[selected_window_index])
    p_values = scan.component_metric_matrix("p_values", max_components=max_components)

    fig = plt.figure(figsize=(12.4, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.8],
        height_ratios=[1.0, 1.0],
        wspace=0.14,
    )
    axes = np.array(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ],
        dtype=object,
    )
    fig.suptitle(title, fontsize=14)

    plot_rdm_panel(
        axes[0, 0],
        scan.results[0].target_rdm,
        title="Target RDM",
        condition_names=display_names,
        colorbar_label="Target dissimilarity",
    )

    heatmap = axes[0, 1].imshow(
        p_values,
        aspect="auto",
        origin="lower",
        cmap="magma_r",
        vmin=0.0,
        vmax=1.0,
        extent=[centers[0], centers[-1], -0.5, max_components - 0.5],
    )
    axes[0, 1].axvline(
        selected_center,
        color="white",
        linestyle="--",
        linewidth=1.0,
        alpha=0.95,
    )
    sig_rows, sig_cols = np.where(np.isfinite(p_values) & (p_values < alpha))
    if sig_rows.size:
        axes[0, 1].scatter(
            centers[sig_cols],
            sig_rows,
            marker="*",
            s=80,
            color="white",
            edgecolors="black",
            linewidths=0.35,
        )
    axes[0, 1].set_title("Component p-values over time")
    axes[0, 1].set_xlabel(f"Window center ({time_unit})")
    axes[0, 1].set_ylabel("Component")
    axes[0, 1].set_yticks(range(max_components))
    fig.colorbar(
        heatmap,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.04,
        label="Permutation p-value",
    )

    selected_result = scan.results[selected_window_index]
    plot_pattern_topomap_panel(
        axes[1, 0],
        selected_result,
        info,
        component=component,
        title=f"Pattern, center={selected_center:.1f} {time_unit}",
    )

    component_p = p_values[component]
    axes[1, 1].plot(centers, component_p, color="#4C72B0", linewidth=1.8)
    axes[1, 1].axhline(alpha, color="crimson", linestyle="--", linewidth=1.0)
    axes[1, 1].axvline(selected_center, color="0.25", linestyle=":", linewidth=1.0)
    finite_sig = np.isfinite(component_p) & (component_p < alpha)
    if np.any(finite_sig):
        y_star = np.full(np.count_nonzero(finite_sig), alpha * 0.72)
        axes[1, 1].scatter(
            centers[finite_sig],
            y_star,
            marker="*",
            s=90,
            color="crimson",
            zorder=3,
        )
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title(f"Component {component} p-value profile")
    axes[1, 1].set_xlabel(f"Window center ({time_unit})")
    axes[1, 1].set_ylabel("p-value", labelpad=0)
    axes[1, 1].grid(True, linewidth=0.3, alpha=0.35)

    save_figure(fig, output_path, dpi=170)


def save_window_sequence_figure(
    scan: SlidingWindowReDisCAResult,
    output_path: str | Path,
    *,
    info: Any,
    times: NDArray[np.floating],
    condition_names: Sequence[str],
    title: str,
    window_indices: Sequence[int],
    component: int = 0,
    X_full: NDArray[np.floating] | None = None,
    alpha: float = 0.05,
    time_scale: float = 1000.0,
    time_unit: str = "ms",
) -> None:
    """Save selected windows as topomap, timecourse, and RDM panel rows."""
    if len(window_indices) == 0:
        raise ValueError("window_indices must contain at least one window")

    display_names = _short_condition_names(condition_names)
    times = np.asarray(times, dtype=np.float64)
    rows = len(window_indices)
    fig = plt.figure(
        figsize=(12.5, max(2.7 * rows + 0.8, 4.6)),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        rows,
        3,
        width_ratios=[1.0, 2.45, 1.0],
        wspace=0.12,
        hspace=0.24,
    )
    fig.suptitle(title, fontsize=14)

    for row, window_index in enumerate(window_indices):
        result = scan.results[int(window_index)]
        window_start = int(scan.window_starts[int(window_index)])
        window_stop = int(scan.window_stops[int(window_index)])
        center = float(time_scale * scan.window_centers[int(window_index)])
        start = float(time_scale * times[window_start])
        stop = float(time_scale * times[window_stop - 1])
        stats = _component_stats_title(result, component, alpha=alpha)

        topo_ax = fig.add_subplot(grid[row, 0])
        plot_pattern_topomap_panel(
            topo_ax,
            result,
            info,
            component=component,
            title=f"Window {window_index}, {center:.1f} {time_unit}\n{stats}",
        )

        if X_full is None:
            component_ts = result.component_timeseries
            ts_times = times[window_start:window_stop]
            highlight = None
        else:
            component_ts = _component_timeseries_for_plot(result, X_full)
            ts_times = times
            highlight = (start, stop)

        ts_ax = fig.add_subplot(grid[row, 1])
        plot_component_timeseries_panel(
            ts_ax,
            component_ts,
            ts_times,
            component=component,
            condition_names=display_names,
            title="Spatially filtered condition time series",
            time_scale=time_scale,
            time_unit=time_unit,
            highlight_interval=highlight,
            show_legend=row == 0,
        )

        rdm_ax = fig.add_subplot(grid[row, 2])
        plot_rdm_panel(
            rdm_ax,
            result.component_rdms[component],
            title=f"Observed RDM\nr={result.pearson_scores[component]:.3f}",
            condition_names=display_names,
            colorbar_label="Observed dissimilarity",
        )

    save_figure(fig, output_path, dpi=170)


def save_component_summary_figure(
    result: ReDisCAResult,
    output_path: str | Path,
    *,
    info: Any,
    times: NDArray[np.floating],
    condition_names: Sequence[str],
    title: str,
    component_indices: Sequence[int],
    X_full: NDArray[np.floating] | None = None,
    alpha: float = 0.05,
    time_scale: float = 1000.0,
    time_unit: str = "ms",
    highlight_interval: tuple[float, float] | None = None,
    include_target: bool = True,
) -> None:
    """Save a fixed-window component summary figure."""
    component_indices = [
        int(idx) for idx in component_indices if idx < result.n_components
    ]
    if len(component_indices) == 0:
        raise ValueError("component_indices must contain at least one valid component")

    display_names = _short_condition_names(condition_names)
    times = np.asarray(times, dtype=np.float64)
    timeseries = _component_timeseries_for_plot(result, X_full)
    if times.shape != (timeseries.shape[-1],):
        raise ValueError(
            f"times must have shape ({timeseries.shape[-1]},), got {times.shape}"
        )

    rows = len(component_indices)
    cols = 4 if include_target else 3
    fig = plt.figure(
        figsize=(13.8 if include_target else 11.2, max(2.9 * rows + 1.0, 4.4)),
        constrained_layout=True,
    )
    width_ratios = [1.0, 1.0, 2.35, 1.0] if include_target else [1.0, 2.35, 1.0]
    grid = fig.add_gridspec(
        rows,
        cols,
        width_ratios=width_ratios,
        wspace=0.16,
        hspace=0.26,
    )
    fig.suptitle(title, fontsize=14)

    if include_target:
        target_ax = fig.add_subplot(grid[0, 0])
        plot_rdm_panel(
            target_ax,
            result.target_rdm,
            title="Target RDM",
            condition_names=display_names,
            colorbar_label="Target dissimilarity",
        )
        for row in range(1, rows):
            blank_ax = fig.add_subplot(grid[row, 0])
            blank_ax.axis("off")

    for row, component in enumerate(component_indices):
        grid_col = 1 if include_target else 0
        stats = _component_stats_title(result, component, alpha=alpha)

        topo_ax = fig.add_subplot(grid[row, grid_col])
        plot_pattern_topomap_panel(
            topo_ax,
            result,
            info,
            component=component,
            title=f"Pattern\n{stats}",
        )
        grid_col += 1

        ts_ax = fig.add_subplot(grid[row, grid_col])
        plot_component_timeseries_panel(
            ts_ax,
            timeseries,
            times,
            component=component,
            condition_names=display_names,
            title="Spatially filtered condition time series",
            time_scale=time_scale,
            time_unit=time_unit,
            highlight_interval=highlight_interval,
            show_legend=row == 0,
        )
        grid_col += 1

        rdm_ax = fig.add_subplot(grid[row, grid_col])
        plot_rdm_panel(
            rdm_ax,
            result.component_rdms[component],
            title=f"Observed RDM\nr={result.pearson_scores[component]:.3f}",
            condition_names=display_names,
            colorbar_label="Observed dissimilarity",
        )

    save_figure(fig, output_path, dpi=170)


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
    mark_threshold: bool = False,
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
    if threshold is not None and mark_threshold:
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
        mark_threshold=False,
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


__all__ = [
    "plot_window_metric_heatmap",
    "save_component_summary_figure",
    "save_evoked_overview",
    "save_figure",
    "save_result_diagnostics",
    "save_scan_overview_figure",
    "save_sliding_window_report",
    "save_target_rdm_figure",
    "save_window_sequence_figure",
    "save_window_result_diagnostics",
    "save_window_metrics_csv",
]
