"""Visualization utilities for ReDisCA results.

All functions accept a ``ReDisCAResult`` (or raw arrays) and return
``(fig, ax)`` or ``(fig, axes)``.  Every plotting function accepts an
optional ``save_path`` — when given, the figure is saved to that file
(parent directories are created automatically).
Only ``matplotlib`` and ``numpy`` are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .types import ReDisCAResult, SlidingWindowReDisCAResult


def _maybe_save(fig: Figure | SubFigure, save_path: str | Path | None, dpi: int) -> None:
    """Save *fig* to *save_path* if it is not ``None``."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {path}")


def _minmax(M: NDArray[np.floating]) -> NDArray[np.floating]:
    """Scale *M* into [0, 1] by min–max normalization."""
    mn, mx = M.min(), M.max()
    return (M - mn) / (mx - mn + 1e-12)


def add_panel_colorbar(
    ax: Axes,
    image,
    *,
    label: str,
    size: str = "4%",
    pad: float = 0.06,
    ticks: Sequence[float] | None = None,
) -> None:
    """Attach a compact colorbar to one panel in a composed figure."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    colorbar = ax.figure.colorbar(image, cax=cax, ticks=ticks)
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.set_label(label, fontsize=8, labelpad=2)


def _contrast_matrix(matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    """Min-max normalize a matrix for annotation contrast decisions."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return (matrix - np.min(matrix)) / (np.ptp(matrix) + 1e-12)


def plot_rdm_panel(
    ax: Axes,
    matrix: NDArray[np.floating],
    *,
    title: str,
    condition_names: Sequence[str],
    cmap: str = "viridis",
    show_values: bool = True,
    colorbar: bool = True,
    colorbar_label: str = "Dissimilarity",
):
    """Plot one compact RDM panel for multi-panel figures."""
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix_display = _contrast_matrix(matrix)
    vmin = float(np.min(matrix))
    vmax = float(np.max(matrix))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    image = ax.imshow(matrix, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)

    C = matrix_display.shape[0]
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(condition_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(condition_names, fontsize=8)
    ax.set_title(title)
    ax.tick_params(length=0)

    if show_values:
        for row in range(C):
            for col in range(C):
                value = float(matrix[row, col])
                color = "white" if matrix_display[row, col] < 0.5 else "black"
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )

    if colorbar:
        add_panel_colorbar(ax, image, label=colorbar_label, size="4%", pad=0.05)

    return image


def plot_component_timeseries_panel(
    ax: Axes,
    timeseries: NDArray[np.floating],
    times: NDArray[np.floating],
    *,
    component: int,
    condition_names: Sequence[str],
    title: str,
    time_scale: float,
    time_unit: str,
    highlight_interval: tuple[float, float] | None = None,
    show_legend: bool = False,
) -> None:
    """Plot one component's condition time series into an existing axes."""
    times = np.asarray(times, dtype=np.float64)
    if times.shape != (timeseries.shape[-1],):
        raise ValueError(
            f"times must have shape ({timeseries.shape[-1]},), got {times.shape}"
        )

    x = time_scale * times
    for condition_idx, condition_name in enumerate(condition_names):
        ax.plot(
            x,
            timeseries[condition_idx, component],
            linewidth=1.5,
            label=condition_name,
        )

    if x[0] <= 0.0 <= x[-1]:
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    if highlight_interval is not None:
        ax.axvspan(
            highlight_interval[0],
            highlight_interval[1],
            color="grey",
            alpha=0.14,
            linewidth=0,
        )
    ax.axhline(0.0, color="0.5", linewidth=0.6, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(f"Time ({time_unit})" if time_unit else "Time")
    ax.set_ylabel("Component amplitude", labelpad=2)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=8, loc="upper right", frameon=True)


def plot_rdm(
    matrix: NDArray[np.floating],
    ax: Axes | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_values: bool = False,
    cmap: str = "viridis",
    condition_names: Sequence[str] | None = None,
    colorbar: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> tuple[Figure | SubFigure, Axes | None]:
    """Plot a single RDM as a heatmap.

    Args:
        matrix: Square matrix (C, C).
        ax: Matplotlib axes to draw on.  Created if *None*.
        title: Axes title.
        vmin, vmax: Colour-scale limits.
        show_values: If *True*, annotate each cell with its numeric value.
        cmap: Matplotlib colormap name.
        condition_names: Optional labels for the RDM rows/columns.
        colorbar: If *True*, add a colorbar to the axes.
        save_path: If given, save the figure to this file path.
        dpi: Resolution for saved figure (default 150).

    Returns:
        ``(fig, ax)`` tuple.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"matrix must be square 2-D array, got shape {matrix.shape}"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.5))
    else:
        fig = ax.figure

    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    C = matrix.shape[0]
    if condition_names is None:
        condition_names = [str(i) for i in range(C)]
    elif len(condition_names) != C:
        raise ValueError(
            f"len(condition_names)={len(condition_names)} != matrix size {C}"
        )

    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(condition_names, rotation=35, ha="right")
    ax.set_yticklabels(condition_names)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Condition")

    if title is not None:
        ax.set_title(title)

    if show_values:
        for i in range(C):
            for j in range(C):
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if matrix[i, j] < (matrix.max() + matrix.min()) / 2 else "black",
                    fontsize=7,
                )

    _maybe_save(fig, save_path, dpi)
    return fig, ax


def plot_top_component_rdms(
    result: ReDisCAResult,
    k: int = 3,
    order: str = "pearson",
    pearson_mode: str = "pos",
    include_target: bool = True,
    normalize_rdms: bool = True,
    shared_colorbar: bool = True,
    show_values: bool = False,
    cmap: str = "viridis",
    condition_names: Sequence[str] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> tuple[Figure, np.ndarray]:
    """Plot the *k* the best component RDMs, optionally preceded by the target.

    Args:
        result: Output of ``fit_redisca``.
        k: Number of components to show.
        order: ``"pearson"`` or ``"lambda"`` — how to rank components.
        pearson_mode: Used when ``order="pearson"``.
            ``"pos"`` (default) — rank by highest positive Pearson r
            (best direct match).
            ``"abs"`` — rank by largest absolute Pearson r (includes
            anti-correlations).
        include_target: If *True*, the first panel shows ``target_rdm``.
        normalize_rdms: If *True* (default), min–max normalise each
            component RDM to [0, 1] so that all panels share the same
            color scale and the *shape* of the RDM is compared rather
            than the absolute magnitude.
        shared_colorbar: If *True* (default), draw one shared colorbar
            for the whole figure. Disable to give each panel its own.
        show_values: Annotate cells with numeric values.
        cmap: Matplotlib colormap name.
        condition_names: Optional labels for RDM rows/columns.
        save_path: If given, save the figure to this file path.
        dpi: Resolution for saved figure (default 150).

    Returns:
        ``(fig, axes)`` where *axes* is a 1-D numpy array of ``Axes``.
    """
    if result.component_rdms is None:
        raise ValueError("ReDisCAResult.component_rdms is None — nothing to plot.")

    r = result.n_components
    k = min(k, r)

    if order == "pearson":
        if pearson_mode == "pos":
            ranked = np.argsort(-result.pearson_scores)
        elif pearson_mode == "abs":
            ranked = np.argsort(-np.abs(result.pearson_scores))
        else:
            raise ValueError(
                f"pearson_mode must be 'pos' or 'abs', got '{pearson_mode}'"
            )
    elif order == "lambda":
        ranked = np.argsort(-result.lambdas)
    else:
        raise ValueError(f"order must be 'pearson' or 'lambda', got '{order}'")

    selected = ranked[:k]

    # prepare matrices for display (optionally normalized)
    display_mats: list[NDArray[np.floating]] = []
    if include_target:
        target_disp = _minmax(result.target_rdm) if normalize_rdms else result.target_rdm.copy()
        display_mats.append(target_disp)

    comp_display: list[NDArray[np.floating]] = []
    for idx in selected:
        m = _minmax(result.component_rdms[idx]) if normalize_rdms else result.component_rdms[idx].copy()
        comp_display.append(m)
        display_mats.append(m)

    # shared color scale across all panels
    shared_vmin = min(float(np.min(M)) for M in display_mats)
    shared_vmax = max(float(np.max(M)) for M in display_mats)

    n_panels = k + int(include_target)
    fig_width = 3.7 * n_panels + (0.5 if shared_colorbar else 0.0)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(fig_width, 3.7),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = np.array([axes])

    col = 0
    images: list[object] = []

    # Target RDM
    if include_target:
        plot_rdm(
            display_mats[0], ax=axes[col],
            title="Target RDM", show_values=show_values, cmap=cmap,
            vmin=shared_vmin, vmax=shared_vmax,
            condition_names=condition_names,
            colorbar=not shared_colorbar,
        )
        images.append(axes[col].images[-1])
        col += 1

    # Component RDMs
    for ci, idx in enumerate(selected):
        title_parts = [f"Comp {idx}"]
        title_parts.append(f"λ={result.lambdas[idx]:.3f}")
        title_parts.append(f"r={result.pearson_scores[idx]:.3f}")
        if result.p_values is not None:
            title_parts.append(f"p={result.p_values[idx]:.3f}")
        title = "\n".join(title_parts)

        plot_rdm(
            comp_display[ci], ax=axes[col],
            title=title, show_values=show_values, cmap=cmap,
            vmin=shared_vmin, vmax=shared_vmax,
            condition_names=condition_names,
            colorbar=not shared_colorbar,
        )
        images.append(axes[col].images[-1])
        col += 1

    if shared_colorbar and images:
        label = "Normalized dissimilarity" if normalize_rdms else "Dissimilarity"
        cbar = fig.colorbar(
            images[-1],
            ax=axes,
            fraction=0.03,
            pad=0.04,
            shrink=0.86,
        )
        cbar.set_label(label)

    _maybe_save(fig, save_path, dpi)
    return fig, axes


def plot_component_scores(
    result: ReDisCAResult,
    order: str = "lambda",
    pearson_mode: str = "pos",
    show_p: bool = False,
    ax: Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> tuple[Figure | SubFigure, Axes | None]:
    """Bar plot of Pearson correlation scores per component.

    Args:
        result: Output of ``fit_redisca``.
        order: ``"lambda"`` (default, natural order) or ``"pearson"``.
        pearson_mode: Used when ``order="pearson"``.
            ``"pos"`` — rank by highest positive r.
            ``"abs"`` — rank by largest |r|.
        show_p: If *True* and ``p_values`` exist, mark significant
            components with a star (★).
        ax: Matplotlib axes to draw on.  Created if *None*.
        save_path: If given, save the figure to this file path.
        dpi: Resolution for saved figure (default 150).

    Returns:
        ``(fig, ax)`` tuple.
    """
    r = result.n_components
    scores = result.pearson_scores.copy()

    if order == "pearson":
        if pearson_mode == "pos":
            idx = np.argsort(-scores)
        elif pearson_mode == "abs":
            idx = np.argsort(-np.abs(scores))
        else:
            raise ValueError(
                f"pearson_mode must be 'pos' or 'abs', got '{pearson_mode}'"
            )
    elif order == "lambda":
        idx = np.arange(r)
    else:
        raise ValueError(f"order must be 'lambda' or 'pearson', got '{order}'")

    scores = scores[idx]
    labels = [str(i) for i in idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 0.6 * r), 4))
    else:
        fig = ax.figure

    colours = ["#4C72B0" if s >= 0 else "#DD8452" for s in scores]
    bars = ax.bar(range(r), scores, color=colours)

    # Significance markers
    if show_p and result.p_values is not None:
        p_sorted = result.p_values[idx]
        sig_sorted = result.significant[idx] if result.significant is not None else p_sorted < 0.05
        for i, (bar, is_sig) in enumerate(zip(bars, sig_sorted)):
            if is_sig:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    "★",
                    ha="center", va="bottom",
                    fontsize=12, color="goldenrod",
                )

    ax.set_xticks(range(r))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Component")
    ax.set_ylabel("Pearson r")
    ax.set_title("Component–Target Pearson Scores")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

    _maybe_save(fig, save_path, dpi)
    return fig, ax


def plot_component_lambdas(
    result: ReDisCAResult,
    ax: Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> tuple[Figure | SubFigure, Axes | None]:
    """Bar plot of eigenvalues (λ) per component.

    Args:
        result: Output of ``fit_redisca``.
        ax: Matplotlib axes to draw on.  Created if *None*.
        save_path: If given, save the figure to this file path.
        dpi: Resolution for saved figure (default 150).

    Returns:
        ``(fig, ax)`` tuple.
    """
    r = result.n_components
    lambdas = result.lambdas

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 0.6 * r), 4))
    else:
        fig = ax.figure

    colours = ["#4C72B0" if v >= 0 else "#DD8452" for v in lambdas]
    ax.bar(range(r), lambdas, color=colours)
    ax.set_xticks(range(r))
    ax.set_xlabel("Component")
    ax.set_ylabel("λ")
    ax.set_title("Component Eigenvalues (λ)")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

    _maybe_save(fig, save_path, dpi)
    return fig, ax


def plot_sliding_window_metric(
    scan: SlidingWindowReDisCAResult,
    metric: str = "p_values",
    *,
    max_components: int | None = None,
    center_scale: float = 1000.0,
    time_unit: str = "ms",
    threshold: float | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    colorbar_label: str | None = None,
    ax: Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> tuple[Figure | SubFigure, Axes | None]:
    """Plot a component-by-window heatmap from a sliding-window result."""
    if hasattr(scan, "scan"):
        scan = scan.scan

    if max_components is None:
        max_components = max(result.n_components for result in scan.results)

    matrix = scan.component_metric_matrix(metric, max_components=max_components)
    centers = center_scale * scan.window_centers

    if cmap is None:
        cmap = {
            "p_values": "magma_r",
            "pearson_scores": "viridis",
            "lambdas": "plasma",
        }.get(metric, "viridis")
    if metric == "p_values":
        vmin = 0.0 if vmin is None else vmin
        vmax = 1.0 if vmax is None else vmax
        threshold = 0.05 if threshold is None else threshold
    if colorbar_label is None:
        colorbar_label = {
            "p_values": "Permutation p-value",
            "pearson_scores": "Pearson r",
            "lambdas": "Eigenvalue",
        }.get(metric, metric)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    else:
        fig = ax.figure

    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[centers[0], centers[-1], -0.5, max_components - 0.5],
    )
    ax.set_xlabel(f"Window center ({time_unit})" if time_unit else "Window center")
    ax.set_ylabel("Component")
    ax.set_yticks(range(max_components))
    ax.set_title(title or metric)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

    if threshold is not None:
        significant_rows, significant_cols = np.where(
            np.isfinite(matrix) & (matrix < threshold)
        )
        if significant_rows.size:
            ax.scatter(
                centers[significant_cols],
                significant_rows,
                marker="*",
                s=70,
                color="white",
                edgecolors="black",
                linewidths=0.3,
            )

    _maybe_save(fig, save_path, dpi)
    return fig, ax


def plot_component_timeseries(
    result: ReDisCAResult,
    idxs: Sequence[int] | None = None,
    *,
    order: str = "lambda",
    pearson_mode: str = "pos",
    time: Sequence[float] | NDArray[np.floating] | None = None,
    time_unit: str = "s",
    show_time_zero: bool = True,
    condition_names: Sequence[str] | None = None,
    legend: str = "auto",
    sharey: bool = True,
    condition_layout: str = "overlay",
    highlight_interval: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> tuple[Figure, np.ndarray]:
    """Plot component time series for selected components and conditions.

    Args:
        result: Output of ``fit_redisca``.
        idxs: Component indices to plot. If omitted, the top 3 components
            are selected according to ``order``.
        order: ``"lambda"`` (default) or ``"pearson"``. Used only when
            ``idxs`` is omitted.
        pearson_mode: Used when ``order="pearson"``.
            ``"pos"`` — rank by highest positive r.
            ``"abs"`` — rank by largest |r|.
        time: Optional x-axis values of length ``T``. If omitted, sample
            indices are used.
        time_unit: Units for the provided ``time`` axis: ``"s"`` or
            ``"ms"``. Ignored when ``time`` is ``None``.
        show_time_zero: If *True* (default) and the provided time axis
            spans zero, add a vertical line at ``t=0``.
        condition_names: Optional names for conditions of length ``C``.
        legend: ``"auto"`` (default), ``"axes"``, ``"figure"``, or
            ``"none"``.
        sharey: Whether component panels should share the y-axis.
        condition_layout: ``"overlay"`` (default) plots all conditions for a
            component on the same axes. ``"separate"`` creates a grid with
            one subplot per component/condition pair for less crowded figures.
        highlight_interval: Optional ``(start, stop)`` interval in displayed
            x-axis units to shade on every panel.
        save_path: If given, save the figure to this file path.
        dpi: Resolution for saved figure (default 150).

    Returns:
        ``(fig, axes)`` where *axes* is a 1-D numpy array in ``"overlay"``
        mode and a 2-D array with shape ``(n_components, n_conditions)`` in
        ``"separate"`` mode.
    """
    C, r, T = result.component_timeseries.shape

    if time is None:
        x = np.arange(T)
        x_label = "Sample"
    else:
        x = np.asarray(time, dtype=np.float64)
        if x.shape != (T,):
            raise ValueError(f"time must have shape ({T},), got {x.shape}")
        if time_unit == "ms":
            x = 1e3 * x
            x_label = "Time (ms)"
        elif time_unit == "s":
            x_label = "Time (s)"
        else:
            raise ValueError(f"time_unit must be 's' or 'ms', got '{time_unit}'")

    if condition_names is None:
        condition_names = [f"Cond {i}" for i in range(C)]
    elif len(condition_names) != C:
        raise ValueError(
            f"len(condition_names)={len(condition_names)} != n_conditions={C}"
        )

    if idxs is None:
        if order == "lambda":
            ranked = np.argsort(-result.lambdas)
        elif order == "pearson":
            if pearson_mode == "pos":
                ranked = np.argsort(-result.pearson_scores)
            elif pearson_mode == "abs":
                ranked = np.argsort(-np.abs(result.pearson_scores))
            else:
                raise ValueError(
                    f"pearson_mode must be 'pos' or 'abs', got '{pearson_mode}'"
                )
        else:
            raise ValueError(f"order must be 'lambda' or 'pearson', got '{order}'")
        idxs = ranked[:min(3, r)].tolist()
    else:
        idxs = [i for i in idxs if i < r]
        if len(idxs) == 0:
            raise ValueError("No valid component indices in idxs.")

    if legend not in ("auto", "axes", "figure", "none"):
        raise ValueError(
            f"legend must be 'auto', 'axes', 'figure', or 'none', got '{legend}'"
        )

    if condition_layout not in ("overlay", "separate"):
        raise ValueError(
            "condition_layout must be 'overlay' or 'separate', "
            f"got '{condition_layout}'"
        )

    if condition_layout == "separate":
        fig, axes = plt.subplots(
            len(idxs),
            C,
            figsize=(max(3.0 * C, 6.0), max(2.4 * len(idxs), 2.8)),
            squeeze=False,
            sharex=True,
            sharey=sharey,
            constrained_layout=True,
        )

        colours = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for row, comp_idx in enumerate(idxs):
            for cond_idx in range(C):
                ax_cur = axes[row, cond_idx]
                color = colours[cond_idx % len(colours)] if colours else None
                ax_cur.plot(
                    x,
                    result.component_timeseries[cond_idx, comp_idx],
                    linewidth=1.5,
                    color=color,
                )

                if show_time_zero and time is not None and x[0] <= 0.0 <= x[-1]:
                    ax_cur.axvline(0.0, color="black", linewidth=0.8, linestyle=":")

                if highlight_interval is not None:
                    ax_cur.axvspan(
                        highlight_interval[0],
                        highlight_interval[1],
                        color="grey",
                        alpha=0.12,
                        linewidth=0,
                    )

                ax_cur.axhline(0, color="grey", linewidth=0.5, linestyle="--")

                if row == 0:
                    ax_cur.set_title(condition_names[cond_idx])
                if row == len(idxs) - 1:
                    ax_cur.set_xlabel(x_label)
                if cond_idx == 0:
                    ylabel = f"Comp {comp_idx}\nAmplitude"
                    ax_cur.set_ylabel(ylabel)

                stats_text = f"lambda={result.lambdas[comp_idx]:.3f}\nr={result.pearson_scores[comp_idx]:.3f}"
                if result.p_values is not None:
                    stats_text += f"\np={result.p_values[comp_idx]:.3f}"
                if cond_idx == C - 1:
                    ax_cur.text(
                        1.02,
                        0.5,
                        stats_text,
                        transform=ax_cur.transAxes,
                        va="center",
                        ha="left",
                        fontsize=8,
                    )

        _maybe_save(fig, save_path, dpi)
        return fig, axes

    legend_mode = legend
    if legend_mode == "auto":
        legend_mode = "figure" if (len(idxs) > 1 or C > 3) else "axes"

    fig, axes = plt.subplots(
        1,
        len(idxs),
        figsize=(5 * len(idxs), 4),
        squeeze=False,
        sharey=sharey,
    )
    axes = axes.ravel()

    legend_handles = None
    legend_labels = None
    for panel, comp_idx in enumerate(idxs):
        ax_cur = axes[panel]
        for cond_idx in range(C):
            ax_cur.plot(
                x,
                result.component_timeseries[cond_idx, comp_idx],
                label=condition_names[cond_idx],
                linewidth=1.5,
            )

        if show_time_zero and time is not None and x[0] <= 0.0 <= x[-1]:
            ax_cur.axvline(0.0, color="black", linewidth=0.8, linestyle=":")

        if highlight_interval is not None:
            ax_cur.axvspan(
                highlight_interval[0],
                highlight_interval[1],
                color="grey",
                alpha=0.12,
                linewidth=0,
            )

        title_parts = [f"Component {comp_idx}"]
        title_parts.append(f"λ={result.lambdas[comp_idx]:.3f}")
        title_parts.append(f"r={result.pearson_scores[comp_idx]:.3f}")
        if result.p_values is not None:
            title_parts.append(f"p={result.p_values[comp_idx]:.3f}")
        ax_cur.set_title("\n".join(title_parts))
        ax_cur.set_xlabel(x_label)
        ax_cur.set_ylabel("Amplitude")
        ax_cur.axhline(0, color="grey", linewidth=0.5, linestyle="--")

        if legend_mode == "axes":
            ax_cur.legend(fontsize=8)
        elif legend_mode == "figure" and legend_handles is None:
            legend_handles, legend_labels = ax_cur.get_legend_handles_labels()

    if legend_mode == "figure" and legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(C, max(2, len(legend_labels))),
            bbox_to_anchor=(0.5, 1.05),
            fontsize=8,
        )

    if legend_mode == "figure":
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    else:
        fig.tight_layout()
    _maybe_save(fig, save_path, dpi)
    return fig, axes


def plot_patterns(
    result: ReDisCAResult,
    idxs: Sequence[int] | None = None,
    mode: str = "bar",
    normalize: str = "maxabs",
    channel_names: Sequence[str] | None = None,
    ax: Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> tuple[Figure, np.ndarray]:
    """Bar-plot of spatial pattern weights per channel.

    This is a lightweight fallback visualization that works without sensor
    positions. For EEG/MEG topographies with channel geometry, prefer the
    MNE-based helpers in ``redisca.viz_mne``.

    Args:
        result: Output of ``fit_redisca``.
        idxs: Component indices to plot.  Defaults to first 3.
        mode: Currently only ``"bar"`` is supported.
        normalize: How to scale pattern weights for display.
            ``"none"`` — raw values.
            ``"maxabs"`` (default) — divide by max |a| so values
            fall in [-1, 1].
            ``"zscore"`` — z-standardise across channels.
        channel_names: Optional channel labels for the x-axis.
        ax: Ignored (subplots are always created internally).
        save_path: If given, save the figure to this file path.
        dpi: Resolution for saved figure (default 150).

    Returns:
        ``(fig, axes)`` where *axes* is a 1-D numpy array of ``Axes``.
    """
    if mode != "bar":
        raise ValueError(f"mode must be 'bar', got '{mode}'")
    if normalize not in ("none", "maxabs", "zscore"):
        raise ValueError(
            f"normalize must be 'none', 'maxabs' or 'zscore', got '{normalize}'"
        )

    r = result.n_components
    N = result.n_channels

    if idxs is None:
        idxs = list(range(min(3, r)))
    else:
        idxs = [i for i in idxs if i < r]
        if len(idxs) == 0:
            raise ValueError("No valid component indices in idxs.")

    n_panels = len(idxs)
    fig, axes = plt.subplots(1, n_panels, figsize=(max(4, 0.45 * N) * n_panels, 4))
    if n_panels == 1:
        axes = np.array([axes])

    if channel_names is not None:
        if len(channel_names) != N:
            raise ValueError(
                f"len(channel_names)={len(channel_names)} != n_channels={N}"
            )
    else:
        channel_names = [str(i) for i in range(N)]

    for panel, comp_idx in enumerate(idxs):
        a = result.A[:, comp_idx].copy()  # (N,)

        if normalize == "maxabs":
            a /= np.max(np.abs(a)) + 1e-12
        elif normalize == "zscore":
            std = np.std(a)
            a = (a - np.mean(a)) / (std + 1e-12)

        ax_cur = axes[panel]

        colours = ["#4C72B0" if v >= 0 else "#DD8452" for v in a]
        ax_cur.bar(range(N), a, color=colours)
        ax_cur.set_xticks(range(N))
        ax_cur.set_xticklabels(channel_names, rotation=45 if N > 8 else 0, ha="right")
        ax_cur.set_xlabel("Channel")
        ylabel = "Pattern weight"
        if normalize == "maxabs":
            ylabel += " (norm.)"
        elif normalize == "zscore":
            ylabel += " (z)"
        ax_cur.set_ylabel(ylabel)
        ax_cur.set_title(f"Pattern — Comp {comp_idx}")
        ax_cur.axhline(0, color="grey", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    _maybe_save(fig, save_path, dpi)
    return fig, axes
