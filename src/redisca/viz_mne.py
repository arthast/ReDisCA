"""MNE-based visualization helpers for ReDisCA.

This module is optional and only required when sensor geometry-aware plots
such as topomaps or standard MNE evoked figures are needed.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .types import ReDisCAResult
from .validation import validate_component_index, validate_component_indices
from .viz import add_panel_colorbar


DEFAULT_TOPOMAP_SPHERE = (0.0, 0.0, 0.0, 0.095)


def _require_mne():
    """Import MNE lazily and raise a helpful error if it is unavailable."""
    try:
        return importlib.import_module("mne")
    except ImportError as exc:
        raise ImportError(
            "redisca.viz_mne requires MNE-Python. "
            "Install it with `pip install redisca[mne]` or `pip install mne`."
        ) from exc


def _maybe_save_figure(
    fig,
    save_path: str | Path | None,
    dpi: int,
    *,
    pad_inches: float = 0.2,
) -> None:
    """Save a Matplotlib figure or a list of figures if requested."""
    if save_path is None:
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(fig, list):
        if len(fig) == 1:
            fig[0].savefig(
                path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=pad_inches,
            )
            print(f"Saved {path}")
            return

        stem = path.stem
        suffix = path.suffix or ".png"
        for i, item in enumerate(fig):
            item_path = path.with_name(f"{stem}_{i}{suffix}")
            item.savefig(
                item_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=pad_inches,
            )
            print(f"Saved {item_path}")
        return

    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    print(f"Saved {path}")


def _normalize_pattern(
    pattern: np.ndarray,
    normalize: str,
) -> np.ndarray:
    """Normalize a pattern vector for display."""
    pattern = np.asarray(pattern, dtype=np.float64).copy()

    if normalize == "none":
        return pattern
    if normalize == "maxabs":
        return pattern / (np.max(np.abs(pattern)) + 1e-12)
    if normalize == "zscore":
        return (pattern - np.mean(pattern)) / (np.std(pattern) + 1e-12)
    raise ValueError(
        f"normalize must be 'none', 'maxabs' or 'zscore', got '{normalize}'"
    )


def plot_pattern_topomap_panel(
    ax: Axes,
    result: ReDisCAResult,
    info,
    *,
    component: int,
    title: str,
    ch_type: str | None = None,
    normalize: str = "maxabs",
    cmap: str = "RdBu_r",
    sensors: bool = True,
    contours: int = 6,
    outlines: str | dict | None = "head",
    sphere: tuple[float, float, float, float] | str | None = DEFAULT_TOPOMAP_SPHERE,
    extrapolate: str = "head",
    border: str | float = "mean",
    vlim: tuple[float | None, float | None] | None = (-1.0, 1.0),
    colorbar: bool = True,
    colorbar_label: str = "Weight",
):
    """Plot one ReDisCA spatial pattern topomap into an existing axes."""
    component = validate_component_index(
        component,
        result.n_components,
        name="component",
    )
    mne = _require_mne()
    pattern = _normalize_pattern(result.A[:, component], normalize)
    image, _ = mne.viz.plot_topomap(
        pattern,
        info,
        ch_type=ch_type,
        axes=ax,
        show=False,
        cmap=cmap,
        sensors=sensors,
        contours=contours,
        outlines=outlines,
        sphere=sphere,
        extrapolate=extrapolate,
        border=border,
        vlim=vlim,
    )
    ax.set_title(title, fontsize=9, pad=6)
    if colorbar:
        add_panel_colorbar(
            ax,
            image,
            label=colorbar_label,
            size="3%",
            pad=0.02,
            ticks=[-1.0, 0.0, 1.0] if vlim == (-1.0, 1.0) else None,
        )
    return image


def plot_pattern_topomaps(
    result: ReDisCAResult,
    info,
    idxs: Sequence[int] | None = None,
    *,
    ch_type: str | None = None,
    normalize: str = "maxabs",
    cmap: str = "RdBu_r",
    sensors: bool = True,
    names: Sequence[str] | None = None,
    mask=None,
    mask_params: dict | None = None,
    contours: int = 6,
    outlines: str | dict | None = "head",
    sphere: tuple[float, float, float, float] | str | None = DEFAULT_TOPOMAP_SPHERE,
    extrapolate: str = "head",
    border: str | float = "mean",
    vlim: tuple[float | None, float | None] | str | None = "joint",
    colorbar: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> tuple[Figure, np.ndarray]:
    """Plot ReDisCA patterns as topomaps using MNE.

    Args:
        result: Output of ``fit_redisca``.
        info: MNE ``Info`` object aligned with the channels used in ``result``.
        idxs: Component indices to plot. Defaults to the first 3.
        ch_type: Optional channel type forwarded to ``mne.viz.plot_topomap``.
        normalize: Display normalization: ``"none"``, ``"maxabs"``, or
            ``"zscore"``.
        cmap: Colormap passed to MNE.
        sensors: Whether to show sensor markers.
        names: Optional sensor labels.
        mask: Optional boolean sensor mask.
        mask_params: Optional style dictionary for masked sensors.
        contours: Number of contour lines.
        outlines: Head outline style passed to MNE.
        sphere: Optional MNE sphere definition for the head outline. The
            default is a standard 95 mm EEG head sphere, which keeps the
            heatmap boundary visually aligned with the head outline for common
            EEG montages.
        extrapolate: Extrapolation mode. The default ``"head"`` keeps the
            heatmap fill aligned with the visible head outline.
        border: Border handling passed to MNE.
        vlim: Colour limits. ``"joint"`` uses the same symmetric scale for all
            selected components.
        colorbar: Whether to add a shared colorbar.
        save_path: If given, save the figure to this file path.
        dpi: Resolution for saved figure.

    Returns:
        ``(fig, axes)`` where *axes* is a 1-D numpy array.
    """
    if not hasattr(info, "ch_names"):
        raise TypeError("info must expose channel names via `info.ch_names`.")
    if len(info.ch_names) != result.n_channels:
        raise ValueError(
            "info channel count must match result.n_channels: "
            f"{len(info.ch_names)} != {result.n_channels}"
        )

    r = result.n_components
    if idxs is None:
        idxs = list(range(min(3, r)))
    else:
        idxs = validate_component_indices(idxs, r, name="idxs")

    mne = _require_mne()
    patterns = [_normalize_pattern(result.A[:, idx], normalize) for idx in idxs]

    if vlim == "joint":
        vmax = max(float(np.max(np.abs(pattern))) for pattern in patterns)
        vlim_resolved = (-vmax, vmax)
    elif vlim is None or isinstance(vlim, tuple):
        vlim_resolved = vlim
    else:
        raise ValueError("vlim must be None, a (vmin, vmax) tuple, or 'joint'")

    fig_width = 4.4 * len(idxs) + (1.15 if colorbar else 0.0)
    fig, axes = plt.subplots(
        1,
        len(idxs),
        figsize=(fig_width, 4.15),
        squeeze=False,
    )
    axes = axes.ravel()

    image = None
    for panel, (comp_idx, pattern) in enumerate(zip(idxs, patterns)):
        image, _ = mne.viz.plot_topomap(
            pattern,
            info,
            ch_type=ch_type,
            sensors=sensors,
            names=names,
            mask=mask,
            mask_params=mask_params,
            contours=contours,
            outlines=outlines,
            sphere=sphere,
            extrapolate=extrapolate,
            border=border,
            cmap=cmap,
            vlim=vlim_resolved,
            axes=axes[panel],
            show=False,
        )

        title_parts = [f"Pattern {comp_idx}"]
        title_parts.append(f"λ={result.lambdas[comp_idx]:.3f}")
        title_parts.append(f"r={result.pearson_scores[comp_idx]:.3f}")
        if result.p_values is not None:
            title_parts.append(f"p={result.p_values[comp_idx]:.3f}")
        axes[panel].set_title("\n".join(title_parts))

    if colorbar:
        fig.subplots_adjust(
            left=0.04,
            right=0.82,
            bottom=0.10,
            top=0.84,
            wspace=0.55,
        )
    else:
        fig.tight_layout()

    if colorbar and image is not None:
        cbar_ax = fig.add_axes([0.88, 0.22, 0.022, 0.52])
        cbar = fig.colorbar(image, cax=cbar_ax)
        if normalize == "maxabs":
            cbar.set_label("Pattern weight (maxabs-normalized)")
        elif normalize == "zscore":
            cbar.set_label("Pattern weight (z-score)")
        else:
            cbar.set_label("Pattern weight")

    _maybe_save_figure(fig, save_path, dpi)
    return fig, axes


def plot_compare_conditions(
    evokeds,
    *,
    picks=None,
    colors=None,
    linestyles=None,
    styles=None,
    ci: bool = True,
    combine=None,
    title: str | None = None,
    show_sensors=None,
    time_unit: str = "s",
    y_margin: float = 0.12,
    save_path: str | Path | None = None,
    dpi: int = 150,
    **kwargs,
):
    """Wrapper around ``mne.viz.plot_compare_evokeds`` with lazy MNE import."""
    mne = _require_mne()
    fig = mne.viz.plot_compare_evokeds(
        evokeds,
        picks=picks,
        colors=colors,
        linestyles=linestyles,
        styles=styles,
        ci=ci,
        combine=combine,
        title=title,
        show_sensors=show_sensors,
        time_unit=time_unit,
        show=False,
        **kwargs,
    )
    figures = fig if isinstance(fig, list) else [fig]
    if y_margin > 0.0:
        for item in figures:
            for ax in getattr(item, "axes", []):
                y_min, y_max = ax.get_ylim()
                y_span = y_max - y_min
                if not np.isfinite(y_span) or y_span <= 0.0:
                    continue

                upper_pad = float(y_margin) * y_span
                lower_pad = 0.35 * upper_pad
                new_min = y_min - lower_pad
                if y_min >= 0.0:
                    new_min = max(0.0, new_min)
                ax.set_ylim(new_min, y_max + upper_pad)

    _maybe_save_figure(fig, save_path, dpi)
    return fig


def plot_condition_joint(
    evoked,
    *,
    times="peaks",
    title: str = "",
    picks=None,
    ts_args: dict | None = None,
    topomap_args: dict | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
):
    """Wrapper around ``mne.viz.plot_evoked_joint`` with lazy MNE import."""
    mne = _require_mne()
    fig = mne.viz.plot_evoked_joint(
        evoked,
        times=times,
        title=title,
        picks=picks,
        ts_args=ts_args,
        topomap_args=topomap_args,
        show=False,
    )
    _maybe_save_figure(fig, save_path, dpi)
    return fig


__all__ = [
    "plot_pattern_topomap_panel",
    "plot_pattern_topomaps",
    "plot_compare_conditions",
    "plot_condition_joint",
]
