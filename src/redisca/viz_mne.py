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
from matplotlib.figure import Figure

from .types import ReDisCAResult


def _require_mne():
    """Import MNE lazily and raise a helpful error if it is unavailable."""
    try:
        return importlib.import_module("mne")
    except ImportError as exc:
        raise ImportError(
            "redisca.viz_mne requires MNE-Python. "
            "Install it with `pip install redisca[mne]` or `pip install mne`."
        ) from exc


def _maybe_save_figure(fig, save_path: str | Path | None, dpi: int) -> None:
    """Save a Matplotlib figure or a list of figures if requested."""
    if save_path is None:
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(fig, list):
        stem = path.stem
        suffix = path.suffix or ".png"
        for i, item in enumerate(fig):
            item_path = path.with_name(f"{stem}_{i}{suffix}")
            item.savefig(item_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved {item_path}")
        return

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
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
        vlim: Colour limits. ``"joint"`` uses the same symmetric scale for all
            selected components.
        colorbar: Whether to add a shared colorbar.
        save_path: If given, save the figure to this file path.
        dpi: Resolution for saved figure.

    Returns:
        ``(fig, axes)`` where *axes* is a 1-D numpy array.
    """
    mne = _require_mne()

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
        idxs = [i for i in idxs if i < r]
        if len(idxs) == 0:
            raise ValueError("No valid component indices in idxs.")

    patterns = [_normalize_pattern(result.A[:, idx], normalize) for idx in idxs]

    if vlim == "joint":
        vmax = max(float(np.max(np.abs(pattern))) for pattern in patterns)
        vlim_resolved = (-vmax, vmax)
    elif vlim is None or isinstance(vlim, tuple):
        vlim_resolved = vlim
    else:
        raise ValueError("vlim must be None, a (vmin, vmax) tuple, or 'joint'")

    fig, axes = plt.subplots(1, len(idxs), figsize=(4 * len(idxs), 4), squeeze=False)
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

    if colorbar and image is not None:
        cbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.03)
        if normalize == "maxabs":
            cbar.set_label("Pattern weight (maxabs-normalized)")
        elif normalize == "zscore":
            cbar.set_label("Pattern weight (z-score)")
        else:
            cbar.set_label("Pattern weight")

    if colorbar:
        fig.subplots_adjust(left=0.06, right=0.92, bottom=0.08, top=0.84, wspace=0.35)
    else:
        fig.tight_layout()
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
    "plot_pattern_topomaps",
    "plot_compare_conditions",
    "plot_condition_joint",
]
