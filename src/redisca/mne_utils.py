"""Small MNE interoperability helpers for ReDisCA workflows.

The helpers in this module keep dataset examples focused on scientific choices:
condition definitions, target RDMs, windows, and statistics. MNE is imported
lazily only by helpers that need it, so the core package remains usable without
the optional MNE dependency.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .fit import fit_redisca
from .types import EvokedReDisCAResult, EvokedSlidingWindowReDisCAResult
from .windowed import sliding_window_fit_redisca_ms


def _require_mne():
    try:
        return importlib.import_module("mne")
    except ImportError as exc:
        raise ImportError(
            "redisca.mne_utils requires MNE-Python for this operation. "
            "Install it with `pip install redisca[mne]` or `pip install mne`."
        ) from exc


def _codes_array(codes: Sequence[int] | range) -> NDArray[np.int_]:
    return np.asarray(list(codes), dtype=int)


def _resolve_condition_order(
    evokeds: Mapping[str, Any],
    condition_order: Sequence[str] | None,
) -> list[str]:
    if condition_order is None:
        condition_order = list(evokeds)
    condition_order = list(condition_order)
    if len(condition_order) == 0:
        raise ValueError("condition_order must contain at least one condition")
    return condition_order


def _time_selection_indices(
    times: NDArray[np.float64],
    *,
    tmin: float | None,
    tmax: float | None,
) -> NDArray[np.int_]:
    if tmin is not None and not np.isfinite(float(tmin)):
        raise ValueError(f"tmin must be finite or None, got {tmin}")
    if tmax is not None and not np.isfinite(float(tmax)):
        raise ValueError(f"tmax must be finite or None, got {tmax}")

    lower = -np.inf if tmin is None else float(tmin)
    upper = np.inf if tmax is None else float(tmax)
    if lower > upper:
        raise ValueError(f"Require tmin <= tmax, got tmin={tmin}, tmax={tmax}")

    indices = np.flatnonzero((times >= lower) & (times <= upper))
    if indices.size == 0:
        raise ValueError(
            f"No time samples fall inside the requested interval "
            f"[{lower:g}, {upper:g}] seconds."
        )
    return indices


def condition_epoch_counts(
    epochs: Any,
    event_code_groups: Mapping[str, Sequence[int] | range],
) -> dict[str, int]:
    """Count accepted epochs per condition from an MNE ``Epochs`` object."""
    event_values = np.asarray(epochs.events[:, 2])
    return {
        condition: int(np.isin(event_values, _codes_array(codes)).sum())
        for condition, codes in event_code_groups.items()
    }


def average_conditions(
    epochs: Any,
    event_code_groups: Mapping[str, Sequence[int] | range],
    *,
    condition_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Average epochs into named evoked responses.

    Args:
        epochs: MNE ``Epochs`` object. The third column of ``epochs.events`` is
            used as the event code.
        event_code_groups: Mapping from condition name to event codes.
        condition_order: Optional explicit order. When omitted, the mapping
            insertion order is used.

    Returns:
        Dictionary ``condition -> Evoked``.
    """
    if condition_order is None:
        condition_order = list(event_code_groups)

    event_values = np.asarray(epochs.events[:, 2])
    evokeds: dict[str, Any] = {}
    for condition in condition_order:
        if condition not in event_code_groups:
            raise KeyError(f"Unknown condition '{condition}' in condition_order")

        mask = np.isin(event_values, _codes_array(event_code_groups[condition]))
        if not mask.any():
            raise ValueError(f"No epochs found for condition '{condition}'")
        evokeds[condition] = epochs[mask].average()

    return evokeds


def evokeds_to_tensor(
    evokeds: Mapping[str, Any],
    condition_order: Sequence[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64], Any]:
    """Convert ordered MNE evokeds into ReDisCA's ``(C, N, T)`` tensor."""
    if len(condition_order) == 0:
        raise ValueError("condition_order must contain at least one condition")

    missing = [name for name in condition_order if name not in evokeds]
    if missing:
        raise KeyError(f"Missing evoked responses for conditions: {missing}")

    first = evokeds[condition_order[0]]
    times = np.asarray(first.times, dtype=np.float64).copy()
    ch_names = tuple(getattr(first, "ch_names", ()))

    data = []
    for name in condition_order:
        evoked = evokeds[name]
        evoked_times = np.asarray(evoked.times, dtype=np.float64)
        if evoked.data.ndim != 2:
            raise ValueError(f"Evoked '{name}' data must have shape (N, T)")
        if evoked_times.shape != times.shape or not np.allclose(evoked_times, times):
            raise ValueError(f"Evoked '{name}' has a different time axis")
        if ch_names and tuple(getattr(evoked, "ch_names", ())) != ch_names:
            raise ValueError(f"Evoked '{name}' has different channel names")
        data.append(np.asarray(evoked.data, dtype=np.float64))

    X = np.stack(data, axis=0)
    return X, times, first.info.copy()


def load_evoked_bundle(
    data_path: str | Path,
    info_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a prepared ReDisCA ``.npz`` bundle as MNE ``Evoked`` objects.

    The bundle must contain ``X`` with shape ``(conditions, channels,
    timepoints)``, ``times`` in seconds, ``condition_order``, and ``sfreq``.
    If ``info_path`` is provided, sensor metadata is read from that MNE ``.fif``
    file so geometry-aware plots such as topomaps work directly.
    """
    mne = _require_mne()
    data_path = Path(data_path)
    with np.load(data_path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        times = np.asarray(data["times"], dtype=np.float64)
        condition_order = [str(name) for name in data["condition_order"].tolist()]
        sfreq = float(data["sfreq"])
        ch_names = (
            [str(name) for name in data["ch_names"].tolist()]
            if "ch_names" in data.files
            else None
        )

    if X.ndim != 3:
        raise ValueError(
            f"X must have shape (conditions, channels, timepoints), got {X.shape}"
        )
    if times.shape != (X.shape[2],):
        raise ValueError(f"times must have shape ({X.shape[2]},), got {times.shape}")
    if len(condition_order) != X.shape[0]:
        raise ValueError(
            "condition_order length must match X.shape[0]: "
            f"{len(condition_order)} != {X.shape[0]}"
        )

    if info_path is None:
        if ch_names is None:
            ch_names = [f"CH{idx:03d}" for idx in range(X.shape[1])]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    else:
        info = mne.io.read_info(info_path, verbose="ERROR")
        if len(info.ch_names) != X.shape[1]:
            raise ValueError(
                "Info channel count must match X.shape[1]: "
                f"{len(info.ch_names)} != {X.shape[1]}"
            )
        if not np.isclose(float(info["sfreq"]), sfreq):
            raise ValueError(
                f"Info sfreq ({info['sfreq']}) does not match bundle sfreq ({sfreq})."
            )

    return {
        condition: mne.EvokedArray(
            X[idx],
            info.copy(),
            tmin=float(times[0]),
            comment=condition,
            verbose="ERROR",
        )
        for idx, condition in enumerate(condition_order)
    }


def fit_redisca_evokeds(
    evokeds: Mapping[str, Any],
    target_rdm: NDArray[np.floating],
    *,
    condition_order: Sequence[str] | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    rank: int | str | None = "auto",
    rank_rtol: float = 1e-8,
    permutation_test: bool = False,
    n_perm: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> EvokedReDisCAResult:
    """Fit ReDisCA directly on condition-averaged MNE evoked responses.

    Args:
        evokeds: Mapping ``condition -> Evoked``.
        target_rdm: Target RDM of shape ``(C, C)`` in ``condition_order``.
        condition_order: Optional explicit condition order. When omitted, the
            mapping insertion order is used.
        tmin: Optional inclusive window start in seconds.
        tmax: Optional inclusive window stop in seconds.
        rank: Forwarded to :func:`redisca.fit_redisca`.
        rank_rtol: Forwarded to :func:`redisca.fit_redisca`.
        permutation_test: Forwarded to :func:`redisca.fit_redisca`.
        n_perm: Forwarded to :func:`redisca.fit_redisca`.
        alpha: Forwarded to :func:`redisca.fit_redisca`.
        random_state: Forwarded to :func:`redisca.fit_redisca`.

    Returns:
        :class:`EvokedReDisCAResult` with the regular ReDisCA result plus the
        selected time axis, MNE ``info``, and condition order.
    """
    order = _resolve_condition_order(evokeds, condition_order)
    X, times, info = evokeds_to_tensor(evokeds, order)
    indices = _time_selection_indices(times, tmin=tmin, tmax=tmax)

    result = fit_redisca(
        X[:, :, indices],
        target_rdm,
        rank=rank,
        rank_rtol=rank_rtol,
        permutation_test=permutation_test,
        n_perm=n_perm,
        alpha=alpha,
        random_state=random_state,
    )
    return EvokedReDisCAResult(
        result=result,
        times=times[indices].copy(),
        info=info,
        condition_order=order,
    )


def sliding_window_fit_redisca_evokeds(
    evokeds: Mapping[str, Any],
    target_rdm: NDArray[np.floating],
    *,
    window_ms: float,
    step_ms: float = 1.0,
    condition_order: Sequence[str] | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    rank: int | str | None = "auto",
    rank_rtol: float = 1e-8,
    permutation_test: bool = False,
    n_perm: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> EvokedSlidingWindowReDisCAResult:
    """Run a ReDisCA sliding-window scan directly on MNE evoked responses.

    ``tmin`` and ``tmax`` restrict where the scan is allowed to start, while
    window centers are still reported on the original MNE time axis.
    """
    order = _resolve_condition_order(evokeds, condition_order)
    X, times, info = evokeds_to_tensor(evokeds, order)
    indices = _time_selection_indices(times, tmin=tmin, tmax=tmax)

    try:
        sfreq = float(info["sfreq"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Evoked info must contain sampling frequency `sfreq`.") from exc

    scan = sliding_window_fit_redisca_ms(
        X,
        target_rdm,
        sfreq=sfreq,
        window_ms=window_ms,
        step_ms=step_ms,
        start=int(indices[0]),
        stop=int(indices[-1]) + 1,
        times=times,
        rank=rank,
        rank_rtol=rank_rtol,
        permutation_test=permutation_test,
        n_perm=n_perm,
        alpha=alpha,
        random_state=random_state,
    )

    return EvokedSlidingWindowReDisCAResult(
        scan=scan,
        times=times,
        info=info,
        condition_order=order,
    )


def make_montage_from_electrodes(
    electrodes,
    *,
    name_col: str = "name",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    coord_scale: float = 0.001,
    coord_frame: str = "head",
):
    """Build an MNE montage from an electrode coordinate table.

    ``coord_scale=0.001`` converts millimetres to metres, which is the format
    expected by MNE's ``make_dig_montage``.
    """
    mne = _require_mne()
    required_cols = [name_col, x_col, y_col, z_col]
    missing = [col for col in required_cols if col not in electrodes.columns]
    if missing:
        raise KeyError(f"Electrode table is missing columns: {missing}")

    ch_pos = {
        str(row[name_col]): (
            np.array([row[x_col], row[y_col], row[z_col]], dtype=float)
            * float(coord_scale)
        )
        for _, row in electrodes.dropna(subset=[x_col, y_col, z_col]).iterrows()
    }
    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame=coord_frame)


__all__ = [
    "average_conditions",
    "condition_epoch_counts",
    "EvokedReDisCAResult",
    "EvokedSlidingWindowReDisCAResult",
    "evokeds_to_tensor",
    "fit_redisca_evokeds",
    "load_evoked_bundle",
    "make_montage_from_electrodes",
    "sliding_window_fit_redisca_evokeds",
]
