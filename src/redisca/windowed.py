"""Sliding-window analysis helpers for ReDisCA."""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray

from .fit import fit_redisca
from .types import ReDisCAResult, SlidingWindowReDisCAResult
from .validation import validate_inputs


def _validate_positive_int(value: int, *, name: str) -> int:
    """Validate a strictly positive integer parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer, got {type(value).__name__}")

    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def sliding_window_fit_redisca(
    X: NDArray[np.floating] | list[NDArray[np.floating]],
    target_rdm: NDArray[np.floating],
    *,
    window_size: int,
    step_size: int = 1,
    start: int = 0,
    stop: int | None = None,
    times: NDArray[np.floating] | None = None,
    rank: int | str | None = "auto",
    tol: float = 1e-10,
    permutation_test: bool = False,
    n_perm: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> SlidingWindowReDisCAResult:
    """Fit ReDisCA over a sequence of sliding time windows.

    Args:
        X: Evoked data of shape ``(C, N, T)`` or a list of ``C`` matrices
            of shape ``(N, T)``.
        target_rdm: Target RDM of shape ``(C, C)``.
        window_size: Window length in samples.
        step_size: Step between consecutive windows in samples.
        start: Inclusive starting sample for the scan.
        stop: Exclusive stopping sample for the scan. Defaults to ``T``.
        times: Optional time axis of shape ``(T,)``. When given, window centers
            are returned in these units instead of raw sample indices.
        rank: Forwarded to :func:`fit_redisca`.
        tol: Forwarded to :func:`fit_redisca`.
        permutation_test: Forwarded to :func:`fit_redisca`.
        n_perm: Forwarded to :func:`fit_redisca`.
        alpha: Forwarded to :func:`fit_redisca`.
        random_state: Forwarded to :func:`fit_redisca`.

    Returns:
        :class:`SlidingWindowReDisCAResult` with one fit per window.

    Raises:
        TypeError: If window parameters are not integers.
        ValueError: If the scan bounds are invalid or no windows fit.
    """
    window_size = _validate_positive_int(window_size, name="window_size")
    step_size = _validate_positive_int(step_size, name="step_size")

    validated = validate_inputs(X, target_rdm)
    X_valid = validated.X
    target_rdm = validated.D

    T = validated.n_timepoints

    if isinstance(start, (bool, np.bool_)) or not isinstance(start, (int, np.integer)):
        raise TypeError(f"start must be an integer, got {type(start).__name__}")
    start = int(start)
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")

    if stop is None:
        stop = T
    elif isinstance(stop, (bool, np.bool_)) or not isinstance(stop, (int, np.integer)):
        raise TypeError(f"stop must be an integer or None, got {type(stop).__name__}")
    else:
        stop = int(stop)

    if stop > T:
        raise ValueError(f"stop must be <= n_timepoints={T}, got {stop}")
    if start >= stop:
        raise ValueError(f"Require start < stop, got start={start}, stop={stop}")
    if window_size > (stop - start):
        raise ValueError(
            f"window_size={window_size} does not fit into the requested interval "
            f"[{start}, {stop}) of length {stop - start}."
        )

    if times is not None:
        times = np.asarray(times, dtype=np.float64)
        if times.shape != (T,):
            raise ValueError(f"times must have shape ({T},), got {times.shape}")

    window_starts = np.arange(start, stop - window_size + 1, step_size, dtype=int)
    if window_starts.size == 0:
        raise ValueError("No valid sliding windows fit the requested parameters.")

    window_stops = window_starts + window_size
    window_centers = np.empty(window_starts.shape[0], dtype=np.float64)
    results: List[ReDisCAResult] = []

    for idx, (window_start, window_stop) in enumerate(zip(window_starts, window_stops)):
        X_window = X_valid[:, :, window_start:window_stop]
        results.append(
            fit_redisca(
                X_window,
                target_rdm,
                rank=rank,
                tol=tol,
                permutation_test=permutation_test,
                n_perm=n_perm,
                alpha=alpha,
                random_state=random_state,
            )
        )

        if times is None:
            window_centers[idx] = 0.5 * (window_start + window_stop - 1)
        else:
            window_centers[idx] = 0.5 * (times[window_start] + times[window_stop - 1])

    return SlidingWindowReDisCAResult(
        results=results,
        window_starts=window_starts,
        window_stops=window_stops,
        window_centers=window_centers,
        sample_times=times,
    )


__all__ = [
    "sliding_window_fit_redisca",
]
