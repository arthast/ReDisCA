"""JSON-friendly summaries for ReDisCA results."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import ReDisCAResult, SlidingWindowReDisCAResult
from .validation import validate_component_index
from .windowed import best_window_index


def summarize_component(
    result: ReDisCAResult,
    *,
    component: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int | bool | None]:
    """Summarize one component with scalar values."""
    component = validate_component_index(
        component,
        result.n_components,
        name="component",
    )

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
    component = validate_component_index(
        component,
        max(result.n_components for result in scan.results),
        name="component",
    )
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
    if isinstance(window_index, (bool, np.bool_)) or not isinstance(
        window_index,
        (int, np.integer),
    ):
        raise TypeError(
            f"window_index must be an integer, got {type(window_index).__name__}"
        )
    window_index = int(window_index)
    if window_index < 0 or window_index >= scan.n_windows:
        raise ValueError(
            "window_index must satisfy "
            f"0 <= window_index < n_windows={scan.n_windows}, got {window_index}"
        )

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
    component = validate_component_index(
        component,
        max(result.n_components for result in scan.results),
        name="component",
    )
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
    """Summarize a sliding-window scan."""
    component = validate_component_index(
        component,
        max(result.n_components for result in scan.results),
        name="component",
    )
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
    """Summarize one fixed-window ReDisCA fit."""
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


def summarize_result(result: ReDisCAResult) -> dict[str, int | list[float] | None]:
    """Convert a ReDisCA result into a compact summary."""
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
    "significant_window_segments",
    "summarize_component",
    "summarize_fixed_window_result",
    "summarize_result",
    "summarize_sliding_window_scan",
    "summarize_window",
]
