#!/usr/bin/env python3
"""Compact ReDisCA analysis on prepared ERP CORE N170 data.

This example starts from ready condition-averaged data produced by
``prepare_erpcore_n170.py``. It follows the same style as compact MNE/RSA
examples: load data, define model RDMs, run analysis, plot the results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N170_ROOT = Path(__file__).resolve().parent

import redisca
from redisca.report import (
    save_component_summary_figure,
    save_scan_overview_figure,
    save_window_sequence_figure,
)
from redisca.summary import best_component_by_p_value
from redisca.viz import (
    plot_component_timeseries,
    plot_rdm,
    plot_sliding_window_metric,
    plot_top_component_rdms,
)
from redisca.viz_mne import plot_compare_conditions, plot_pattern_topomaps


# =============================================================================
# User settings
# =============================================================================

READY_NPZ = N170_ROOT / "prepared" / "erpcore_n170_sub001_ready.npz"
READY_INFO_FIF = N170_ROOT / "prepared" / "erpcore_n170_sub001_info.fif"
OUTPUT_ROOT = N170_ROOT / "outputs"
FIGURE_ROOT = OUTPUT_ROOT / "figures"

RANK: int | str | None = "auto"
PERMUTATION_TEST = True
# The library uses one permutation test: reshuffle target-RDM upper-triangle
# entries and report component-wise p-values.
N_PERM = 1000
ALPHA = 0.05
RANDOM_STATE = 0


def output_path(*parts: str) -> Path | None:
    """Return a save path for plot helpers, or None when saving is disabled."""
    return FIGURE_ROOT.joinpath(*parts)


FIGURE_ROOT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Load ready data
# =============================================================================

# The prepared bundle is loaded as a dict: condition name -> MNE Evoked.
# This keeps the rest of the script close to MNE-style examples.
evokeds = redisca.load_evoked_bundle(READY_NPZ, READY_INFO_FIF)
condition_order = list(evokeds)
X, times, info = redisca.evokeds_to_tensor(evokeds, condition_order)

plot_compare_conditions(
    evokeds,
    combine="gfp",
    title="ERP CORE N170: condition averages",
    save_path=output_path("overview", "condition_evoked_overview.png"),
)


# =============================================================================
# Meaningful-vs-meaningless sliding-window analysis
# =============================================================================

meaningful_rdm = redisca.binary_rdm(
    condition_order,
    positive_conditions={"face", "car"},
)
plot_rdm(
    meaningful_rdm,
    title="Target RDM: meaningful vs meaningless",
    condition_names=condition_order,
    show_values=True,
    save_path=output_path("meaningful_vs_meaningless", "target_rdm.png"),
)

meaningful_scan = redisca.sliding_window_fit_redisca_evokeds(
    evokeds,
    meaningful_rdm,
    window_ms=150.0,
    step_ms=25.0,
    rank=RANK,
    permutation_test=PERMUTATION_TEST,
    n_perm=N_PERM,
    alpha=ALPHA,
    random_state=RANDOM_STATE,
)

plot_sliding_window_metric(
    meaningful_scan,
    metric="p_values",
    max_components=6,
    threshold=ALPHA,
    mark_threshold=False,
    title="Meaningful vs meaningless: permutation p-values",
    save_path=output_path("meaningful_vs_meaningless", "p_values.png"),
)

meaningful_centers_ms = 1000.0 * meaningful_scan.window_centers
meaningful_window_idx = int(np.argmin(np.abs(meaningful_centers_ms - 400.0)))
meaningful_result = meaningful_scan.results[meaningful_window_idx]
plot_pattern_topomaps(
    meaningful_result,
    meaningful_scan.info,
    idxs=[0],
    save_path=output_path("meaningful_vs_meaningless", "pattern.png"),
)

plot_top_component_rdms(
    meaningful_result,
    k=1,
    condition_names=condition_order,
    show_values=True,
    save_path=output_path("meaningful_vs_meaningless", "rdms.png"),
)

sequence_start = max(0, meaningful_window_idx - 1)
sequence_stop = min(meaningful_scan.n_windows, sequence_start + 3)
sequence_start = max(0, sequence_stop - 3)
meaningful_sequence_indices = list(range(sequence_start, sequence_stop))
save_scan_overview_figure(
    meaningful_scan,
    FIGURE_ROOT / "meaningful_vs_meaningless" / "scan_overview.png",
    info=info,
    condition_names=condition_order,
    title="Meaningful vs meaningless: sliding-window summary",
    alpha=ALPHA,
    selected_window_index=meaningful_window_idx,
)
save_window_sequence_figure(
    meaningful_scan,
    FIGURE_ROOT / "meaningful_vs_meaningless" / "window_sequence.png",
    info=info,
    times=times,
    condition_names=condition_order,
    title="Meaningful vs meaningless: windows around 400 ms",
    window_indices=meaningful_sequence_indices,
    X_full=X,
    alpha=ALPHA,
)


# =============================================================================
# Face-specific N170 analysis
# =============================================================================

face_rdm = redisca.binary_rdm(
    condition_order,
    positive_conditions={"face"},
)
face = redisca.fit_redisca_evokeds(
    evokeds,
    face_rdm,
    tmin=0.150,
    tmax=0.250,
    rank=RANK,
    permutation_test=PERMUTATION_TEST,
    n_perm=N_PERM,
    alpha=ALPHA,
    random_state=RANDOM_STATE,
)
face_component = best_component_by_p_value(face)

plot_pattern_topomaps(
    face.result,
    face.info,
    idxs=[face_component],
    save_path=output_path("face_specific", "pattern.png"),
)

plot_top_component_rdms(
    face.result,
    k=2,
    condition_names=condition_order,
    show_values=True,
    save_path=output_path("face_specific", "rdms.png"),
)

plot_component_timeseries(
    face.result,
    idxs=[face_component],
    time=face.times,
    time_unit="s",
    condition_names=condition_order,
    save_path=output_path("face_specific", "timeseries.png"),
)

category_highlight_ms = (float(1000.0 * face.times[0]), float(1000.0 * face.times[-1]))
save_component_summary_figure(
    face.result,
    FIGURE_ROOT / "face_specific" / "component_summary.png",
    info=info,
    times=times,
    condition_names=condition_order,
    title="Face-specific N170: component summary",
    component_indices=[face_component],
    X_full=X,
    highlight_interval=category_highlight_ms,
    alpha=ALPHA,
)


# =============================================================================
# Car-specific N170 analysis
# =============================================================================

car_rdm = redisca.binary_rdm(
    condition_order,
    positive_conditions={"car"},
)
car = redisca.fit_redisca_evokeds(
    evokeds,
    car_rdm,
    tmin=0.150,
    tmax=0.250,
    rank=RANK,
    permutation_test=PERMUTATION_TEST,
    n_perm=N_PERM,
    alpha=ALPHA,
    random_state=RANDOM_STATE,
)
car_component = best_component_by_p_value(car)

plot_pattern_topomaps(
    car.result,
    car.info,
    idxs=[car_component],
    save_path=output_path("car_specific", "pattern.png"),
)

plot_top_component_rdms(
    car.result,
    k=2,
    condition_names=condition_order,
    show_values=True,
    save_path=output_path("car_specific", "rdms.png"),
)

car_components = [car_component]
if car.n_components > 1:
    if car.p_values is not None and np.isfinite(car.p_values).any():
        car_components = [
            int(idx)
            for idx in np.argsort(car.p_values)[: min(2, car.n_components)]
        ]
    else:
        car_components = list(range(min(2, car.n_components)))

save_component_summary_figure(
    car.result,
    FIGURE_ROOT / "car_specific" / "component_summary.png",
    info=info,
    times=times,
    condition_names=condition_order,
    title="Car-specific N170: component summary",
    component_indices=car_components,
    X_full=X,
    highlight_interval=category_highlight_ms,
    alpha=ALPHA,
)

print(f"Saved figures to {FIGURE_ROOT}")
plt.close("all")
