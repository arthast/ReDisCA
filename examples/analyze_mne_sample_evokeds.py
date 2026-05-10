#!/usr/bin/env python3
"""Analyze ready MNE sample evoked responses with ReDisCA.

This example is intentionally close to the style of MNE-RSA examples: it uses a
standard MNE dataset, starts from already averaged ``Evoked`` responses, defines
a target RDM, and runs ReDisCA.

No raw EEG/MEG preprocessing is performed here. The goal is to show the normal
library use case: prepared condition responses in, ReDisCA components and
patterns out.
"""

from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from redisca import (
    binary_rdm,
    evokeds_to_tensor,
    fit_redisca_evokeds,
    sliding_window_fit_redisca_evokeds,
)
from redisca.report import (
    save_evoked_overview,
    save_result_diagnostics,
    save_scan_overview_figure,
    save_sliding_window_report,
    save_target_rdm_figure,
    save_window_sequence_figure,
)


# =============================================================================
# User settings
# =============================================================================

OUTPUT_ROOT = ROOT / "examples" / "repro_outputs" / "mne_sample_evokeds"

# Conditions available in MNE's sample_audvis-ave.fif file.
CONDITION_ORDER = [
    "Left Auditory",
    "Right Auditory",
    "Left visual",
    "Right visual",
]

# Fixed-window analysis in seconds.
FIXED_TMIN = 0.050
FIXED_TMAX = 0.200

# Optional sliding-window scan.
RUN_SLIDING_WINDOW = True
SCAN_TMIN = 0.000
SCAN_TMAX = 0.500
WINDOW_MS = 100.0
STEP_MS = 20.0

# ReDisCA settings. Permutation testing uses the library's single test:
# reshuffle target-RDM upper-triangle entries and report component-wise p-values.
# Increase N_PERM for a more serious significance estimate.
RANK: int | str | None = "auto"
PERMUTATION_TEST = True
N_PERM = 100
ALPHA = 0.05
RANDOM_STATE = 0

# Components/windows to summarize after the sliding-window scan.
MAX_SCAN_COMPONENTS = 4
COMPONENTS_FOR_WINDOW_TOPO = (0, 1)
MAX_TOPO_WINDOWS_PER_COMPONENT = 3


# =============================================================================
# Load ready MNE evoked data
# =============================================================================

sample_root = mne.datasets.sample.data_path(verbose=True)
sample_path = sample_root / "MEG" / "sample"
evoked_path = sample_path / "sample_audvis-ave.fif"

evokeds_list = mne.read_evokeds(
    evoked_path,
    baseline=(None, 0.0),
    proj=True,
    verbose="ERROR",
)
available_evokeds = {evoked.comment: evoked for evoked in evokeds_list}

missing_conditions = [
    condition for condition in CONDITION_ORDER
    if condition not in available_evokeds
]
if missing_conditions:
    raise ValueError(
        f"Missing conditions {missing_conditions}. "
        f"Available evokeds: {sorted(available_evokeds)}"
    )

# ReDisCA in this example is run on EEG sensors only. Bad channels are excluded
# using the metadata already stored in the MNE sample file.
evokeds = {
    condition: available_evokeds[condition].copy().pick("eeg", exclude="bads")
    for condition in CONDITION_ORDER
}
X, times, info = evokeds_to_tensor(evokeds, CONDITION_ORDER)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
save_evoked_overview(
    evokeds,
    OUTPUT_ROOT / "condition_evoked_overview.png",
    condition_order=CONDITION_ORDER,
    title="MNE sample: condition evoked responses",
)


# =============================================================================
# Target RDM: auditory versus visual
# =============================================================================

target_rdm = binary_rdm(
    CONDITION_ORDER,
    positive_conditions={"Left Auditory", "Right Auditory"},
)

save_target_rdm_figure(
    target_rdm,
    OUTPUT_ROOT / "auditory_vs_visual_target_rdm.png",
    title="Target RDM: auditory vs visual",
)


# =============================================================================
# Fixed-window ReDisCA
# =============================================================================

analysis = fit_redisca_evokeds(
    evokeds,
    target_rdm,
    condition_order=CONDITION_ORDER,
    tmin=FIXED_TMIN,
    tmax=FIXED_TMAX,
    rank=RANK,
    permutation_test=PERMUTATION_TEST,
    n_perm=N_PERM,
    alpha=ALPHA,
    random_state=RANDOM_STATE,
)

save_result_diagnostics(
    analysis.result,
    OUTPUT_ROOT / "fixed_window",
    times=analysis.times,
    time_unit="ms",
    info=analysis.info,
    condition_names=analysis.condition_order,
    target_title="Target RDM: auditory vs visual",
)


# =============================================================================
# Optional sliding-window ReDisCA
# =============================================================================

if RUN_SLIDING_WINDOW:
    scan_analysis = sliding_window_fit_redisca_evokeds(
        evokeds,
        target_rdm,
        condition_order=CONDITION_ORDER,
        window_ms=WINDOW_MS,
        step_ms=STEP_MS,
        tmin=SCAN_TMIN,
        tmax=SCAN_TMAX,
        rank=RANK,
        permutation_test=PERMUTATION_TEST,
        n_perm=N_PERM,
        alpha=ALPHA,
        random_state=RANDOM_STATE,
    )

    scan = scan_analysis.scan
    best_idx = scan_analysis.best_window_index(component=0)
    save_sliding_window_report(
        scan,
        OUTPUT_ROOT,
        prefix="auditory_vs_visual",
        title_prefix="MNE sample: auditory vs visual window scan",
        max_components=MAX_SCAN_COMPONENTS,
        center_scale=1000.0,
        alpha=ALPHA,
        highlight_window_index=best_idx,
    )

    p_values = scan.component_metric_matrix(
        "p_values",
        max_components=MAX_SCAN_COMPONENTS,
    )
    for component in COMPONENTS_FOR_WINDOW_TOPO:
        if component >= p_values.shape[0]:
            continue

        component_p = p_values[component]
        finite = np.isfinite(component_p)
        significant_windows = np.flatnonzero(finite & (component_p < ALPHA))

        if significant_windows.size:
            ranked = significant_windows[np.argsort(component_p[significant_windows])]
            selected_idx = int(ranked[0])
            window_indices = np.sort(ranked[:MAX_TOPO_WINDOWS_PER_COMPONENT])
            sequence_label = "significant windows"
        else:
            selected_idx = scan_analysis.best_window_index(component=component)
            ranked = np.argsort(np.where(finite, component_p, np.inf))
            ranked = ranked[np.isfinite(component_p[ranked])]
            window_indices = np.sort(ranked[:MAX_TOPO_WINDOWS_PER_COMPONENT])
            if window_indices.size == 0:
                window_indices = np.array([selected_idx], dtype=int)
            sequence_label = "lowest p-value windows"

        component_dir = OUTPUT_ROOT / f"component_{component}_window_scan"
        save_scan_overview_figure(
            scan,
            component_dir / "scan_overview.png",
            info=info,
            condition_names=CONDITION_ORDER,
            title=(
                "MNE sample: auditory vs visual "
                f"component {component} sliding-window summary"
            ),
            component=component,
            max_components=MAX_SCAN_COMPONENTS,
            alpha=ALPHA,
            selected_window_index=selected_idx,
            center_scale=1000.0,
            time_unit="ms",
        )
        save_window_sequence_figure(
            scan,
            component_dir / "selected_window_topographies.png",
            info=info,
            times=times,
            condition_names=CONDITION_ORDER,
            title=(
                "MNE sample: auditory vs visual "
                f"component {component} {sequence_label}"
            ),
            window_indices=window_indices.tolist(),
            component=component,
            X_full=X,
            alpha=ALPHA,
            time_scale=1000.0,
            time_unit="ms",
        )

    best_result = scan_analysis.best_result(component=0)
    best_times = scan_analysis.best_window_times(component=0)
    save_result_diagnostics(
        best_result,
        OUTPUT_ROOT / "best_sliding_window",
        times=best_times,
        time_unit="ms",
        info=scan_analysis.info,
        condition_names=scan_analysis.condition_order,
        target_title="Target RDM: auditory vs visual",
    )

print(f"\nSaved outputs to {OUTPUT_ROOT}")
