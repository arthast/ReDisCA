#!/usr/bin/env python3
"""Run article-style ReDisCA analysis on prepared ERP CORE N170 data.

This file is intentionally written as a user script, not as library internals.
It assumes preprocessing has already been done by ``prepare_erpcore_n170.py``.

The prepared bundle contains:

- ``X``: averaged responses, shape ``(conditions, channels, timepoints)``
- ``times``: time axis in seconds
- ``condition_order``: condition names used by the rows/columns of RDMs
- ``info.fif``: MNE channel metadata needed for topographic maps

The script below does only the analysis step:

1. Load ready data.
2. Define target RDMs.
3. Run ReDisCA.
4. Save figures, exports, and a JSON summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mne
import numpy as np

N170_ROOT = Path(__file__).resolve().parent
ROOT = N170_ROOT.parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from redisca import binary_rdm, fit_redisca, sliding_window_fit_redisca_ms
from redisca.report import (
    save_evoked_overview,
    save_result_diagnostics,
    save_sliding_window_report,
    save_target_rdm_figure,
    save_window_result_diagnostics,
    summarize_fixed_window_result,
    summarize_sliding_window_scan,
)


# =============================================================================
# User settings
# =============================================================================

# Input files produced by examples/n170/prepare_erpcore_n170.py.
READY_NPZ = N170_ROOT / "prepared" / "erpcore_n170_sub001_ready.npz"
READY_INFO_FIF = N170_ROOT / "prepared" / "erpcore_n170_sub001_info.fif"
READY_METADATA_JSON = N170_ROOT / "prepared" / "erpcore_n170_sub001_ready_metadata.json"

# All figures, CSV files, exported arrays, and summary.json go here.
OUTPUT_ROOT = N170_ROOT / "outputs"

# ReDisCA settings.
# rank="auto" keeps all numerically available dimensions after rank checks.
RANK: int | str | None = "auto"

# Permutation test estimates how surprising the component eigenvalue is under
# shuffled condition labels. Higher N_PERM gives more reliable p-values.
PERMUTATION_TEST = True
N_PERM = 1000
ALPHA = 0.05
RANDOM_STATE = 0

# Article-style analysis windows.
# The article scans meaningful-vs-meaningless effects with 150 ms windows.
MEANINGFUL_WINDOW_MS = 150.0
MEANINGFUL_STEP_MS = 25.0

# Face-specific and car-specific N170 analyses use the N170-centered window.
CATEGORY_WINDOW_START_S = 0.150
CATEGORY_WINDOW_STOP_S = 0.250


# =============================================================================
# Load prepared ERP CORE N170 data
# =============================================================================

# X has shape (C, N, T):
# C = number of conditions, N = EEG channels, T = time samples.
with np.load(READY_NPZ, allow_pickle=False) as data:
    X = np.asarray(data["X"], dtype=np.float64)
    times = np.asarray(data["times"], dtype=np.float64)
    condition_order = [str(name) for name in data["condition_order"].tolist()]
    sfreq = float(data["sfreq"])
    epoch_counts = {
        condition: int(count)
        for condition, count in zip(condition_order, data["epoch_counts"])
    }

# MNE Info stores channel names, sampling frequency, and sensor positions.
# ReDisCA itself can work without it, but topomap plots need this geometry.
info = mne.io.read_info(READY_INFO_FIF, verbose="ERROR")

with READY_METADATA_JSON.open("r", encoding="utf-8") as handle:
    ready_metadata = json.load(handle)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# This overview is a sanity-check plot of the prepared condition averages.
# It is not a ReDisCA result yet.
evokeds = {
    condition: mne.EvokedArray(
        X[idx],
        info.copy(),
        tmin=float(times[0]),
        comment=condition,
        verbose="ERROR",
    )
    for idx, condition in enumerate(condition_order)
}
save_evoked_overview(
    evokeds,
    OUTPUT_ROOT / "condition_evoked_overview.png",
    condition_order=condition_order,
    title="ERP CORE N170: prepared condition evoked responses",
)


# =============================================================================
# Target RDMs
# =============================================================================

# A target RDM describes which conditions should be similar or different in
# the component we want ReDisCA to find.
#
# In these binary RDMs:
# - 0 means "these two conditions should be similar";
# - 1 means "these two conditions should be different".

# Meaningful stimuli are face/car. Meaningless stimuli are scrambled images.
meaningful_rdm = binary_rdm(
    condition_order,
    positive_conditions={"face", "car"},
)

# Face-specific component: face should differ from all three other conditions.
face_rdm = binary_rdm(
    condition_order,
    positive_conditions={"face"},
)

# Car-specific component: car should differ from all three other conditions.
car_rdm = binary_rdm(
    condition_order,
    positive_conditions={"car"},
)

save_target_rdm_figure(
    meaningful_rdm,
    OUTPUT_ROOT / "meaningful_vs_meaningless_target_rdm.png",
    title="Target RDM: meaningful vs meaningless",
    condition_names=condition_order,
)


# =============================================================================
# Meaningful-vs-meaningless sliding-window analysis
# =============================================================================

# Sliding-window analysis repeats ReDisCA on many neighboring time windows.
# This answers: "when does the data best match this target RDM?"
meaningful_scan = sliding_window_fit_redisca_ms(
    X,
    meaningful_rdm,
    sfreq=sfreq,
    window_ms=MEANINGFUL_WINDOW_MS,
    step_ms=MEANINGFUL_STEP_MS,
    times=times,
    rank=RANK,
    permutation_test=PERMUTATION_TEST,
    n_perm=N_PERM,
    alpha=ALPHA,
    random_state=RANDOM_STATE,
)

# The helper below gives a ready JSON-friendly summary:
# best by p-value, best by Pearson r, significant segments, and the window
# closest to the article's meaningful-vs-meaningless effect around 400 ms.
meaningful_summary = summarize_sliding_window_scan(
    meaningful_scan,
    times=times,
    component=0,
    alpha=ALPHA,
    n_perm=N_PERM,
    time_scale=1000.0,
    time_unit="ms",
    reference_center=400.0,
    reference_label="window_around_400_ms",
)

best_meaningful_pvalue_idx = int(
    meaningful_summary["best_by_p_value"]["window_index"]
)
best_meaningful_pearson_idx = int(
    meaningful_summary["best_by_pearson"]["window_index"]
)

# Save heatmaps over time: Pearson r, p-values, eigenvalues, and a binary
# "significant windows" view. The vertical line marks the best p-value window.
meaningful_scan_artifacts = save_sliding_window_report(
    meaningful_scan,
    OUTPUT_ROOT,
    prefix="meaningful_vs_meaningless",
    title_prefix="ERP CORE N170: meaningful vs meaningless window scan",
    max_components=6,
    center_scale=1000.0,
    alpha=ALPHA,
    highlight_window_index=best_meaningful_pvalue_idx,
    include_significance=True,
)

# Save detailed figures for the best p-value window.
meaningful_best_dir = OUTPUT_ROOT / "meaningful_vs_meaningless_best_window"
save_window_result_diagnostics(
    meaningful_scan,
    best_meaningful_pvalue_idx,
    meaningful_best_dir,
    times=times,
    time_unit="ms",
    info=info,
    condition_names=condition_order,
    target_title="Target RDM: meaningful vs meaningless",
)

# If the highest-correlation window is different, save it too. This keeps the
# report honest: high RDM correlation and statistical significance are separate.
meaningful_best_pearson_dir = meaningful_best_dir
if best_meaningful_pearson_idx != best_meaningful_pvalue_idx:
    meaningful_best_pearson_dir = OUTPUT_ROOT / "meaningful_vs_meaningless_best_pearson_window"
    save_window_result_diagnostics(
        meaningful_scan,
        best_meaningful_pearson_idx,
        meaningful_best_pearson_dir,
        times=times,
        time_unit="ms",
        info=info,
        condition_names=condition_order,
        target_title="Target RDM: meaningful vs meaningless",
    )

meaningful_summary["artifacts"] = {
    "target_rdm": "meaningful_vs_meaningless_target_rdm.png",
    **meaningful_scan_artifacts,
    "best_window_dir": meaningful_best_dir.name,
    "best_pearson_window_dir": meaningful_best_pearson_dir.name,
}


# =============================================================================
# Face-specific fixed-window analysis
# =============================================================================

# For classic N170, we inspect the fixed 150-250 ms window.
category_window_mask = (
    (times >= CATEGORY_WINDOW_START_S)
    & (times <= CATEGORY_WINDOW_STOP_S)
)
if not category_window_mask.any():
    raise ValueError(
        f"No samples found in category window "
        f"{CATEGORY_WINDOW_START_S:g}-{CATEGORY_WINDOW_STOP_S:g} s."
    )

category_window_times = times[category_window_mask]

face_result = fit_redisca(
    X[:, :, category_window_mask],
    face_rdm,
    rank=RANK,
    permutation_test=PERMUTATION_TEST,
    n_perm=N_PERM,
    alpha=ALPHA,
    random_state=RANDOM_STATE,
)

save_result_diagnostics(
    face_result,
    OUTPUT_ROOT / "face_specific",
    times=category_window_times,
    time_unit="ms",
    info=info,
    condition_names=condition_order,
    target_title="Target RDM: face-specific N170",
)

face_summary = summarize_fixed_window_result(
    face_result,
    times=category_window_times,
    alpha=ALPHA,
    time_scale=1000.0,
    time_unit="ms",
)
face_summary["artifacts"] = {
    "target_rdm": "target_rdm.png",
    "top_component_rdms": "top_component_rdms.png",
    "component_scores": "component_scores.png",
    "component_lambdas": "component_lambdas.png",
    "component_timeseries_by_condition": "component_timeseries_by_condition.png",
    "pattern_topomaps": "pattern_topomaps.png",
    "export_dir": "export",
}


# =============================================================================
# Car-specific fixed-window analysis
# =============================================================================

# The article also reports a car-specific analysis in the same N170 window.
car_result = fit_redisca(
    X[:, :, category_window_mask],
    car_rdm,
    rank=RANK,
    permutation_test=PERMUTATION_TEST,
    n_perm=N_PERM,
    alpha=ALPHA,
    random_state=RANDOM_STATE,
)

save_result_diagnostics(
    car_result,
    OUTPUT_ROOT / "car_specific",
    times=category_window_times,
    time_unit="ms",
    info=info,
    condition_names=condition_order,
    target_title="Target RDM: car-specific N170",
)

car_summary = summarize_fixed_window_result(
    car_result,
    times=category_window_times,
    alpha=ALPHA,
    time_scale=1000.0,
    time_unit="ms",
)
car_summary["artifacts"] = {
    "target_rdm": "target_rdm.png",
    "top_component_rdms": "top_component_rdms.png",
    "component_scores": "component_scores.png",
    "component_lambdas": "component_lambdas.png",
    "component_timeseries_by_condition": "component_timeseries_by_condition.png",
    "pattern_topomaps": "pattern_topomaps.png",
    "export_dir": "export",
}


# =============================================================================
# Summary
# =============================================================================

# summary.json is meant to be read by a person before opening the figures.
# The article_comparison block says what the article found and whether this
# particular run reproduced that statistically according to permutation p-values.
summary = {
    "dataset": "ERP CORE N170 / sub-001 prepared bundle",
    "ready_npz": str(READY_NPZ),
    "ready_info_fif": str(READY_INFO_FIF),
    "conditions": condition_order,
    "epoch_counts": epoch_counts,
    "data_shape": {
        "n_conditions": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_timepoints": int(X.shape[2]),
        "sfreq": sfreq,
        "time_start_ms": float(1000.0 * times[0]),
        "time_stop_ms": float(1000.0 * times[-1]),
    },
    "ready_bundle_metadata": ready_metadata,
    "analysis_settings": {
        "rank": RANK,
        "permutation_test": PERMUTATION_TEST,
        "n_perm": N_PERM,
        "alpha": ALPHA,
        "random_state": RANDOM_STATE,
        "meaningful_window_ms": MEANINGFUL_WINDOW_MS,
        "meaningful_step_ms": MEANINGFUL_STEP_MS,
        "category_window_start_ms": float(1000.0 * CATEGORY_WINDOW_START_S),
        "category_window_stop_ms": float(1000.0 * CATEGORY_WINDOW_STOP_S),
    },
    "top_level_artifacts": {
        "condition_evoked_overview": "condition_evoked_overview.png",
    },
    "meaningful_vs_meaningless": meaningful_summary,
    "face_specific": face_summary,
    "car_specific": car_summary,
    "article_comparison": {
        "meaningful_vs_meaningless": {
            "article_expected": (
                "A significant first component around 400 ms, spanning "
                "three adjacent windows, with occipital topography."
            ),
            "current_run": {
                "has_significant_segment_component_0": meaningful_summary[
                    "has_significant_segment"
                ],
                "best_by_p_value": meaningful_summary["best_by_p_value"],
                "window_around_400_ms": meaningful_summary["window_around_400_ms"],
            },
        },
        "face_specific": {
            "article_expected": (
                "One significant face-specific component in the N170-centered "
                "150-250 ms window."
            ),
            "current_run": {
                "has_significant_component": face_summary[
                    "has_significant_component"
                ],
                "best_by_pearson": face_summary["best_by_pearson"],
                "best_by_p_value": face_summary["best_by_p_value"],
            },
        },
        "car_specific": {
            "article_expected": (
                "Two significant car-specific components in the same "
                "N170-centered window."
            ),
            "current_run": {
                "has_significant_component": car_summary[
                    "has_significant_component"
                ],
                "best_by_pearson": car_summary["best_by_pearson"],
                "best_by_p_value": car_summary["best_by_p_value"],
            },
        },
        "note": (
            "This script reproduces the analysis structure from the article. "
            "A result is counted as reproduced statistically only when the "
            "permutation p-values pass alpha."
        ),
    },
}

with (OUTPUT_ROOT / "summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print(json.dumps(summary, indent=2))
print(f"\nSaved outputs to {OUTPUT_ROOT}")
