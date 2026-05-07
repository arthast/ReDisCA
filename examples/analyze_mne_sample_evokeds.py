#!/usr/bin/env python3
"""Analyze ready MNE sample evoked responses with ReDisCA.

This example is intentionally close to the style of MNE-RSA examples: it uses a
standard MNE dataset, starts from already averaged ``Evoked`` responses, defines
a target RDM, and runs ReDisCA.

No raw EEG/MEG preprocessing is performed here. The goal is to show the normal
library use case: prepared condition responses in, ReDisCA components and
patterns out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mne

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from redisca import (
    binary_rdm,
    fit_redisca_evokeds,
    sliding_window_fit_redisca_evokeds,
)
from redisca.report import (
    save_evoked_overview,
    save_result_diagnostics,
    save_sliding_window_report,
    save_target_rdm_figure,
    summarize_result,
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
SCAN_TMAX = 0.350
WINDOW_MS = 100.0
STEP_MS = 20.0

# ReDisCA settings. Increase N_PERM for a more serious significance estimate.
RANK: int | str | None = "auto"
PERMUTATION_TEST = True
N_PERM = 100
RANDOM_STATE = 0


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

summary: dict[str, object] = {
    "dataset": "MNE sample / sample_audvis-ave.fif",
    "conditions": CONDITION_ORDER,
    "sensor_type": "EEG",
    "target": "auditory vs visual",
    "fixed_window": {
        "window_start_ms": float(1000.0 * analysis.times[0]),
        "window_stop_ms": float(1000.0 * analysis.times[-1]),
        "summary": summarize_result(analysis.result),
        "artifacts_dir": "fixed_window",
    },
}


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
        random_state=RANDOM_STATE,
    )

    scan = scan_analysis.scan
    best_idx = scan_analysis.best_window_index(component=0)
    scan_artifacts = save_sliding_window_report(
        scan,
        OUTPUT_ROOT,
        prefix="auditory_vs_visual",
        title_prefix="MNE sample: auditory vs visual window scan",
        max_components=6,
        center_scale=1000.0,
        highlight_window_index=best_idx,
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

    summary["sliding_window"] = {
        "best_window_index": int(best_idx),
        "best_window_start_ms": float(1000.0 * best_times[0]),
        "best_window_stop_ms": float(1000.0 * best_times[-1]),
        "best_window_summary": summarize_result(best_result),
        "artifacts": {
            **scan_artifacts,
            "best_window_dir": "best_sliding_window",
        },
    }


# =============================================================================
# Summary
# =============================================================================

with (OUTPUT_ROOT / "summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print(json.dumps(summary, indent=2))
print(f"\nSaved outputs to {OUTPUT_ROOT}")
