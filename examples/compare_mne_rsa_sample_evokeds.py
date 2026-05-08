#!/usr/bin/env python3
"""Compare ReDisCA with MNE-RSA on ready MNE sample evoked responses.

Both analyses use the same four condition-averaged EEG responses and the same
auditory-vs-visual target RDM. The two methods answer related but different
questions:

- MNE-RSA asks how strongly the original EEG pattern in each searchlight window
  matches the target RDM.
- ReDisCA asks which spatial components have component RDMs that match the
  target RDM.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

try:
    import mne_rsa
except ImportError as exc:
    raise ImportError(
        "This example needs mne-rsa. Install it with "
        "`.venv/bin/python -m pip install -e '.[mne,mne-rsa]'`."
    ) from exc

from redisca import binary_rdm, sliding_window_fit_redisca_evokeds
from redisca.report import (
    save_figure,
    save_sliding_window_report,
    save_target_rdm_figure,
)
from redisca.viz_mne import plot_pattern_topomap_panel


# =============================================================================
# User settings
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "examples" / "repro_outputs" / "mne_rsa_comparison"

CONDITION_ORDER = [
    "Left Auditory",
    "Right Auditory",
    "Left visual",
    "Right visual",
]

SCAN_TMIN = 0.000
SCAN_TMAX = 0.350
WINDOW_MS = 100.0
STEP_MS = 20.0

# MNE-RSA's temporal radius is half of the analyzed window.
# A 50 ms radius therefore corresponds to a 100 ms window.
MNE_RSA_TEMPORAL_RADIUS = WINDOW_MS / 2000.0
MNE_RSA_SPATIAL_RADIUS = 0.05

RANK: int | str | None = "auto"
PERMUTATION_TEST = True
N_PERM = 100
RANDOM_STATE = 0

MAX_COMPONENTS_TO_PLOT = 6
TOPOMAP_SPHERE = (0.0, 0.0, 0.0, 0.095)


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

evokeds = {
    condition: available_evokeds[condition].copy().pick("eeg", exclude="bads")
    for condition in CONDITION_ORDER
}
evokeds_ordered = [evokeds[condition] for condition in CONDITION_ORDER]

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Shared target RDM
# =============================================================================

target_rdm = binary_rdm(
    CONDITION_ORDER,
    positive_conditions={"Left Auditory", "Right Auditory"},
)

save_target_rdm_figure(
    target_rdm,
    OUTPUT_ROOT / "target_rdm.png",
    title="Target RDM: auditory vs visual",
    condition_names=CONDITION_ORDER,
)


# =============================================================================
# ReDisCA sliding-window analysis
# =============================================================================

redisca_analysis = sliding_window_fit_redisca_evokeds(
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

redisca_scan = redisca_analysis.scan
best_redisca_idx = redisca_analysis.best_window_index(component=0)

save_sliding_window_report(
    redisca_scan,
    OUTPUT_ROOT / "redisca",
    prefix="redisca",
    title_prefix="ReDisCA: auditory vs visual",
    max_components=MAX_COMPONENTS_TO_PLOT,
    center_scale=1000.0,
    highlight_window_index=best_redisca_idx,
)


# =============================================================================
# MNE-RSA analysis
# =============================================================================

# sqeuclidean keeps the data-RDM metric close to ReDisCA's squared
# condition-difference geometry. Pearson is used so both methods report
# the same type of RDM-to-target similarity score.
mne_rsa_temporal = mne_rsa.rsa_evokeds(
    evokeds_ordered,
    target_rdm,
    spatial_radius=None,
    temporal_radius=MNE_RSA_TEMPORAL_RADIUS,
    evoked_rdm_metric="sqeuclidean",
    rsa_metric="pearson",
    tmin=SCAN_TMIN,
    tmax=SCAN_TMAX,
    n_jobs=1,
    verbose=False,
)

mne_rsa_sensor = mne_rsa.rsa_evokeds(
    evokeds_ordered,
    target_rdm,
    spatial_radius=MNE_RSA_SPATIAL_RADIUS,
    temporal_radius=MNE_RSA_TEMPORAL_RADIUS,
    evoked_rdm_metric="sqeuclidean",
    rsa_metric="pearson",
    tmin=SCAN_TMIN,
    tmax=SCAN_TMAX,
    n_jobs=1,
    verbose=False,
)

mne_rsa_scores = np.asarray(mne_rsa_temporal.data[0], dtype=float)
mne_rsa_times_ms = 1000.0 * np.asarray(mne_rsa_temporal.times, dtype=float)
best_mne_rsa_idx = int(np.nanargmax(mne_rsa_scores))
best_mne_rsa_time_ms = float(mne_rsa_times_ms[best_mne_rsa_idx])


# =============================================================================
# Comparison figures
# =============================================================================

redisca_centers_ms = 1000.0 * redisca_scan.window_centers
best_redisca_center_ms = float(redisca_centers_ms[best_redisca_idx])
redisca_scores = redisca_scan.component_metric_matrix(
    "pearson_scores",
    max_components=MAX_COMPONENTS_TO_PLOT,
)
mne_rsa_at_redisca_centers = np.interp(
    redisca_centers_ms,
    mne_rsa_times_ms,
    mne_rsa_scores,
)

comparison_table = pd.DataFrame(
    {
        "window_center_ms": redisca_centers_ms,
        "redisca_component0_pearson": redisca_scores[0],
        "mne_rsa_pearson_interpolated": mne_rsa_at_redisca_centers,
    }
)
comparison_table.to_csv(OUTPUT_ROOT / "comparison_scores.csv", index=False)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(11.0, 8.0),
    constrained_layout=True,
    height_ratios=[1.15, 1.0, 0.55],
)

axes[0].plot(
    mne_rsa_times_ms,
    mne_rsa_scores,
    color="#4C72B0",
    linewidth=2.0,
    label="MNE-RSA: all EEG sensors",
)
axes[0].plot(
    redisca_centers_ms,
    redisca_scores[0],
    color="#DD8452",
    linewidth=2.0,
    marker="o",
    markersize=4,
    label="ReDisCA: component 0",
)
axes[0].axvline(
    best_redisca_center_ms,
    color="#DD8452",
    linestyle="--",
    linewidth=1.0,
)
axes[0].axvline(
    best_mne_rsa_time_ms,
    color="#4C72B0",
    linestyle=":",
    linewidth=1.2,
)
axes[0].set_title("Auditory vs visual similarity over time")
axes[0].set_ylabel("Pearson r")
axes[0].set_ylim(-1.05, 1.05)
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.22)

redisca_heatmap = axes[1].imshow(
    redisca_scores,
    aspect="auto",
    origin="lower",
    cmap="RdBu_r",
    vmin=-1.0,
    vmax=1.0,
    extent=[
        redisca_centers_ms[0],
        redisca_centers_ms[-1],
        -0.5,
        MAX_COMPONENTS_TO_PLOT - 0.5,
    ],
)
axes[1].axvline(
    best_redisca_center_ms,
    color="black",
    linestyle="--",
    linewidth=1.0,
)
axes[1].set_title("ReDisCA component scores")
axes[1].set_ylabel("Component")
axes[1].set_yticks(range(MAX_COMPONENTS_TO_PLOT))
fig.colorbar(
    redisca_heatmap,
    ax=axes[1],
    fraction=0.035,
    pad=0.02,
    label="Pearson r",
)

mne_rsa_heatmap = axes[2].imshow(
    mne_rsa_scores[np.newaxis, :],
    aspect="auto",
    origin="lower",
    cmap="RdBu_r",
    vmin=-1.0,
    vmax=1.0,
    extent=[
        mne_rsa_times_ms[0],
        mne_rsa_times_ms[-1],
        -0.5,
        0.5,
    ],
)
axes[2].axvline(
    best_mne_rsa_time_ms,
    color="black",
    linestyle=":",
    linewidth=1.2,
)
axes[2].set_title("MNE-RSA temporal searchlight score")
axes[2].set_xlabel("Time (ms)")
axes[2].set_yticks([0])
axes[2].set_yticklabels(["RSA"])
fig.colorbar(
    mne_rsa_heatmap,
    ax=axes[2],
    fraction=0.035,
    pad=0.02,
    label="Pearson r",
)

save_figure(fig, OUTPUT_ROOT / "redisca_vs_mne_rsa_scores.png")

topomap_mne_rsa_idx = int(
    np.argmin(np.abs(mne_rsa_times_ms - best_redisca_center_ms))
)
topomap_mne_rsa_time_ms = float(mne_rsa_times_ms[topomap_mne_rsa_idx])

fig = plt.figure(figsize=(10.8, 4.4), constrained_layout=True)
grid = fig.add_gridspec(
    1,
    4,
    width_ratios=[1.0, 0.045, 1.0, 0.045],
    wspace=0.05,
)
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 2])]
cbar_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 3])]

best_redisca_result = redisca_scan.results[best_redisca_idx]
redisca_image = plot_pattern_topomap_panel(
    axes[0],
    best_redisca_result,
    redisca_analysis.info,
    component=0,
    title=(
        "ReDisCA pattern 0\n"
        f"center={best_redisca_center_ms:.1f} ms"
    ),
    colorbar=False,
)
fig.colorbar(
    redisca_image,
    cax=cbar_axes[0],
    label="Pattern weight",
)

mne_image, _ = mne.viz.plot_topomap(
    mne_rsa_sensor.data[:, topomap_mne_rsa_idx],
    mne_rsa_sensor.info,
    axes=axes[1],
    show=False,
    cmap="RdBu_r",
    vlim=(-1.0, 1.0),
    sensors=True,
    contours=6,
    outlines="head",
    sphere=TOPOMAP_SPHERE,
    extrapolate="head",
)
axes[1].set_title(
    "MNE-RSA sensor searchlight\n"
    f"time={topomap_mne_rsa_time_ms:.1f} ms"
)
fig.colorbar(
    mne_image,
    cax=cbar_axes[1],
    label="Pearson r",
)

save_figure(fig, OUTPUT_ROOT / "redisca_vs_mne_rsa_topomaps.png")

print(f"\nSaved outputs to {OUTPUT_ROOT}")
