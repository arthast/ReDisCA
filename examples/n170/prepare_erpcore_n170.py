#!/usr/bin/env python3
"""Prepare ERP CORE N170 subject 001 for ReDisCA.

This script is intentionally separate from ``reproduce_erpcore_n170.py``. It
does dataset-specific work once:

1. Load ERP CORE raw EEG.
2. Set channel types, montage, and average reference.
3. Fit ICA on EEG channels.
4. Detect EOG-related ICA components automatically.
5. Save ICA diagnostics so components can be inspected manually.
6. Apply selected ICA exclusions.
7. Epoch, baseline-correct, average conditions, and save a compact ReDisCA
   bundle.

To manually override ICA rejection after inspecting the saved figures, edit
``MANUAL_ICA_EXCLUDE`` below and rerun the script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA

N170_ROOT = Path(__file__).resolve().parent
ROOT = N170_ROOT.parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from redisca.mne_utils import (
    average_conditions,
    condition_epoch_counts,
    evokeds_to_tensor,
    make_montage_from_electrodes,
)


# =============================================================================
# User settings
# =============================================================================

DATA_ROOT = N170_ROOT / "data"
OUTPUT_DIR = N170_ROOT / "prepared"
WORK_DIR = N170_ROOT / "work"
ICA_DIAGNOSTICS_DIR = WORK_DIR / "ica_diagnostics"
ICA_FIF = WORK_DIR / "erpcore_n170_sub001_ica.fif"

# Download missing raw files into DATA_ROOT. The data directory is ignored by git.
DOWNLOAD_DATASET = True
FORCE_DOWNLOAD = False
DATA_SOURCE = "osf"  # "osf" or "manual"

# Lightweight preprocessing used before averaging.
LOW_FREQ = 0.1
HIGH_FREQ = 30.0
EPOCH_TMIN = -0.1
EPOCH_TMAX = 0.5
BASELINE = (EPOCH_TMIN, 0.0)

# ICA settings.
RUN_ICA = True
ICA_N_COMPONENTS: float | int | None = 0.99
ICA_METHOD = "fastica"
ICA_RANDOM_STATE = 0
ICA_FIT_HIGH_PASS = 1.0
ICA_EOG_THRESHOLD = 3.0
EOG_CHANNELS = ["HEOG_left", "HEOG_right", "VEOG_lower"]

# If automatic EOG detection is too conservative or too aggressive, inspect
# examples/n170/work/ica_diagnostics/ and set component indices here.
MANUAL_ICA_EXCLUDE: list[int] = []


# =============================================================================
# Dataset definitions
# =============================================================================

REQUIRED_ERP_CORE_FILES = [
    "sub-001_task-N170_eeg.json",
    "sub-001_task-N170_eeg.set",
    "sub-001_task-N170_events.tsv",
    "sub-001_task-N170_electrodes.tsv",
    "sub-001_task-N170_coordsystem.json",
    "sub-001_task-N170_eeg.fdt",
    "sub-001_task-N170_channels.tsv",
]

ERP_CORE_N170_OSF_URLS = {
    "sub-001_task-N170_eeg.json": (
        "https://files.de-1.osf.io/v1/resources/pfde9/providers/osfstorage//"
        "60075c4eba01090877890aa3"
    ),
    "sub-001_task-N170_eeg.set": (
        "https://files.de-1.osf.io/v1/resources/pfde9/providers/osfstorage//"
        "60075c50ba0109087e8916aa"
    ),
    "sub-001_task-N170_events.tsv": (
        "https://files.de-1.osf.io/v1/resources/pfde9/providers/osfstorage//"
        "60075c5286541a08f914ab2d"
    ),
    "sub-001_task-N170_electrodes.tsv": (
        "https://files.de-1.osf.io/v1/resources/pfde9/providers/osfstorage//"
        "610215220c4cba026dbce4d0"
    ),
    "sub-001_task-N170_coordsystem.json": (
        "https://files.de-1.osf.io/v1/resources/pfde9/providers/osfstorage//"
        "610215260c4cba0277bc7a17"
    ),
    "sub-001_task-N170_eeg.fdt": (
        "https://files.de-1.osf.io/v1/resources/pfde9/providers/osfstorage//"
        "60075c4be80d3708caa58293"
    ),
    "sub-001_task-N170_channels.tsv": (
        "https://files.de-1.osf.io/v1/resources/pfde9/providers/osfstorage//"
        "60075c42e80d3708c5a57469"
    ),
}

CONDITION_ORDER = [
    "face",
    "car",
    "scrambled_face",
    "scrambled_car",
]

EVENT_CODE_GROUPS = {
    "face": range(1, 41),
    "car": range(41, 81),
    "scrambled_face": range(101, 141),
    "scrambled_car": range(141, 181),
}


# =============================================================================
# Load raw ERP CORE data
# =============================================================================

def find_erpcore_n170_subject_001(root: Path) -> Path | None:
    """Find the ERP CORE N170 ``sub-001/eeg`` directory under *root*."""
    preferred_candidates = [
        root / "erpcore_n170" / "sub-001" / "eeg",
        root / "sub-001" / "eeg",
    ]
    for candidate in preferred_candidates:
        if all((candidate / name).exists() for name in REQUIRED_ERP_CORE_FILES):
            return candidate

    for eeg_set in root.rglob("sub-001_task-N170_eeg.set"):
        candidate = eeg_set.parent
        if all((candidate / name).exists() for name in REQUIRED_ERP_CORE_FILES):
            return candidate

    return None


def fetch_n170_from_osf() -> Path:
    """Fetch the ERP CORE N170 subject 001 BIDS/EEGLAB files from OSF."""
    eeg_dir = DATA_ROOT / "erpcore_n170" / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)

    for file_name in REQUIRED_ERP_CORE_FILES:
        destination = eeg_dir / file_name
        if destination.exists() and not FORCE_DOWNLOAD:
            continue

        url = ERP_CORE_N170_OSF_URLS[file_name]
        temporary_destination = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading ERP CORE N170 {file_name} ...")
        try:
            urlretrieve(url, temporary_destination)
            temporary_destination.replace(destination)
        except (OSError, URLError) as exc:
            if temporary_destination.exists():
                temporary_destination.unlink()
            raise RuntimeError(
                "Failed to download an ERP CORE N170 file from OSF.\n"
                f"File: {file_name}\n"
                f"URL: {url}\n"
                f"Destination: {destination}\n"
                "Check internet access, or place the required files manually "
                "under examples/n170/data/erpcore_n170/sub-001/eeg/."
            ) from exc

    return eeg_dir


if DATA_SOURCE not in {"osf", "manual"}:
    raise ValueError("DATA_SOURCE must be 'osf' or 'manual'.")

DATA_ROOT.mkdir(parents=True, exist_ok=True)
eeg_dir = None if FORCE_DOWNLOAD else find_erpcore_n170_subject_001(DATA_ROOT)

if eeg_dir is None and DOWNLOAD_DATASET and DATA_SOURCE == "osf":
    eeg_dir = fetch_n170_from_osf()

if eeg_dir is None:
    expected_dir = DATA_ROOT / "erpcore_n170" / "sub-001" / "eeg"
    missing_list = "\n".join(f"  - {name}" for name in REQUIRED_ERP_CORE_FILES)
    raise FileNotFoundError(
        "ERP CORE N170 subject 001 files are not available.\n"
        f"Expected manual location:\n  {expected_dir}\n"
        f"Required files:\n{missing_list}\n"
        "Set DOWNLOAD_DATASET=True and DATA_SOURCE='osf' to fetch the N170 "
        "files from OSF, or place the files manually in the expected location."
    )

raw = mne.io.read_raw_eeglab(
    eeg_dir / "sub-001_task-N170_eeg.set",
    preload=True,
    verbose="ERROR",
)

channels = pd.read_csv(eeg_dir / "sub-001_task-N170_channels.tsv", sep="\t")
channel_types = {
    row["name"]: row["type"].lower()
    for _, row in channels.iterrows()
}
electrodes = pd.read_csv(eeg_dir / "sub-001_task-N170_electrodes.tsv", sep="\t")

raw.set_channel_types(channel_types, verbose="ERROR")
raw.set_montage(
    make_montage_from_electrodes(electrodes),
    on_missing="ignore",
    verbose="ERROR",
)
raw.set_eeg_reference("average", verbose="ERROR")


# =============================================================================
# ICA artifact correction
# =============================================================================

ica = None
auto_ica_exclude: list[int] = []
eog_scores_by_channel: dict[str, list[float]] = {}
eog_components_by_channel: dict[str, list[int]] = {}

if RUN_ICA:
    ICA_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    raw_for_ica = raw.copy().filter(
        ICA_FIT_HIGH_PASS,
        HIGH_FREQ,
        picks="eeg",
        verbose="ERROR",
    )
    ica = ICA(
        n_components=ICA_N_COMPONENTS,
        method=ICA_METHOD,
        random_state=ICA_RANDOM_STATE,
        max_iter="auto",
        verbose="ERROR",
    )
    ica.fit(raw_for_ica, picks="eeg", verbose="ERROR")

    component_figs = ica.plot_components(
        picks=range(ica.n_components_),
        show=False,
    )
    if not isinstance(component_figs, list):
        component_figs = [component_figs]
    for fig_idx, fig in enumerate(component_figs):
        suffix = "" if len(component_figs) == 1 else f"_{fig_idx}"
        fig.savefig(
            ICA_DIAGNOSTICS_DIR / f"ica_components{suffix}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

    for eog_ch in EOG_CHANNELS:
        if eog_ch not in raw.ch_names:
            continue

        eog_indices, eog_scores = ica.find_bads_eog(
            raw_for_ica,
            ch_name=eog_ch,
            threshold=ICA_EOG_THRESHOLD,
            verbose="ERROR",
        )
        eog_indices = [int(idx) for idx in eog_indices]
        auto_ica_exclude.extend(eog_indices)
        eog_components_by_channel[eog_ch] = eog_indices
        eog_scores_by_channel[eog_ch] = [float(score) for score in eog_scores]

        fig = ica.plot_scores(
            eog_scores,
            exclude=eog_indices,
            labels=eog_ch,
            title=f"ICA EOG scores: {eog_ch}",
            show=False,
        )
        fig.savefig(
            ICA_DIAGNOSTICS_DIR / f"ica_eog_scores_{eog_ch}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

    auto_ica_exclude = sorted(set(auto_ica_exclude))
    ica.exclude = sorted(set(auto_ica_exclude + MANUAL_ICA_EXCLUDE))
    ica.save(ICA_FIF, overwrite=True, verbose="ERROR")

    raw = raw.copy()
    if ica.exclude:
        ica.apply(raw, exclude=ica.exclude, verbose="ERROR")


# =============================================================================
# Epoch and average conditions
# =============================================================================

raw.filter(LOW_FREQ, HIGH_FREQ, picks="eeg", verbose="ERROR")

events_tsv = pd.read_csv(eeg_dir / "sub-001_task-N170_events.tsv", sep="\t")
stim = events_tsv.loc[events_tsv["trial_type"] == "stimulus", ["sample", "value"]]
events = np.c_[
    stim["sample"].to_numpy(dtype=int),
    np.zeros(len(stim), dtype=int),
    stim["value"].to_numpy(dtype=int),
]

epochs = mne.Epochs(
    raw,
    events,
    event_id=None,
    tmin=EPOCH_TMIN,
    tmax=EPOCH_TMAX,
    baseline=BASELINE,
    preload=True,
    picks="eeg",
    verbose="ERROR",
)

evokeds = average_conditions(
    epochs,
    EVENT_CODE_GROUPS,
    condition_order=CONDITION_ORDER,
)
X, times, info = evokeds_to_tensor(evokeds, CONDITION_ORDER)
epoch_counts = condition_epoch_counts(epochs, EVENT_CODE_GROUPS)


# =============================================================================
# Save prepared ReDisCA bundle
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ready_npz = OUTPUT_DIR / "erpcore_n170_sub001_ready.npz"
ready_info = OUTPUT_DIR / "erpcore_n170_sub001_info.fif"
ready_metadata = OUTPUT_DIR / "erpcore_n170_sub001_ready_metadata.json"

np.savez_compressed(
    ready_npz,
    X=X,
    times=times,
    condition_order=np.asarray(CONDITION_ORDER),
    sfreq=float(info["sfreq"]),
    epoch_counts=np.asarray([epoch_counts[name] for name in CONDITION_ORDER], dtype=int),
    channel_names=np.asarray(info["ch_names"]),
)
mne.io.write_info(
    ready_info,
    info,
    overwrite=True,
    verbose="ERROR",
)

metadata = {
    "source_dataset": "ERP CORE N170 / sub-001",
    "source_eeg_dir": str(eeg_dir),
    "preprocessing": {
        "low_freq_hz": LOW_FREQ,
        "high_freq_hz": HIGH_FREQ,
        "reference": "average",
        "epoch_tmin_s": EPOCH_TMIN,
        "epoch_tmax_s": EPOCH_TMAX,
        "baseline_s": list(BASELINE),
        "picks": "eeg",
    },
    "ica": {
        "run": RUN_ICA,
        "method": ICA_METHOD if RUN_ICA else None,
        "n_components_setting": ICA_N_COMPONENTS,
        "n_components_fit": None if ica is None else int(ica.n_components_),
        "fit_high_pass_hz": ICA_FIT_HIGH_PASS if RUN_ICA else None,
        "eog_threshold": ICA_EOG_THRESHOLD if RUN_ICA else None,
        "eog_channels": EOG_CHANNELS,
        "auto_exclude": auto_ica_exclude,
        "manual_exclude": MANUAL_ICA_EXCLUDE,
        "final_exclude": [] if ica is None else [int(idx) for idx in ica.exclude],
        "eog_components_by_channel": eog_components_by_channel,
        "diagnostics_dir": str(ICA_DIAGNOSTICS_DIR) if RUN_ICA else None,
        "ica_fif": str(ICA_FIF) if RUN_ICA else None,
    },
    "condition_order": CONDITION_ORDER,
    "event_code_groups": {
        key: [min(value), max(value)]
        for key, value in EVENT_CODE_GROUPS.items()
    },
    "epoch_counts": epoch_counts,
    "shape": list(X.shape),
    "sfreq": float(info["sfreq"]),
    "time_start_s": float(times[0]),
    "time_stop_s": float(times[-1]),
}
ready_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(json.dumps(metadata, indent=2))
print(f"\nSaved prepared bundle to {OUTPUT_DIR}")
