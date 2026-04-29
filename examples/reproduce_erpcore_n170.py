#!/usr/bin/env python3
"""Reproduce an article-style ReDisCA analysis on ERP CORE N170 subject 001.

This script follows the open-data EEG scenario described in the ReDisCA paper:

1. Download subject ``sub-001`` from the ERP CORE N170 dataset if needed.
2. Build four averaged evoked conditions: face, car, scrambled face,
   scrambled car.
3. Run a sliding-window meaningful-vs-meaningless ReDisCA scan.
4. Run a face-specific fixed-window analysis around the N170 latency.
5. Save plots, summaries, and export bundles.

The implementation is intentionally lightweight and sensor-space only. It aims
to provide a reproducible public-data workflow inside this repository rather
than a byte-for-byte recreation of every figure from the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

os.environ.setdefault("MPLCONFIGDIR", "/tmp/redisca-mpl-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from redisca import export_result, fit_redisca, sliding_window_fit_redisca
from redisca.viz import plot_component_timeseries, plot_top_component_rdms
from redisca.viz_mne import plot_pattern_topomaps


ERP_CORE_SUBJECT_001_URLS = {
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


def build_binary_rdm(
    condition_order: list[str],
    positive_conditions: set[str],
) -> np.ndarray:
    """Create a binary detector-style RDM from named conditions."""
    C = len(condition_order)
    is_positive = np.array(
        [condition in positive_conditions for condition in condition_order],
        dtype=bool,
    )
    rdm = np.zeros((C, C), dtype=float)
    for i in range(C):
        for j in range(i + 1, C):
            rdm[i, j] = float(is_positive[i] != is_positive[j])
            rdm[j, i] = rdm[i, j]
    return rdm


MEANINGFUL_RDM = build_binary_rdm(
    CONDITION_ORDER,
    positive_conditions={"face", "car"},
)
FACE_RDM = build_binary_rdm(
    CONDITION_ORDER,
    positive_conditions={"face"},
)


def download_erpcore_subject_001(data_root: Path) -> Path:
    """Download the minimal ERP CORE N170 files needed for subject 001."""
    eeg_dir = data_root / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)

    for name, url in ERP_CORE_SUBJECT_001_URLS.items():
        destination = eeg_dir / name
        if destination.exists():
            continue
        print(f"Downloading {name} ...")
        urlretrieve(url, destination)

    return eeg_dir


def make_erpcore_montage(eeg_dir: Path) -> mne.channels.DigMontage:
    """Build an MNE montage from the dataset-provided electrode coordinates."""
    electrodes = pd.read_csv(eeg_dir / "sub-001_task-N170_electrodes.tsv", sep="\t")
    ch_pos = {
        row["name"]: np.array([row["x"], row["y"], row["z"]], dtype=float) / 1000.0
        for _, row in electrodes.dropna(subset=["x", "y", "z"]).iterrows()
    }
    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")


def load_erpcore_n170_epochs(
    eeg_dir: Path,
    *,
    low_freq: float = 0.1,
    high_freq: float = 30.0,
    tmin: float = -0.1,
    tmax: float = 0.5,
) -> mne.Epochs:
    """Load and lightly preprocess ERP CORE N170 subject 001."""
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
    raw.set_channel_types(channel_types, verbose="ERROR")
    raw.set_montage(make_erpcore_montage(eeg_dir), on_missing="ignore", verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    raw.filter(low_freq, high_freq, picks="eeg", verbose="ERROR")

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
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, 0.0),
        preload=True,
        picks="eeg",
        verbose="ERROR",
    )
    return epochs


def make_condition_evokeds(epochs: mne.Epochs) -> dict[str, mne.Evoked]:
    """Average the four N170 stimulus conditions."""
    evokeds: dict[str, mne.Evoked] = {}
    event_values = epochs.events[:, 2]

    for condition, codes in EVENT_CODE_GROUPS.items():
        mask = np.isin(event_values, np.array(list(codes), dtype=int))
        evokeds[condition] = epochs[mask].average()

    return evokeds


def evokeds_to_tensor(
    evokeds: dict[str, mne.Evoked],
    condition_order: list[str],
) -> tuple[np.ndarray, np.ndarray, mne.Info]:
    """Convert ordered evokeds into the ``(C, N, T)`` tensor expected by ReDisCA."""
    X = np.stack([evokeds[name].data for name in condition_order], axis=0)
    times = evokeds[condition_order[0]].times.copy()
    info = evokeds[condition_order[0]].info.copy()
    return X, times, info


def ms_to_samples(duration_ms: float, sfreq: float) -> int:
    """Convert milliseconds to the nearest positive sample count."""
    return max(1, int(round(duration_ms * sfreq / 1000.0)))


def plot_window_metric_heatmap(
    matrix: np.ndarray,
    centers_ms: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot a component-by-window heatmap."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[centers_ms[0], centers_ms[-1], 0.5, matrix.shape[0] + 0.5],
    )
    ax.set_xlabel("Window center (ms)")
    ax.set_ylabel("Component")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=colorbar_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def best_window_index(
    scan,
    *,
    component: int = 0,
) -> int:
    """Pick the article-style best window for a component.

    Preference order:
    1. Lowest finite p-value for the requested component.
    2. Highest Pearson score if p-values are unavailable.
    """
    p_values = scan.component_metric_matrix("p_values", max_components=component + 1)
    row = p_values[component]
    if np.isfinite(row).any():
        return int(np.nanargmin(row))

    pearson = scan.component_metric_matrix("pearson_scores", max_components=component + 1)
    return int(np.nanargmax(pearson[component]))


def summarize_result(result) -> dict[str, float | int | list[float]]:
    """Convert a ReDisCA result into a JSON-friendly summary."""
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


def run_meaningfulness_scan(
    X: np.ndarray,
    times: np.ndarray,
    info: mne.Info,
    output_root: Path,
    *,
    window_ms: float,
    step_ms: float,
    n_perm: int,
    random_state: int,
) -> dict[str, object]:
    """Run the meaningful-vs-meaningless sliding-window analysis."""
    sfreq = float(info["sfreq"])
    scan = sliding_window_fit_redisca(
        X,
        MEANINGFUL_RDM,
        window_size=ms_to_samples(window_ms, sfreq),
        step_size=ms_to_samples(step_ms, sfreq),
        times=times,
        permutation_test=True,
        n_perm=n_perm,
        random_state=random_state,
    )

    centers_ms = 1000.0 * scan.window_centers
    pearson = scan.component_metric_matrix("pearson_scores", max_components=6)
    p_values = scan.component_metric_matrix("p_values", max_components=6)

    plot_window_metric_heatmap(
        pearson,
        centers_ms,
        title="ERP CORE N170: meaningful vs meaningless window scan (Pearson r)",
        colorbar_label="Pearson r",
        output_path=output_root / "meaningful_vs_meaningless_pearson.png",
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_window_metric_heatmap(
        p_values,
        centers_ms,
        title="ERP CORE N170: meaningful vs meaningless window scan (p-values)",
        colorbar_label="p-value",
        output_path=output_root / "meaningful_vs_meaningless_pvalues.png",
        cmap="magma_r",
        vmin=0.0,
        vmax=1.0,
    )

    best_idx = best_window_index(scan, component=0)
    best_result = scan.results[best_idx]
    best_dir = output_root / "meaningful_vs_meaningless_best_window"
    best_dir.mkdir(parents=True, exist_ok=True)

    plot_top_component_rdms(
        best_result,
        k=3,
        order="pearson",
        include_target=True,
        save_path=best_dir / "top_component_rdms.png",
    )
    plot_component_timeseries(
        best_result,
        idxs=[0],
        time=times[scan.window_starts[best_idx]:scan.window_stops[best_idx]],
        condition_names=CONDITION_ORDER,
        save_path=best_dir / "component_timeseries.png",
    )
    plot_pattern_topomaps(
        best_result,
        info,
        idxs=[0],
        save_path=best_dir / "pattern_topomap.png",
    )
    export_result(best_result, best_dir / "export")

    return {
        "best_window_index": best_idx,
        "best_window_start_ms": float(1000.0 * times[scan.window_starts[best_idx]]),
        "best_window_stop_ms": float(1000.0 * times[scan.window_stops[best_idx] - 1]),
        "best_window_center_ms": float(centers_ms[best_idx]),
        "best_window_summary": summarize_result(best_result),
    }


def run_face_specific_analysis(
    X: np.ndarray,
    times: np.ndarray,
    info: mne.Info,
    output_root: Path,
    *,
    window_start_s: float,
    window_stop_s: float,
    n_perm: int,
    random_state: int,
) -> dict[str, object]:
    """Run the face-specific fixed-window analysis around the N170 latency."""
    mask = (times >= window_start_s) & (times <= window_stop_s)
    result = fit_redisca(
        X[:, :, mask],
        FACE_RDM,
        permutation_test=True,
        n_perm=n_perm,
        random_state=random_state,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    plot_top_component_rdms(
        result,
        k=3,
        order="pearson",
        include_target=True,
        save_path=output_root / "top_component_rdms.png",
    )
    plot_component_timeseries(
        result,
        idxs=[0],
        time=times[mask],
        condition_names=CONDITION_ORDER,
        save_path=output_root / "component_timeseries.png",
    )
    plot_pattern_topomaps(
        result,
        info,
        idxs=[0],
        save_path=output_root / "pattern_topomap.png",
    )
    export_result(result, output_root / "export")

    return {
        "window_start_ms": float(1000.0 * window_start_s),
        "window_stop_ms": float(1000.0 * window_stop_s),
        "summary": summarize_result(result),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "examples" / "data" / "erpcore_n170",
        help="Directory where ERP CORE subject 001 files are stored/downloaded.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "examples" / "repro_outputs" / "erpcore_n170",
        help="Directory for plots, exports, and summaries.",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=150.0,
        help="Meaningfulness scan window width in milliseconds.",
    )
    parser.add_argument(
        "--step-ms",
        type=float,
        default=25.0,
        help="Meaningfulness scan step size in milliseconds.",
    )
    parser.add_argument(
        "--face-window-start-ms",
        type=float,
        default=150.0,
        help="Face-specific analysis window start in milliseconds.",
    )
    parser.add_argument(
        "--face-window-stop-ms",
        type=float,
        default=250.0,
        help="Face-specific analysis window stop in milliseconds.",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=50,
        help="Number of permutations for significance estimates.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for permutation testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eeg_dir = download_erpcore_subject_001(args.data_root)
    epochs = load_erpcore_n170_epochs(eeg_dir)
    evokeds = make_condition_evokeds(epochs)
    X, times, info = evokeds_to_tensor(evokeds, CONDITION_ORDER)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    meaningful_summary = run_meaningfulness_scan(
        X,
        times,
        info,
        output_root,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        n_perm=args.n_perm,
        random_state=args.random_state,
    )
    face_summary = run_face_specific_analysis(
        X,
        times,
        info,
        output_root / "face_specific",
        window_start_s=args.face_window_start_ms / 1000.0,
        window_stop_s=args.face_window_stop_ms / 1000.0,
        n_perm=args.n_perm,
        random_state=args.random_state,
    )

    summary = {
        "dataset": "ERP CORE N170 / sub-001",
        "conditions": CONDITION_ORDER,
        "meaningful_vs_meaningless": meaningful_summary,
        "face_specific": face_summary,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs to {output_root}")


if __name__ == "__main__":
    main()
