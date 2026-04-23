"""Export helpers for ReDisCA results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .types import ReDisCAResult


def _array_payload(result: ReDisCAResult) -> dict[str, np.ndarray]:
    """Build the array payload saved into the NPZ bundle."""
    payload: dict[str, np.ndarray] = {
        "W": result.W,
        "A": result.A,
        "lambdas": result.lambdas,
        "pearson_scores": result.pearson_scores,
        "component_timeseries": result.component_timeseries,
        "component_rdms": result.component_rdms,
        "target_rdm": result.target_rdm,
    }

    if result.p_values is not None:
        payload["p_values"] = result.p_values
    if result.significant is not None:
        payload["significant"] = result.significant.astype(bool, copy=False)

    return payload


def export_result(
    result: ReDisCAResult,
    output_dir: str | Path,
    *,
    compressed: bool = True,
) -> dict[str, Path]:
    """Export a ReDisCA result bundle to disk.

    The export contains:
    - ``redisca_result.npz`` with all dense arrays
    - ``component_scores.csv`` with per-component summary metrics
    - ``target_rdm.csv`` for convenient inspection
    - ``metadata.json`` with scalar metadata and file references

    Args:
        result: Result returned by ``fit_redisca``.
        output_dir: Destination directory.
        compressed: If True, store arrays with ``np.savez_compressed``.

    Returns:
        Mapping from logical artifact names to written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    array_path = output_dir / "redisca_result.npz"
    summary_path = output_dir / "component_scores.csv"
    target_rdm_path = output_dir / "target_rdm.csv"
    metadata_path = output_dir / "metadata.json"

    payload = _array_payload(result)
    save_npz = np.savez_compressed if compressed else np.savez
    save_npz(array_path, **payload)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "component",
                "lambda",
                "pearson_score",
                "p_value",
                "significant",
            ],
        )
        writer.writeheader()
        for idx in range(result.n_components):
            writer.writerow(
                {
                    "component": idx,
                    "lambda": float(result.lambdas[idx]),
                    "pearson_score": float(result.pearson_scores[idx]),
                    "p_value": (
                        ""
                        if result.p_values is None
                        else float(result.p_values[idx])
                    ),
                    "significant": (
                        ""
                        if result.significant is None
                        else bool(result.significant[idx])
                    ),
                }
            )

    np.savetxt(target_rdm_path, result.target_rdm, delimiter=",")

    metadata = {
        "n_conditions": result.n_conditions,
        "n_channels": result.n_channels,
        "n_timepoints": result.n_timepoints,
        "n_components": result.n_components,
        "artifacts": {
            "arrays": array_path.name,
            "component_scores": summary_path.name,
            "target_rdm": target_rdm_path.name,
        },
        "arrays_in_npz": sorted(payload.keys()),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "arrays": array_path,
        "component_scores": summary_path,
        "target_rdm": target_rdm_path,
        "metadata": metadata_path,
    }
