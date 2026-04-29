from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def load_example_module(module_name: str, relative_path: str):
    """Load an example script as a module for smoke testing."""
    root = Path(__file__).resolve().parents[1]
    module_path = root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_like_benchmark_single_source_metrics_are_bounded():
    module = load_example_module(
        "paper_like_benchmark",
        "examples/paper_like_benchmark.py",
    )
    rng = np.random.default_rng(0)

    metrics = module.simulate_single_source_trial(
        rng,
        snr=0.7,
        n_channels=16,
        n_timepoints=80,
        smooth_sigma=6.0,
        rdm_noise_scale=0.2,
    )

    assert metrics["n_conditions"] == 5
    assert metrics["n_components"] >= 1
    assert 0.0 <= metrics["target_rdm_corr"] <= 1.0
    assert 0.0 <= metrics["true_rdm_corr"] <= 1.0
    assert 0.0 <= metrics["pattern_cosine"] <= 1.0


def test_paper_like_benchmark_run_produces_expected_row_counts():
    module = load_example_module(
        "paper_like_benchmark_run",
        "examples/paper_like_benchmark.py",
    )
    args = argparse.Namespace(
        n_iter=2,
        snr_levels=[0.4, 0.8],
        n_channels=12,
        n_timepoints=60,
        smooth_sigma=5.0,
        rdm_noise_scale=0.2,
        random_state=0,
        output_root=Path("unused"),
    )

    single_df, multi_df = module.run_benchmark(args)

    assert len(single_df) == 4
    assert len(multi_df) == 16
    assert set(single_df["n_conditions"]) == {5}
    assert set(multi_df["n_conditions"]) == {6}


def test_noisy_target_rdm_stays_valid():
    module = load_example_module(
        "paper_like_benchmark_rdm",
        "examples/paper_like_benchmark.py",
    )
    rng = np.random.default_rng(123)
    source_timeseries = module.sample_condition_timeseries(
        n_conditions=5,
        n_timepoints=50,
        rng=rng,
        smooth_sigma=4.0,
    )
    true_rdm = module.source_rdm_from_timeseries(source_timeseries)
    target_rdm = module.make_noisy_target_rdm(true_rdm, rng, noise_scale=0.3)

    assert target_rdm.shape == true_rdm.shape
    assert np.allclose(target_rdm, target_rdm.T)
    assert np.allclose(np.diag(target_rdm), 0.0)
    assert np.all(target_rdm >= 0.0)
