from __future__ import annotations

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


def test_synthetic_benchmark_source_metrics_are_bounded():
    module = load_example_module(
        "synthetic_benchmark",
        "examples/synthetic_benchmark.py",
    )
    rng = np.random.default_rng(0)

    dataset = module.make_synthetic_dataset(
        rng,
        snr=0.7,
        n_conditions=5,
        n_sources=2,
        n_channels=16,
        n_timepoints=80,
        n_trials=4,
        rdm_noise_scale=0.2,
    )
    metrics = module.evaluate_source(dataset, source_index=0)

    assert metrics["source_index"] == 0
    assert metrics["component"] >= 0
    assert 0.0 <= metrics["target_rdm_corr"] <= 1.0
    assert 0.0 <= metrics["true_rdm_corr"] <= 1.0
    assert 0.0 <= metrics["pattern_corr"] <= 1.0
    assert "filter_corr" not in metrics
    assert 0.0 < metrics["p_value"] <= 1.0
    assert isinstance(metrics["significant"], bool)


def test_synthetic_benchmark_run_produces_expected_row_counts():
    module = load_example_module(
        "synthetic_benchmark_run",
        "examples/synthetic_benchmark.py",
    )

    df = module.run_synthetic_benchmark(
        n_iter=2,
        snr_levels=[0.4, 0.8],
        n_conditions=5,
        n_sources=3,
        n_channels=12,
        n_timepoints=60,
        n_trials=3,
        rdm_noise_scale=0.2,
        random_state=0,
    )

    assert len(df) == 12
    assert set(df["source_index"]) == {0, 1, 2}
    assert set(df["snr"]) == {0.4, 0.8}


def test_noisy_target_rdm_stays_valid():
    module = load_example_module(
        "synthetic_benchmark_rdm",
        "examples/synthetic_benchmark.py",
    )
    rng = np.random.default_rng(123)
    source_timeseries, _ = module.sample_source_timeseries(
        n_conditions=5,
        n_timepoints=50,
        rng=rng,
    )
    true_rdm = module.source_rdm_from_timeseries(source_timeseries)
    target_rdm = module.make_noisy_target_rdm(true_rdm, rng, noise_scale=0.3)

    assert target_rdm.shape == true_rdm.shape
    assert np.allclose(target_rdm, target_rdm.T)
    assert np.allclose(np.diag(target_rdm), 0.0)
    assert np.all(target_rdm >= 0.0)
