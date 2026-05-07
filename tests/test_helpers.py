import types

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from redisca import (
    average_conditions as public_average_conditions,
    binary_rdm,
    best_window_index,
    condition_epoch_counts as public_condition_epoch_counts,
    evokeds_to_tensor as public_evokeds_to_tensor,
    fit_redisca_evokeds as public_fit_redisca_evokeds,
    ms_to_samples,
    sliding_window_fit_redisca_evokeds as public_sliding_window_fit_redisca_evokeds,
)
from redisca.mne_utils import (
    average_conditions,
    condition_epoch_counts,
    evokeds_to_tensor,
    fit_redisca_evokeds,
    sliding_window_fit_redisca_evokeds,
)
from redisca.report import (
    save_evoked_overview,
    save_sliding_window_report,
    summarize_fixed_window_result,
    summarize_sliding_window_scan,
)
from redisca.types import ReDisCAResult, SlidingWindowReDisCAResult
from redisca.windowed import sliding_window_fit_redisca_ms


class DummyEpochSlice:
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs

    def average(self):
        return f"average-{self.n_epochs}"


class DummyEpochs:
    def __init__(self, event_codes):
        self.events = np.c_[
            np.arange(len(event_codes)),
            np.zeros(len(event_codes), dtype=int),
            np.asarray(event_codes, dtype=int),
        ]

    def __getitem__(self, mask):
        return DummyEpochSlice(int(np.asarray(mask).sum()))


def make_dummy_evokeds():
    rng = np.random.default_rng(0)
    condition_order = ["a", "b", "c", "d"]
    times = np.linspace(-0.1, 0.2, 31)
    info = {"sfreq": 100.0}
    ch_names = [f"E{i}" for i in range(6)]
    evokeds = {
        name: types.SimpleNamespace(
            data=rng.standard_normal((6, times.size)) + 0.1 * idx,
            times=times,
            info=info,
            ch_names=ch_names,
        )
        for idx, name in enumerate(condition_order)
    }
    return evokeds, condition_order


def make_dummy_result(*, lambda_value=1.0, pearson=0.5, p_value=None):
    p_values = None if p_value is None else np.array([p_value], dtype=float)
    significant = None if p_value is None else np.array([p_value < 0.05])
    return ReDisCAResult(
        W=np.ones((2, 1)),
        A=np.ones((2, 1)),
        lambdas=np.array([lambda_value], dtype=float),
        pearson_scores=np.array([pearson], dtype=float),
        component_timeseries=np.zeros((2, 1, 3)),
        component_rdms=np.zeros((1, 2, 2)),
        target_rdm=np.zeros((2, 2)),
        n_conditions=2,
        n_channels=2,
        n_timepoints=3,
        n_components=1,
        p_values=p_values,
        significant=significant,
    )


class TestBinaryRdm:
    def test_binary_rdm_groups_conditions(self):
        rdm = binary_rdm(
            ["face", "car", "scrambled_face", "scrambled_car"],
            positive_conditions={"face", "car"},
        )
        expected = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=float,
        )
        assert_array_equal(rdm, expected)

    def test_binary_rdm_rejects_unknown_positive_condition(self):
        with pytest.raises(ValueError, match="absent from condition_order"):
            binary_rdm(["a", "b"], positive_conditions={"c"})


class TestMneUtils:
    def test_common_helpers_are_available_from_package_root(self):
        assert public_average_conditions is average_conditions
        assert public_condition_epoch_counts is condition_epoch_counts
        assert public_evokeds_to_tensor is evokeds_to_tensor
        assert public_fit_redisca_evokeds is fit_redisca_evokeds
        assert public_sliding_window_fit_redisca_evokeds is sliding_window_fit_redisca_evokeds

    def test_condition_epoch_counts(self):
        epochs = DummyEpochs([1, 2, 41, 42, 101])
        counts = condition_epoch_counts(
            epochs,
            {
                "face": range(1, 3),
                "car": range(41, 43),
                "scrambled": [101],
            },
        )
        assert counts == {"face": 2, "car": 2, "scrambled": 1}

    def test_average_conditions_uses_explicit_order(self):
        epochs = DummyEpochs([1, 2, 41, 42, 101])
        evokeds = average_conditions(
            epochs,
            {
                "face": range(1, 3),
                "car": range(41, 43),
                "scrambled": [101],
            },
            condition_order=["scrambled", "face"],
        )
        assert list(evokeds) == ["scrambled", "face"]
        assert evokeds == {"scrambled": "average-1", "face": "average-2"}

    def test_evokeds_to_tensor_preserves_order(self):
        times = np.array([0.0, 0.1, 0.2])
        info = {"sfreq": 10.0}
        evokeds = {
            "b": types.SimpleNamespace(
                data=np.full((2, 3), 2.0),
                times=times,
                info=info,
                ch_names=["C1", "C2"],
            ),
            "a": types.SimpleNamespace(
                data=np.full((2, 3), 1.0),
                times=times,
                info=info,
                ch_names=["C1", "C2"],
            ),
        }

        X, out_times, out_info = evokeds_to_tensor(evokeds, ["a", "b"])

        assert X.shape == (2, 2, 3)
        assert_allclose(X[0], 1.0)
        assert_allclose(X[1], 2.0)
        assert_allclose(out_times, times)
        assert out_info == info

    def test_save_evoked_overview_writes_single_readable_file(self, tmp_path):
        times = np.array([-0.1, 0.0, 0.1])
        evokeds = {
            "a": types.SimpleNamespace(data=np.ones((2, 3)) * 1e-6, times=times),
            "b": types.SimpleNamespace(data=np.ones((2, 3)) * 2e-6, times=times),
        }

        output_path = tmp_path / "overview.png"
        save_evoked_overview(evokeds, output_path, condition_order=["a", "b"])

        assert output_path.exists()
        assert not (tmp_path / "overview_0.png").exists()

    def test_fit_redisca_evokeds_selects_time_window(self):
        evokeds, condition_order = make_dummy_evokeds()
        target_rdm = binary_rdm(condition_order, {"a", "b"})

        analysis = fit_redisca_evokeds(
            evokeds,
            target_rdm,
            condition_order=condition_order,
            tmin=0.0,
            tmax=0.1,
        )

        assert analysis.condition_order == condition_order
        assert analysis.result.n_timepoints == analysis.times.size
        assert analysis.times[0] >= -1e-12
        assert analysis.times[-1] <= 0.1 + 1e-12
        assert analysis.n_components == analysis.result.n_components

    def test_sliding_window_fit_redisca_evokeds_uses_mne_time_metadata(self):
        evokeds, condition_order = make_dummy_evokeds()
        target_rdm = binary_rdm(condition_order, {"a", "b"})

        analysis = sliding_window_fit_redisca_evokeds(
            evokeds,
            target_rdm,
            condition_order=condition_order,
            window_ms=50.0,
            step_ms=50.0,
            tmin=0.0,
            tmax=0.2,
        )

        first_nonnegative = int(np.flatnonzero(analysis.times >= 0.0)[0])
        assert analysis.window_starts[0] == first_nonnegative
        assert analysis.window_stops[0] == first_nonnegative + 5
        assert analysis.n_windows == analysis.scan.n_windows
        assert analysis.best_result().n_conditions == len(condition_order)
        assert analysis.best_window_times().ndim == 1


class TestWindowHelpers:
    def test_ms_to_samples(self):
        assert ms_to_samples(150.0, 1024.0) == 154

    def test_sliding_window_fit_redisca_ms_converts_window_units(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 8, 50))
        target_rdm = binary_rdm(["a", "b", "c", "d"], {"a", "b"})

        scan = sliding_window_fit_redisca_ms(
            X,
            target_rdm,
            sfreq=100.0,
            window_ms=100.0,
            step_ms=50.0,
        )

        assert scan.window_starts.tolist() == [0, 5, 10, 15, 20, 25, 30, 35, 40]
        assert scan.window_stops.tolist() == [10, 15, 20, 25, 30, 35, 40, 45, 50]

    def test_save_sliding_window_report_writes_standard_artifacts(self, tmp_path):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 8, 50))
        target_rdm = binary_rdm(["a", "b", "c", "d"], {"a", "b"})
        scan = sliding_window_fit_redisca_ms(
            X,
            target_rdm,
            sfreq=100.0,
            window_ms=100.0,
            step_ms=50.0,
        )

        artifacts = save_sliding_window_report(
            scan,
            tmp_path,
            prefix="demo",
            center_scale=1000.0,
            highlight_window_index=0,
        )

        assert set(artifacts) == {
            "pearson_heatmap",
            "pvalue_heatmap",
            "lambda_heatmap",
            "window_metrics",
        }
        for file_name in artifacts.values():
            assert (tmp_path / file_name).exists()

    def test_best_window_index_prefers_lowest_p_value(self):
        class DummyScan:
            def component_metric_matrix(self, name, max_components):
                if name == "p_values":
                    return np.array([[0.4, 0.01, 0.2]])
                if name == "pearson_scores":
                    return np.array([[0.9, 0.1, 0.2]])
                raise AssertionError(name)

        assert best_window_index(DummyScan()) == 1

    def test_report_summarizes_sliding_window_scan(self):
        scan = SlidingWindowReDisCAResult(
            results=[
                make_dummy_result(lambda_value=1.0, pearson=0.3, p_value=0.5),
                make_dummy_result(lambda_value=2.0, pearson=0.8, p_value=0.01),
                make_dummy_result(lambda_value=1.5, pearson=0.7, p_value=0.03),
                make_dummy_result(lambda_value=3.0, pearson=0.9, p_value=0.2),
            ],
            window_starts=np.array([0, 10, 20, 30]),
            window_stops=np.array([10, 20, 30, 40]),
            window_centers=np.array([0.05, 0.15, 0.25, 0.35]),
            sample_times=np.linspace(0.0, 0.39, 40),
        )

        summary = summarize_sliding_window_scan(
            scan,
            alpha=0.05,
            time_scale=1000.0,
            time_unit="ms",
            reference_center=250.0,
            reference_label="window_around_250_ms",
        )

        assert summary["best_by_p_value"]["window_index"] == 1
        assert summary["best_by_pearson"]["window_index"] == 3
        assert summary["has_significant_segment"] is True
        assert summary["significant_segments"][0]["n_windows"] == 2
        assert summary["window_around_250_ms"]["window_index"] == 2

    def test_report_summarizes_fixed_window_result(self):
        result = make_dummy_result(lambda_value=2.0, pearson=0.9, p_value=0.03)

        summary = summarize_fixed_window_result(
            result,
            times=np.array([0.15, 0.20, 0.25]),
            alpha=0.05,
            time_scale=1000.0,
            time_unit="ms",
        )

        assert summary["window_start_ms"] == 150.0
        assert summary["window_stop_ms"] == 250.0
        assert summary["has_significant_component"] is True
        assert summary["best_by_p_value"]["significant"] is True
