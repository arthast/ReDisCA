from __future__ import annotations

import numpy as np
import pytest

from redisca import sliding_window_fit_redisca
from redisca.types import SlidingWindowReDisCAResult


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((4, 8, 60))
    target_rdm = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=float,
    )
    return X, target_rdm


class TestSlidingWindowFitRedisca:
    def test_returns_expected_type_and_window_count(self, simple_data):
        X, target_rdm = simple_data

        result = sliding_window_fit_redisca(
            X,
            target_rdm,
            window_size=15,
            step_size=10,
            rank=3,
        )

        assert isinstance(result, SlidingWindowReDisCAResult)
        assert result.n_windows == 5
        assert result.window_starts.tolist() == [0, 10, 20, 30, 40]
        assert result.window_stops.tolist() == [15, 25, 35, 45, 55]
        assert len(result.results) == result.n_windows
        assert all(item.n_timepoints == 15 for item in result.results)

    def test_window_centers_follow_time_axis(self, simple_data):
        X, target_rdm = simple_data
        times = np.linspace(-0.1, 0.49, X.shape[-1])

        result = sliding_window_fit_redisca(
            X,
            target_rdm,
            window_size=20,
            step_size=20,
            times=times,
            rank=2,
        )

        expected = np.array(
            [
                0.5 * (times[0] + times[19]),
                0.5 * (times[20] + times[39]),
                0.5 * (times[40] + times[59]),
            ]
        )
        np.testing.assert_allclose(result.window_centers, expected)

    def test_component_metric_matrix_stacks_attributes(self, simple_data):
        X, target_rdm = simple_data

        result = sliding_window_fit_redisca(
            X,
            target_rdm,
            window_size=20,
            step_size=20,
            rank=2,
            permutation_test=True,
            n_perm=10,
            random_state=0,
        )

        pearson = result.component_metric_matrix("pearson_scores")
        p_values = result.component_metric_matrix("p_values")

        assert pearson.shape == (2, 3)
        assert p_values.shape == (2, 3)
        assert np.all(np.isfinite(pearson))
        assert np.all(np.isfinite(p_values))

    def test_component_metric_matrix_fills_missing_optional_metrics(self, simple_data):
        X, target_rdm = simple_data

        result = sliding_window_fit_redisca(
            X,
            target_rdm,
            window_size=20,
            step_size=20,
            rank=2,
        )

        matrix = result.component_metric_matrix("p_values")
        assert matrix.shape == (2, 3)
        assert np.isnan(matrix).all()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"window_size": 0}, "window_size must be >= 1"),
            ({"step_size": 0}, "step_size must be >= 1"),
            ({"start": -1}, "start must be >= 0"),
            ({"start": 30, "stop": 20}, "Require start < stop"),
            ({"window_size": 100}, "does not fit"),
        ],
    )
    def test_invalid_window_parameters_raise(self, simple_data, kwargs, match):
        X, target_rdm = simple_data
        params = {"window_size": 10, "step_size": 5}
        params.update(kwargs)

        with pytest.raises((TypeError, ValueError), match=match):
            sliding_window_fit_redisca(
                X,
                target_rdm,
                **params,
            )

    def test_invalid_times_shape_raises(self, simple_data):
        X, target_rdm = simple_data
        with pytest.raises(ValueError, match="times must have shape"):
            sliding_window_fit_redisca(
                X,
                target_rdm,
                window_size=10,
                times=np.arange(10),
            )
