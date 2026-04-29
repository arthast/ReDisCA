"""Data types for the ReDisCA library."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class ReDisCAResult:
    """Result of the ReDisCA algorithm.

    Attributes:
        W: Spatial filters (N, r), columns are filters in sensor space.
        A: Pattern matrix (N, r), columns are topographies.
            In the rank-reduced setting, sensor data are reconstructed as
            ``X_c ≈ A @ U_c`` where ``U_c = W.T @ X_c``.
        lambdas: Eigenvalues (r,), sorted descending. lambda_i = w_i.T @ R_bar_d @ w_i.
        pearson_scores: Pearson correlations (r,) between target RDM and component RDMs.
        component_timeseries: Component time series (C, r, T).
        component_rdms: Component RDMs (r, C, C), symmetric with zero diagonal.
        target_rdm: Original target RDM (C, C).
        n_conditions: Number of conditions C.
        n_channels: Number of channels N.
        n_timepoints: Number of time points T.
        n_components: Number of components r (may be < N if rank reduced).
        p_values: Optional p-values from permutation test (r,).
        significant: Optional boolean mask for significant components (r,).
    """
    W: NDArray[np.floating]
    A: NDArray[np.floating]
    lambdas: NDArray[np.floating]
    pearson_scores: NDArray[np.floating]
    component_timeseries: NDArray[np.floating]
    component_rdms: NDArray[np.floating]
    target_rdm: NDArray[np.floating]
    n_conditions: int
    n_channels: int
    n_timepoints: int
    n_components: int

    # Optional: permutation test outputs
    p_values: Optional[NDArray[np.floating]] = None
    significant: Optional[NDArray[np.bool_]] = None


@dataclass
class PermutationTestResult:
    """Result of the permutation test.

    Attributes:
        p_values: p-value for each component (r,).
        significant: Boolean mask — True where p < alpha (r,).
        null_max_lambdas: Max eigenvalue from each permutation (n_perm,).
            Only populated when ``return_null=True``.
    """
    p_values: NDArray[np.floating]
    significant: NDArray[np.bool_]
    null_max_lambdas: Optional[NDArray[np.floating]] = None


@dataclass
class ValidationResult:
    """Result of input data validation.

    Attributes:
        X: Normalized data array (C, N, T).
        D: Validated RDM (C, C).
        n_conditions: Number of conditions C.
        n_channels: Number of channels N.
        n_timepoints: Number of time points T.
        n_pairs: Number of condition pairs P = C*(C-1)/2.
    """
    X: NDArray[np.floating]
    D: NDArray[np.floating]
    n_conditions: int
    n_channels: int
    n_timepoints: int
    n_pairs: int


@dataclass
class SlidingWindowReDisCAResult:
    """Collection of ReDisCA fits computed over sliding time windows.

    Attributes:
        results: Per-window ReDisCA fits ordered from left to right.
        window_starts: Inclusive window start indices of shape ``(n_windows,)``.
        window_stops: Exclusive window stop indices of shape ``(n_windows,)``.
        window_centers: Window centers either in samples or in the user-provided
            time units, shape ``(n_windows,)``.
        sample_times: Optional original time axis of shape ``(T,)`` used to
            compute ``window_centers``.
    """

    results: list[ReDisCAResult]
    window_starts: NDArray[np.integer]
    window_stops: NDArray[np.integer]
    window_centers: NDArray[np.floating]
    sample_times: Optional[NDArray[np.floating]] = None

    @property
    def n_windows(self) -> int:
        """Number of fitted windows."""
        return len(self.results)

    def component_metric_matrix(
        self,
        attr: str,
        *,
        max_components: int | None = None,
        fill_value: float = np.nan,
    ) -> NDArray[np.floating]:
        """Stack a 1-D per-component attribute across windows.

        Args:
            attr: Name of a ``ReDisCAResult`` attribute such as
                ``"pearson_scores"``, ``"lambdas"``, or ``"p_values"``.
            max_components: Optional maximum number of rows in the output.
                Defaults to the maximum number of components observed across
                all windows.
            fill_value: Value used where a window has fewer components or where
                the requested attribute is ``None``.

        Returns:
            Array of shape ``(max_components, n_windows)``.

        Raises:
            AttributeError: If ``attr`` is not present on ``ReDisCAResult``.
            ValueError: If the attribute is not 1-D when present.
        """
        if max_components is None:
            max_components = max(result.n_components for result in self.results)

        out = np.full(
            (max_components, self.n_windows),
            fill_value,
            dtype=np.float64,
        )

        for window_idx, result in enumerate(self.results):
            values = getattr(result, attr)
            if values is None:
                continue

            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(
                    f"Attribute {attr!r} must be 1-D per result, got shape {arr.shape}."
                )

            n_rows = min(max_components, arr.shape[0])
            out[:n_rows, window_idx] = arr[:n_rows]

        return out
