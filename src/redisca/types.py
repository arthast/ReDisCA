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
        A: Pattern matrix (r, N), rows are topographies (A @ W ≈ I).
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
