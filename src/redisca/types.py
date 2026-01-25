"""Data types for the ReDisCA library."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class ReDisCAResult:
    """Result of the ReDisCA algorithm.

    Attributes:
        W: Spatial filters matrix (N, N).
        A: Patterns/topographies matrix (N, N).
        lambdas: Eigenvalues (N,).
        component_timeseries: Component time series (C, N, T).
        component_rdms: Observed RDMs per component (N, C, C).
        spearman_scores: Spearman correlation with target RDM (N,).
        target_rdm: Original theoretical RDM (C, C).
        n_conditions: Number of conditions C.
        n_channels: Number of channels N.
        n_timepoints: Number of time points T.
    """
    W: NDArray[np.floating]
    A: NDArray[np.floating]
    lambdas: NDArray[np.floating]
    component_timeseries: NDArray[np.floating]
    component_rdms: NDArray[np.floating]
    spearman_scores: NDArray[np.floating]
    target_rdm: NDArray[np.floating]
    n_conditions: int
    n_channels: int
    n_timepoints: int

    # Optional fields for permutation test
    p_values: Optional[NDArray[np.floating]] = None
    significant_components: Optional[NDArray[np.bool_]] = None


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
