"""Input data validation for ReDisCA."""

from typing import List, Union
import warnings

import numpy as np
from numpy.typing import NDArray

from .types import ValidationResult


def validate_inputs(
        X: Union[NDArray[np.floating], List[NDArray[np.floating]]],
        D: NDArray[np.floating],
) -> ValidationResult:
    """Validate and normalize input data.

    Args:
        X: Evoked data. Either array (C, N, T) or list of C matrices (N, T).
        D: Theoretical RDM, matrix (C, C).

    Returns:
        ValidationResult with normalized data.

    Raises:
        ValueError: If input data is invalid.
        TypeError: If data type is incorrect.
    """
    # Convert X to numpy array if it's a list
    X_normalized = _normalize_X(X)
    D_normalized = np.asarray(D, dtype=np.float64)

    # Get dimensions
    C, N, T = X_normalized.shape

    # Validate X
    _validate_X(X_normalized, C, N, T)

    # Validate D
    _validate_D(D_normalized, C)

    # Warning for small number of conditions
    if C < 4:
        warnings.warn(
            f"Number of conditions C={C} < 4. Results may be uninformative.",
            UserWarning
        )

    n_pairs = C * (C - 1) // 2

    return ValidationResult(
        X=X_normalized,
        D=D_normalized,
        n_conditions=C,
        n_channels=N,
        n_timepoints=T,
        n_pairs=n_pairs,
    )


def _normalize_X(
        X: Union[NDArray[np.floating], List[NDArray[np.floating]]]
) -> NDArray[np.floating]:
    """Convert X to shape (C, N, T)."""
    if isinstance(X, list):
        if len(X) == 0:
            raise ValueError("X cannot be an empty list.")

        # Check that all elements are arrays of the same shape
        shapes = [np.asarray(x).shape for x in X]
        if len(set(shapes)) != 1:
            raise ValueError(
                f"All matrices in list X must have the same shape. "
                f"Got shapes: {shapes}"
            )

        X_normalized = np.stack([np.asarray(x, dtype=np.float64) for x in X], axis=0)
    else:
        X_normalized = np.asarray(X, dtype=np.float64)

    if X_normalized.ndim != 3:
        raise ValueError(
            f"X must have 3 dimensions (C, N, T). Got: {X_normalized.ndim}"
        )

    return X_normalized


def _validate_X(X: NDArray[np.floating], C: int, N: int, T: int) -> None:
    """Validate array X."""
    if C < 3:
        raise ValueError(
            f"Need at least 3 conditions for meaningful analysis. Got: {C}"
        )

    if N < 1:
        raise ValueError(f"Number of channels N must be >= 1. Got: {N}")

    if T < 1:
        raise ValueError(f"Number of time points T must be >= 1. Got: {T}")

    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values.")

    if np.any(np.isinf(X)):
        raise ValueError("X contains Inf values.")


def _validate_D(D: NDArray[np.floating], C: int) -> None:
    """Validate RDM matrix."""
    if D.ndim != 2:
        raise ValueError(f"D must be a 2D matrix. Got: {D.ndim}D")

    if D.shape[0] != D.shape[1]:
        raise ValueError(
            f"D must be square. Got shape: {D.shape}"
        )

    if D.shape[0] != C:
        raise ValueError(
            f"Size of D ({D.shape[0]}) does not match number of conditions C ({C})."
        )

    if np.any(np.isnan(D)):
        raise ValueError("D contains NaN values.")

    if np.any(np.isinf(D)):
        raise ValueError("D contains Inf values.")

    # Check symmetry
    if not np.allclose(D, D.T):
        raise ValueError("D must be a symmetric matrix.")

    # Check diagonal is zero
    if not np.allclose(np.diag(D), 0):
        raise ValueError("D diagonal must be zero (self-dissimilarity = 0).")

    # Check non-negativity
    if np.any(D < 0):
        raise ValueError("D must be non-negative (distances >= 0).")

