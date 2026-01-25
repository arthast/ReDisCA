"""Computational core of the ReDisCA algorithm.

Contains basic functions for computing difference matrices
and preparing data for the generalized eigenvalue problem.
"""

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import zscore


def pair_indices(C: int) -> List[Tuple[int, int]]:
    """Generate list of all condition pairs (i, j) where i < j.

    Args:
        C: Number of conditions.

    Returns:
        List of tuples (i, j).

    Example:
        >>> pair_indices(3)
        [(0, 1), (0, 2), (1, 2)]
    """
    return [(i, j) for i in range(C) for j in range(i + 1, C)]


def compute_R_ij(
    X_i: NDArray[np.floating],
    X_j: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Compute difference covariance matrix for a pair of conditions.

    R_ij = (X_i - X_j) @ (X_i - X_j).T

    Args:
        X_i: Evoked data for condition i, shape (N, T).
        X_j: Evoked data for condition j, shape (N, T).

    Returns:
        Matrix R_ij of shape (N, N).
    """
    delta_X = X_i - X_j
    return delta_X @ delta_X.T


def compute_all_R_ij(
    X: NDArray[np.floating],
    pairs: List[Tuple[int, int]]
) -> List[NDArray[np.floating]]:
    """Compute R_ij for all condition pairs.

    Args:
        X: Data of shape (C, N, T).
        pairs: List of pairs (i, j).

    Returns:
        List of R_ij matrices.
    """
    return [compute_R_ij(X[i], X[j]) for i, j in pairs]


def vectorize_upper(D: NDArray[np.floating]) -> NDArray[np.floating]:
    """Extract upper triangle of matrix into a vector.

    Args:
        D: Square matrix (C, C).

    Returns:
        Vector of length C*(C-1)/2.

    Example:
        >>> D = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        >>> vectorize_upper(D)
        array([1, 2, 3])
    """
    C = D.shape[0]
    indices = np.triu_indices(C, k=1)
    return D[indices]


def standardize(d_vec: NDArray[np.floating]) -> NDArray[np.floating]:
    """Z-standardization of a vector.

    Args:
        d_vec: Input vector.

    Returns:
        Standardized vector (mean=0, std=1).

    Raises:
        ValueError: If std=0 (uninformative RDM).
    """
    std = np.std(d_vec, ddof=0)
    if std < 1e-10:
        raise ValueError(
            "Standard deviation of target RDM is close to zero. "
            "RDM is uninformative (all elements are nearly equal)."
        )
    return zscore(d_vec, ddof=0)


def compute_R_bar(R_list: List[NDArray[np.floating]]) -> NDArray[np.floating]:
    """Compute mean matrix R_bar.

    R_bar = (1/P) * sum(R_ij), where P is the number of pairs.

    Args:
        R_list: List of R_ij matrices.

    Returns:
        Mean matrix R_bar of shape (N, N).
    """
    P = len(R_list)
    R_sum = np.sum(R_list, axis=0)
    return R_sum / P


def compute_R_bar_d(
    R_list: List[NDArray[np.floating]],
    R_bar: NDArray[np.floating],
    d_tilde: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Compute target weighted matrix R_bar_d.

    R_bar_d = sum(d_tilde[k] * (R_ij - R_bar))

    Args:
        R_list: List of R_ij matrices.
        R_bar: Mean matrix.
        d_tilde: Standardized target RDM vector.

    Returns:
        Matrix R_bar_d of shape (N, N).
    """
    N = R_bar.shape[0]
    R_bar_d = np.zeros((N, N), dtype=np.float64)

    for k, R_ij in enumerate(R_list):
        R_bar_d += d_tilde[k] * (R_ij - R_bar)

    return R_bar_d
