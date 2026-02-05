"""Main entry point for ReDisCA algorithm."""

from typing import Union, List

import numpy as np
from numpy.typing import NDArray

from .types import ReDisCAResult
from .validation import validate_inputs
from .core import (
    pair_indices,
    compute_all_R_ij,
    vectorize_upper,
    standardize,
    compute_R_bar,
    compute_R_bar_d,
    solve_gep,
    compute_patterns,
    compute_component_timeseries,
    compute_component_rdms,
    compute_pearson_scores,
)


def fit_redisca(
    X: Union[NDArray[np.floating], List[NDArray[np.floating]]],
    target_rdm: NDArray[np.floating],
    rank: int | str | None = "auto",
    tol: float = 1e-10,
) -> ReDisCAResult:
    """Fit ReDisCA model to data.

    Main entry point for the ReDisCA algorithm. Finds spatial filters
    whose component RDMs maximally correlate with the target RDM.

    If the generalized eigenvalue problem is not full rank,
    the procedure is performed in a lower dimensional principal space and
    the obtained topographies are transformed back to the original sensor space.

    Args:
        X: Evoked data. Either:
            - array of shape (C, N, T)
            - list of C matrices of shape (N, T)
            where C = number of conditions (>= 3),
            N = number of channels, T = number of time points.
        target_rdm: Theoretical RDM of shape (C, C).
        rank: Number of principal components to retain.
            - "auto": automatically determine rank from eigenvalues > tol
            - int: use specified rank
            - None: use full rank (no dimensionality reduction)
        tol: Threshold for determining rank when rank="auto".

    Returns:
        ReDisCAResult containing filters, patterns, eigenvalues,
        component time series, component RDMs, and Pearson scores.
        Components are sorted by eigenvalues (descending).

    Raises:
        ValueError: If input validation fails.
    """
    validated = validate_inputs(X, target_rdm)
    X = validated.X
    target_rdm = validated.D

    C, N, T = X.shape

    pairs = pair_indices(C)
    R_list = compute_all_R_ij(X, pairs)

    d_vec = vectorize_upper(target_rdm)
    d_tilde = standardize(d_vec)

    R_bar = compute_R_bar(R_list)
    R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

    W, lambdas = solve_gep(R_bar_d, R_bar, rank=rank, tol=tol)
    A = compute_patterns(W, R_bar)
    component_timeseries = compute_component_timeseries(W, X)
    component_rdms = compute_component_rdms(W, R_list, pairs, C)
    pearson_scores = compute_pearson_scores(target_rdm, component_rdms)

    r = W.shape[1]

    return ReDisCAResult(
        W=W,
        A=A,
        lambdas=lambdas,
        component_timeseries=component_timeseries,
        component_rdms=component_rdms,
        pearson_scores=pearson_scores,
        target_rdm=target_rdm,
        n_conditions=C,
        n_channels=N,
        n_timepoints=T,
        n_components=r,
    )

