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
    symmetrize_matrix
)
from .stats import permutation_test_redisca


def fit_redisca(
        X: Union[NDArray[np.floating], List[NDArray[np.floating]]],
        target_rdm: NDArray[np.floating],
        rank: int | str | None = "auto",
        tol: float = 1e-10,
        permutation_test: bool = False,
        n_perm: int = 1000,
        alpha: float = 0.05,
        random_state: int | None = None,
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
            - "auto": automatically use the effective numerical rank, i.e. the
              number of eigenvalues of R_bar greater than tol
            - int: use specified rank
            - None: do not impose a user-specified rank cap; keep all numerically
              valid directions, i.e. all eigenvalues of R_bar greater than tol.
              This may return fewer than N components if R_bar is rank-deficient
              or numerically singular.
        tol: Threshold for treating eigenvalues of R_bar as numerically positive.
             Used to determine the effective numerical rank.
        permutation_test: If True, run a permutation test to assess the
            significance of each component.
        n_perm: Number of permutations (only used when permutation_test=True).
        alpha: Significance level (only used when permutation_test=True).
        random_state: Random seed for the permutation test.

    Returns:
        ReDisCAResult containing filters, patterns, eigenvalues,
        component time series, component RDMs, and Pearson scores.
        Components are sorted by eigenvalues (descending).
        When ``permutation_test=True`` the result also contains
        ``p_values`` and ``significant``.

    Raises:
        ValueError: If input validation fails.
        RuntimeError: If the generalized eigenvalue problem becomes
            numerically unstable or filters cannot be properly normalized.
    """
    validated = validate_inputs(X, target_rdm)
    X = validated.X
    target_rdm = validated.D

    C, N, T = X.shape

    pairs = pair_indices(C)
    R_list = compute_all_R_ij(X, pairs)

    d_vec = vectorize_upper(target_rdm, pairs)
    d_tilde = standardize(d_vec)

    R_bar = compute_R_bar(R_list)
    R_bar = symmetrize_matrix(R_bar, name="R_bar")

    R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)
    R_bar_d = symmetrize_matrix(R_bar_d, name="R_bar_d")

    W, lambdas = solve_gep(R_bar_d, R_bar, rank=rank, tol=tol)
    A = compute_patterns(W, R_bar)
    component_timeseries = compute_component_timeseries(W, X)
    component_rdms = compute_component_rdms(W, R_list, pairs, C)
    pearson_scores = compute_pearson_scores(target_rdm, pairs, component_rdms)

    r = W.shape[1]

    p_values = None
    significant = None

    if permutation_test:
        perm_result = permutation_test_redisca(
            R_list=R_list,
            R_bar=R_bar,
            target_rdm=target_rdm,
            observed_lambdas=lambdas,
            n_perm=n_perm,
            rank=rank,
            tol=tol,
            alpha=alpha,
            random_state=random_state,
        )
        p_values = perm_result.p_values
        significant = perm_result.significant

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
        p_values=p_values,
        significant=significant,
    )
