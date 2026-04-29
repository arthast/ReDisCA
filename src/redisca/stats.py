"""Statistical inference for ReDisCA via permutation testing.

Implements a permutation test for assessing the significance of ReDisCA
components. The null hypothesis is that the correspondence between the
target RDM and the data is absent. Condition labels are permuted to build
a null distribution of the max eigenvalue (lambda_max) across permutations.
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError as SciPyLinAlgError

from .types import PermutationTestResult
from .core import (
    vectorize_upper,
    standardize,
    compute_R_bar_d,
    solve_gep,
    pair_indices
)


def _validate_permutation_params(
    n_perm: int,
    alpha: float,
    tol: float,
) -> None:
    """Validate statistical inference parameters."""
    if isinstance(n_perm, (bool, np.bool_)) or not isinstance(n_perm, (int, np.integer)):
        raise TypeError(
            "n_perm must be a positive integer, got "
            f"{type(n_perm).__name__}"
        )
    if int(n_perm) < 1:
        raise ValueError(f"n_perm must be >= 1, got {n_perm}")

    if not np.isfinite(alpha) or not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}")

    if not np.isfinite(tol) or float(tol) <= 0.0:
        raise ValueError(f"tol must be a positive finite number, got {tol}")



def _permute_rdm(
    D: NDArray[np.floating],
    perm: NDArray[np.intp],
) -> NDArray[np.floating]:
    """Permute rows and columns of an RDM according to *perm*.

    D^{(π)}_{ij} = D_{π(i), π(j)}

    Symmetry and zero diagonal are preserved by construction.
    """
    return D[np.ix_(perm, perm)]


def _max_lambda_for_permuted_rdm(
    D_perm: NDArray[np.floating],
    R_list: List[NDArray[np.floating]],
    R_bar: NDArray[np.floating],
    pairs: list[tuple[int, int]],
    rank: int | str | None,
    tol: float,
) -> float:
    """Compute the maximum eigenvalue for a permuted target RDM.

    Builds d_tilde from the permuted RDM, constructs R_bar_d, solves the
    GEP and returns lambda_max.

    Returns:
        The largest eigenvalue, or raises on failure.
    """
    d_vec = vectorize_upper(D_perm, pairs)
    d_tilde = standardize(d_vec)            # may raise ValueError
    R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)
    _, lambdas = solve_gep(R_bar_d, R_bar, rank=rank, tol=tol)
    return float(lambdas[0])                # lambdas sorted descending


def permutation_test_redisca(
    R_list: List[NDArray[np.floating]],
    R_bar: NDArray[np.floating],
    target_rdm: NDArray[np.floating],
    observed_lambdas: NDArray[np.floating],
    n_perm: int = 1000,
    rank: int | str | None = "auto",
    tol: float = 1e-10,
    alpha: float = 0.05,
    random_state: int | None = None,
    return_null: bool = False,
) -> PermutationTestResult:
    """Run a permutation test on ReDisCA components.

    The test permutes condition labels to build a null distribution of
    ``lambda_max`` (the largest eigenvalue from each permutation).  For
    each observed ``lambda_k`` the p-value is

        p_k = (1 + #{b : lambda_max^{(b)} >= lambda_k}) / (n_perm + 1)

    This is a *max-statistic* approach that controls the family-wise error
    rate across components.

    Args:
        R_list: Pre-computed list of R_ij matrices (one per condition pair).
        R_bar: Pre-computed mean difference covariance matrix (N, N).
        target_rdm: Target RDM (C, C).
        observed_lambdas: Eigenvalues from the original fit (r,), sorted
            descending.
        n_perm: Number of permutations (default 1000).
        rank: Rank parameter forwarded to ``solve_gep``.
        tol: Tolerance forwarded to ``solve_gep``.
        alpha: Significance level (default 0.05).
        random_state: Seed for reproducibility.
        return_null: If True, include the full null distribution in the
            result.

    Returns:
        PermutationTestResult with p-values, significance mask, and
        optionally the null distribution.
    """
    _validate_permutation_params(n_perm=n_perm, alpha=alpha, tol=tol)

    C = target_rdm.shape[0]
    pairs = pair_indices(C)
    rng = np.random.default_rng(random_state)

    null_max_lambdas = np.empty(n_perm, dtype=np.float64)

    max_attempts = n_perm * 50
    collected = 0
    attempts = 0
    while collected < n_perm:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Permutation test failed: only {collected}/{n_perm} valid "
                f"permutations after {max_attempts} attempts. The target RDM "
                f"may be too degenerate for this number of conditions (C={C})."
            )
        attempts += 1
        perm = rng.permutation(C)

        # Skip the identity permutation
        if np.array_equal(perm, np.arange(C)):
            continue

        D_perm = _permute_rdm(target_rdm, perm)

        try:
            lam_max = _max_lambda_for_permuted_rdm(
                D_perm, R_list, R_bar, pairs, rank, tol
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError, SciPyLinAlgError):
            # Degenerate permutation (e.g. constant upper triangle after
            # permutation) — retry with a new permutation so that we
            # always collect exactly n_perm valid values.
            continue

        null_max_lambdas[collected] = lam_max
        collected += 1

    # p-values using the +1 correction
    r = len(observed_lambdas)
    p_values = np.empty(r, dtype=np.float64)
    for k in range(r):
        p_values[k] = (1 + np.sum(null_max_lambdas >= observed_lambdas[k])) / (n_perm + 1)

    significant = p_values < alpha

    return PermutationTestResult(
        p_values=p_values,
        significant=significant,
        null_max_lambdas=null_max_lambdas if return_null else None,
    )

