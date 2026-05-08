"""Statistical inference for ReDisCA via permutation testing.

Implements a permutation test for assessing the significance of ReDisCA
components. The null hypothesis is that the correspondence between the
target RDM and the data is absent. The test destroys the correspondence
between condition-pair data matrices and target-RDM pair labels.
"""

from __future__ import annotations

from typing import List
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError as SciPyLinAlgError

from .types import PermutationTestResult
from .validation import validate_permutation_params
from .core import (
    vectorize_upper,
    standardize,
    compute_R_bar_d,
    solve_gep,
    pair_indices
)


def _max_lambda_for_permuted_pairs(
    d_vec_perm: NDArray[np.floating],
    R_list: List[NDArray[np.floating]],
    R_bar: NDArray[np.floating],
    rank: int | str | None,
    tol: float,
) -> float:
    """Compute the maximum eigenvalue for permuted target-RDM pair labels."""
    d_tilde = standardize(d_vec_perm)       # may raise ValueError
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

    The test builds a null distribution of ``lambda_max`` (the largest
    eigenvalue from each permutation). If the requested number of valid
    permutations cannot be collected within the attempt budget, the test falls
    back to the valid permutations collected so far and emits a warning instead
    of failing. For each observed ``lambda_k`` the p-value is

        p_k = (1 + #{b : lambda_max^{(b)} >= lambda_k}) / (n_valid + 1)

    This is a *max-statistic* approach that controls the family-wise error
    rate across components.

    Each surrogate directly shuffles the upper-triangular target-RDM entries
    against the fixed ``R_ij`` list. This destroys the correspondence between
    condition-pair data matrices and target-RDM pair labels, which is the null
    model described for ReDisCA via the SPoC permutation procedure.

    Args:
        R_list: Pre-computed list of R_ij matrices (one per condition pair).
        R_bar: Pre-computed mean difference covariance matrix (N, N).
        target_rdm: Target RDM (C, C).
        observed_lambdas: Eigenvalues from the original fit (r,), sorted
            descending.
        n_perm: Requested number of valid permutations (default 1000).
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
    validate_permutation_params(n_perm=n_perm, alpha=alpha, tol=tol)

    C = target_rdm.shape[0]
    pairs = pair_indices(C)
    d_vec = vectorize_upper(target_rdm, pairs)
    rng = np.random.default_rng(random_state)

    null_max_lambdas = np.empty(n_perm, dtype=np.float64)

    max_attempts = n_perm * 50
    collected = 0
    attempts = 0
    while collected < n_perm:
        if attempts >= max_attempts:
            warn(
                f"Permutation test collected only {collected}/{n_perm} valid "
                f"permutations after {max_attempts} attempts. Returning p-values "
                "based on the collected null distribution. The target RDM may "
                f"be too degenerate for this number of conditions (C={C}).",
                RuntimeWarning,
                stacklevel=2,
            )
            break
        attempts += 1

        try:
            perm = rng.permutation(len(d_vec))
            if np.array_equal(perm, np.arange(len(d_vec))):
                continue

            d_vec_perm = d_vec[perm]
            if np.array_equal(d_vec_perm, d_vec):
                continue

            lam_max = _max_lambda_for_permuted_pairs(
                d_vec_perm, R_list, R_bar, rank, tol
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError, SciPyLinAlgError):
            # Degenerate permutation (e.g. constant upper triangle after
            # permutation) - retry with a new permutation.
            continue

        null_max_lambdas[collected] = lam_max
        collected += 1

    null_max_lambdas = null_max_lambdas[:collected]

    # p-values using the +1 correction
    r = len(observed_lambdas)
    p_values = np.empty(r, dtype=np.float64)
    if collected == 0:
        warn(
            "Permutation test collected no valid permutations. Returning "
            "conservative p-values equal to 1.0 for all components.",
            RuntimeWarning,
            stacklevel=2,
        )
        p_values.fill(1.0)
    else:
        for k in range(r):
            p_values[k] = (
                1 + np.sum(null_max_lambdas >= observed_lambdas[k])
            ) / (collected + 1)

    significant = p_values < alpha

    return PermutationTestResult(
        p_values=p_values,
        significant=significant,
        null_max_lambdas=null_max_lambdas if return_null else None,
    )
