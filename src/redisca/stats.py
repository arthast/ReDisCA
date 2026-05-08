"""Statistical inference for ReDisCA via permutation testing.

The permutation test destroys the correspondence between condition-pair data
matrices and target-RDM entries by reshuffling the upper triangle of the target
RDM. Component p-values are computed component-wise, without max-statistic
correction.
"""

from __future__ import annotations

from warnings import warn

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError as SciPyLinAlgError

from .core import (
    compute_R_bar_d,
    pair_indices,
    solve_gep,
    standardize,
    vectorize_upper,
)
from .types import PermutationTestResult
from .validation import validate_permutation_params


def _lambdas_for_weighted_pairs(
    d_tilde: NDArray[np.floating],
    R_list: list[NDArray[np.floating]],
    R_bar: NDArray[np.floating],
    rank: int | str | None,
    tol: float,
) -> NDArray[np.floating]:
    """Compute sorted eigenvalues for one surrogate ReDisCA problem."""
    R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)
    _, lambdas = solve_gep(R_bar_d, R_bar, rank=rank, tol=tol)
    return lambdas


def permutation_test_redisca(
    R_list: list[NDArray[np.floating]],
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
    """Run the ReDisCA permutation test.

    Each surrogate directly reshuffles the upper-triangular target-RDM entries
    against the fixed list of condition-pair matrices ``R_ij``. For each
    surrogate, ReDisCA is refit and ordered surrogate eigenvalues are stored.

    For each observed positive eigenvalue ``lambda_k``, the p-value is

        p_k = (1 + #{b : lambda_k^{(b)} >= lambda_k}) / (n_valid + 1)

    where ``lambda_k^{(b)}`` is the kth ordered eigenvalue in surrogate ``b``.
    Non-positive observed eigenvalues are assigned ``p=1`` because the test is
    designed for strong positive matches to the target RDM.

    If the requested number of valid permutations cannot be collected within
    the attempt budget, the test returns results based on the collected
    permutations and emits a warning instead of failing.
    """
    validate_permutation_params(n_perm=n_perm, alpha=alpha, tol=tol)

    pairs = pair_indices(target_rdm.shape[0])
    d_vec = vectorize_upper(target_rdm, pairs)
    rng = np.random.default_rng(random_state)

    r = len(observed_lambdas)
    null_lambdas = np.full((n_perm, r), np.nan, dtype=np.float64)

    max_attempts = n_perm * 50
    collected = 0
    attempts = 0
    while collected < n_perm:
        if attempts >= max_attempts:
            warn(
                f"Permutation test collected only {collected}/{n_perm} valid "
                f"permutations after {max_attempts} attempts. Returning p-values "
                "based on the collected null distribution. The target RDM may "
                f"be too degenerate for this number of conditions "
                f"(C={target_rdm.shape[0]}).",
                RuntimeWarning,
                stacklevel=2,
            )
            break
        attempts += 1

        try:
            pair_perm = rng.permutation(len(d_vec))
            if np.array_equal(pair_perm, np.arange(len(d_vec))):
                continue

            d_vec_perm = d_vec[pair_perm]
            if np.array_equal(d_vec_perm, d_vec):
                continue

            d_tilde_perm = standardize(d_vec_perm)
            lambdas_perm = _lambdas_for_weighted_pairs(
                d_tilde_perm,
                R_list,
                R_bar,
                rank,
                tol,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError, SciPyLinAlgError):
            # Degenerate permutation, for example a nearly constant permuted
            # RDM vector. Try another random permutation.
            continue

        n_lambdas = min(r, len(lambdas_perm))
        null_lambdas[collected, :n_lambdas] = lambdas_perm[:n_lambdas]
        collected += 1

    null_lambdas = null_lambdas[:collected]

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
            if observed_lambdas[k] <= 0.0:
                p_values[k] = 1.0
                continue

            null_k = null_lambdas[:, k]
            valid = np.isfinite(null_k)
            n_valid = int(np.count_nonzero(valid))
            if n_valid == 0:
                p_values[k] = 1.0
            else:
                p_values[k] = (
                    1 + np.sum(null_k[valid] >= observed_lambdas[k])
                ) / (n_valid + 1)

    significant = p_values < alpha

    return PermutationTestResult(
        p_values=p_values,
        significant=significant,
        null_lambdas=null_lambdas if return_null else None,
    )
