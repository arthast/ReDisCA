"""Computational core of the ReDisCA algorithm.

Contains basic functions for computing difference matrices
and preparing data for the generalized eigenvalue problem.
"""

from typing import List, Tuple
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh


def pair_indices(C: int) -> List[Tuple[int, int]]:
    """Generate list of all condition pairs (i, j) where i < j.

    Args:
        C: Number of conditions.

    Returns:
        List of tuples (i, j).
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


def vectorize_upper(
        D: NDArray[np.floating],
        pairs: list[tuple[int, int]] | None = None,
) -> NDArray[np.floating]:
    """Extract upper triangle of matrix into a vector.

    Args:
        D: Square matrix (C, C).
        pairs: Optional list of pairs (i, j). If omitted, all upper-triangular
            pairs are derived from ``D`` automatically.

    Returns:
        Vector of length C*(C-1)/2.
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be a square 2D matrix. Got shape: {D.shape}")

    if pairs is None:
        pairs = pair_indices(D.shape[0])

    return np.array([D[i, j] for i, j in pairs], dtype=D.dtype)


def _zscore_with_policy(
        vec: NDArray[np.floating],
        *,
        tol: float = 1e-10,
        on_constant: str,
) -> NDArray[np.floating] | None:
    """Z-score a vector with explicit handling of nearly constant inputs.

    Args:
        vec: Input vector.
        tol: Tolerance for detecting a nearly constant vector.
        on_constant: Policy for nearly constant vectors:
            - ``"raise"``: raise ``ValueError``
            - ``"none"``: return ``None``

    Returns:
        Standardized vector, or ``None`` when ``on_constant="none"`` and the
        input vector is nearly constant.

    Raises:
        ValueError: If ``on_constant="raise"`` and the vector is nearly constant.
    """
    std = np.std(vec, ddof=0)
    if std < tol:
        if on_constant == "raise":
            raise ValueError(
                "Standard deviation of target RDM is close to zero. "
                "RDM is uninformative (all elements are nearly equal)."
            )
        if on_constant == "none":
            return None
        raise ValueError(f"Unsupported on_constant policy: {on_constant!r}")

    return (vec - np.mean(vec)) / std


def standardize(d_vec: NDArray[np.floating]) -> NDArray[np.floating]:
    """Z-standardization of a vector.

    Args:
        d_vec: Input vector.

    Returns:
        Standardized vector (mean=0, std=1).

    Raises:
        ValueError: If std=0 (uninformative RDM).
    """
    return _zscore_with_policy(d_vec, on_constant="raise")


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


def symmetrize_matrix(
        M: NDArray[np.floating],
        *,
        name: str,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        raise_on_large: bool = False,
) -> NDArray[np.floating]:
    """Check symmetry, warn/error on noticeable antisymmetric part,
    then return 0.5 * (M + M.T).
    """
    M_sym = 0.5 * (M + M.T)
    M_anti = 0.5 * (M - M.T)

    anti_norm = np.linalg.norm(M_anti, ord="fro")
    sym_norm = np.linalg.norm(M_sym, ord="fro")
    rel = anti_norm / max(sym_norm, np.finfo(float).eps)

    if anti_norm > atol and rel > rtol:
        msg = (
            f"{name} is not symmetric within tolerance: "
            f"||anti||_F={anti_norm:.3e}, rel={rel:.3e}. "
            "This may indicate an upstream bug; matrix will be symmetrized."
        )
        if raise_on_large:
            raise ValueError(msg)
        warn(msg, RuntimeWarning, stacklevel=2)

    return M_sym


def renormalize_filters_in_metric(
        W: NDArray[np.floating],
        R_bar: NDArray[np.floating],
        *,
        min_norm_sq: float = 1e-12,
        check_rtol: float = 1e-8,
        check_atol: float = 1e-10,
) -> NDArray[np.floating]:
    """Renormalize filters so that w.T @ R_bar @ w = 1 for each column of W.

    Raises:
        RuntimeError: If a filter cannot be safely normalized or the
        post-normalization check fails.
    """
    W = W.copy()

    for i in range(W.shape[1]):
        w = W[:, i]
        norm_sq = float(w @ R_bar @ w)

        if not np.isfinite(norm_sq):
            raise RuntimeError(
                f"Failed to renormalize filter {i}: "
                f"w.T @ R_bar @ w is not finite ({norm_sq})."
            )

        if norm_sq <= min_norm_sq:
            raise RuntimeError(
                f"Failed to renormalize filter {i}: "
                f"w.T @ R_bar @ w = {norm_sq:.3e} <= {min_norm_sq:.3e}. "
                "This indicates a numerically degenerate filter or an upstream bug."
            )

        w = w / np.sqrt(norm_sq)

        norm_sq_after = float(w @ R_bar @ w)
        if (
                not np.isfinite(norm_sq_after)
                or not np.isclose(norm_sq_after, 1.0, rtol=check_rtol, atol=check_atol)
        ):
            raise RuntimeError(
                f"Post-renormalization check failed for filter {i}: "
                f"w.T @ R_bar @ w = {norm_sq_after:.16e} instead of 1."
            )

        W[:, i] = w

    return W


def validate_rank(
        rank: int | str | None,
        N: int,
) -> int | str | None:
    """Validate rank parameter."""
    if rank is None:
        return None

    if isinstance(rank, str):
        if rank == "auto":
            return rank
        raise ValueError(
            f"rank must be 'auto', None, or an integer in [1, {N}], got string {rank!r}"
        )

    if isinstance(rank, (bool, np.bool_)):
        raise TypeError(
            "rank must be 'auto', None, or a positive integer; bool is not allowed"
        )

    if isinstance(rank, (int, np.integer)):
        rank = int(rank)
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if rank > N:
            raise ValueError(f"rank must be <= N ({N}), got {rank}")
        return rank

    raise TypeError(
        f"rank must be 'auto', None, or a positive integer in [1, {N}], "
        f"got {type(rank).__name__}"
    )


def resolve_rank(
        rank: int | str | None,
        available_rank: int,
        N: int,
        tol: float,
) -> int:
    rank = validate_rank(rank, N)

    if available_rank == 0:
        raise ValueError(
            "No positive eigenvalues found in R_bar. "
            "Data or time window is uninformative."
        )

    if rank == "auto":
        return available_rank

    if rank is None:
        if available_rank < N:
            warn(
                f"rank=None means no explicit user rank cap, not forced full rank. "
                f"R_bar has only {available_rank} eigenvalues > tol={tol}, "
                f"so {available_rank} components will be returned instead of N={N}.",
                RuntimeWarning,
                stacklevel=2
            )
        return available_rank

    if rank > available_rank:
        warn(
            f"Requested rank={rank}, but only {available_rank} eigenvalues of "
            f"R_bar exceed tol={tol}. Using rank={available_rank}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return available_rank

    return rank


def solve_gep(
        R_bar_d: NDArray[np.floating],
        R_bar: NDArray[np.floating],
        rank: int | str | None = "auto",
        tol: float = 1e-10,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Solve generalized eigenvalue problem using principal subspace approach.

    As per the paper: if the GEP is not full rank, the procedure is performed
    in the lower dimensional principal space and topographies are transformed
    back to the original sensor space.

    Solves: R_bar_d @ w = lambda * R_bar @ w
    Subject to: w.T @ R_bar @ w = 1

    Args:
        R_bar_d: Target weighted matrix (N, N).
        R_bar: Mean difference covariance matrix (N, N).
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

    Returns:
        Tuple of:
            W: Spatial filters (N, r), columns are filters in sensor space.
            lambdas: Eigenvalues (r,), sorted descending.

    Raises:
        ValueError: If no positive eigenvalues found.
        RuntimeError: If selected directions are numerically inconsistent
            or filters cannot be properly renormalized.
    """
    N = R_bar.shape[0]

    # Compute eigendecomposition of R_bar for principal subspace
    eig_vals, eig_vecs = eigh(R_bar)

    available_rank = int(np.sum(eig_vals > tol))
    r = resolve_rank(rank, available_rank, N, tol)

    # Select top r eigenvectors (largest eigenvalues)
    idx = np.argsort(eig_vals)[::-1][:r]
    V_r = eig_vecs[:, idx]
    D_r = eig_vals[idx]

    # Check that selected eigenvalues are positive
    if np.any(D_r <= tol):
        raise RuntimeError(
            "Internal error: selected eigenvalues contain values <= tol "
            "despite available-rank filtering."
        )

    # Transform to principal subspace
    R_bar_d_pca = V_r.T @ R_bar_d @ V_r
    R_bar_d_pca = symmetrize_matrix(
        R_bar_d_pca,
        name="R_bar_d_pca",
        atol=1e-12,
        rtol=1e-10,
    )
    R_bar_pca = np.diag(D_r)

    # Solve GEP in reduced space
    lambdas_pca, W_pca = eigh(R_bar_d_pca, R_bar_pca)

    # Transform filters back to sensor space: W = V_r @ W_pca
    W = V_r @ W_pca  # (N, r)

    # Renormalize filters: w.T @ R_bar @ w = 1
    W = renormalize_filters_in_metric(W, R_bar)

    # Recompute lambdas in sensor space: lambda_i = w_i.T @ R_bar_d @ w_i
    lambdas = np.array([W[:, i] @ R_bar_d @ W[:, i] for i in range(r)], dtype=np.float64)

    # Sort by eigenvalues (descending)
    sort_idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[sort_idx]
    W = W[:, sort_idx]

    return W, lambdas


def compute_patterns(
        W: NDArray[np.floating],
        R_bar: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Compute source topographies (patterns) from spatial filters.

    Always uses the numerically stable formula:
        A = R_bar @ W @ (W.T @ R_bar @ W)^{-1}

    This avoids direct matrix inversion of W which is ill-conditioned.
    The result satisfies W.T @ A ≈ I_r.

    Args:
        W: Spatial filters (N, r), columns are filters.
        R_bar: Mean difference covariance matrix (N, N).

    Returns:
        A: Pattern matrix (N, r), columns are topographies.
    """
    WtRW = W.T @ R_bar @ W  # (r, r)
    RW = R_bar @ W  # (N, r)

    try:
        A = np.linalg.solve(WtRW.T, RW.T).T  # (N, r)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse as fallback
        warn(
            "WtRW is singular in compute_patterns; using pseudo-inverse. "
            "Returned patterns may be numerically unstable.",
            RuntimeWarning,
            stacklevel=2,
        )
        A = RW @ np.linalg.pinv(WtRW)
    return A


def compute_component_timeseries(
        W: NDArray[np.floating],
        X: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Compute component time series for all conditions.

    U_c = W.T @ X_c for each condition c.

    Args:
        W: Spatial filters matrix (N, r), columns are filters.
        X: Evoked data (C, N, T).

    Returns:
        Component time series (C, r, T), where r is the number of components.
    """
    C = X.shape[0]
    r = W.shape[1]
    T = X.shape[2]
    U = np.zeros((C, r, T), dtype=np.float64)

    for c in range(C):
        U[c] = W.T @ X[c]

    return U


def compute_component_rdms(
        W: NDArray[np.floating],
        R_list: List[NDArray[np.floating]],
        pairs: List[Tuple[int, int]],
        C: int
) -> NDArray[np.floating]:
    """Compute RDM for each component.

    D_hat[n, i, j] = w_n.T @ R_ij @ w_n

    Uses optimized computation: diag(W.T @ R_ij @ W) gives all components at once.

    Args:
        W: Spatial filters matrix (N, r), columns are filters.
        R_list: List of R_ij matrices (must match pairs order).
        pairs: List of condition pairs (i, j).
        C: Number of conditions.

    Returns:
        Component RDMs of shape (r, C, C), symmetric matrices with zeros on diagonal.
    """
    r = W.shape[1]
    D_hat = np.zeros((r, C, C), dtype=np.float64)

    for k, (i, j) in enumerate(pairs):
        R_ij = R_list[k]
        # Compute all components at once: M = W.T @ R_ij @ W, vals = diag(M)
        M = W.T @ R_ij @ W
        vals = np.diag(M)
        D_hat[:, i, j] = vals
        D_hat[:, j, i] = vals  # Symmetric

    return D_hat


def _standardize_or_none(
        vec: NDArray[np.floating],
        tol: float = 1e-10,
) -> NDArray[np.floating] | None:
    """Return z-scored vector, or None if vector is nearly constant."""
    return _zscore_with_policy(vec, tol=tol, on_constant="none")


def compute_pearson_scores(
        target_rdm: NDArray[np.floating],
        component_rdms: NDArray[np.floating] | list[tuple[int, int]],
        pairs: list[tuple[int, int]] | NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Compute Pearson correlation between target RDM and each component RDM.

    Supports both the public API style
        compute_pearson_scores(target_rdm, component_rdms, pairs=None)
    and the older internal style
        compute_pearson_scores(target_rdm, pairs, component_rdms).

    Target RDM must be informative: nearly constant target raises ValueError.
    Nearly constant component RDM gets score = 0.0 by design.
    """
    # Backward-compatible support for the older argument order:
    # compute_pearson_scores(target_rdm, pairs, component_rdms)
    if (
            pairs is not None
            and isinstance(component_rdms, list)
            and all(isinstance(pair, tuple) and len(pair) == 2 for pair in component_rdms)
    ):
        component_rdms, pairs = pairs, component_rdms

    component_rdms = np.asarray(component_rdms, dtype=np.float64)
    if component_rdms.ndim != 3:
        raise ValueError(
            "component_rdms must have shape (r, C, C). "
            f"Got shape: {component_rdms.shape}"
        )

    if pairs is None:
        pairs = pair_indices(target_rdm.shape[0])

    r = component_rdms.shape[0]
    scores = np.zeros(r, dtype=np.float64)

    target_vec = vectorize_upper(target_rdm, pairs)
    target_z = standardize(target_vec)  # intentionally raises on degenerate target

    for comp in range(r):
        comp_vec = vectorize_upper(component_rdms[comp], pairs)
        comp_z = _standardize_or_none(comp_vec)

        if comp_z is None:
            scores[comp] = 0.0
            continue

        scores[comp] = np.mean(target_z * comp_z)

    return scores
