"""Computational core of the ReDisCA algorithm.

Contains basic functions for computing difference matrices
and preparing data for the generalized eigenvalue problem.
"""

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.stats import zscore


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


def vectorize_upper(D: NDArray[np.floating]) -> NDArray[np.floating]:
    """Extract upper triangle of matrix into a vector.

    Args:
        D: Square matrix (C, C).

    Returns:
        Vector of length C*(C-1)/2.
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
            - "auto": automatically determine rank from positive eigenvalues
            - int: use specified rank
            - None: use full rank (no dimensionality reduction)
        tol: Threshold for determining positive eigenvalues when rank="auto".

    Returns:
        Tuple of:
            W: Spatial filters (N, r), columns are filters in sensor space.
            lambdas: Eigenvalues (r,), sorted descending.

    Raises:
        ValueError: If no positive eigenvalues found (uninformative data).
    """
    N = R_bar.shape[0]

    # Symmetrize matrices for numerical stability
    R_bar_d = 0.5 * (R_bar_d + R_bar_d.T)
    R_bar = 0.5 * (R_bar + R_bar.T)

    # Compute eigendecomposition of R_bar for principal subspace
    eig_vals, eig_vecs = eigh(R_bar)

    # Determine rank based on positive eigenvalues only (R_bar should be PSD)
    if rank == "auto":
        # Keep only components with positive eigenvalues (above tolerance)
        positive_mask = eig_vals > tol
        r = np.sum(positive_mask)
        if r == 0:
            raise ValueError(
                "No positive eigenvalues found in R_bar. "
                "Data or time window is uninformative."
            )
    elif rank is None:
        r = N
    else:
        r = min(int(rank), N)

    # Select top r eigenvectors (largest eigenvalues)
    idx = np.argsort(eig_vals)[::-1][:r]
    V_r = eig_vecs[:, idx]  # (N, r)
    D_r = eig_vals[idx]     # (r,)

    # Check that selected eigenvalues are positive
    if np.any(D_r <= tol):
        # Reduce to only positive ones
        pos_mask = D_r > tol
        if not np.any(pos_mask):
            raise ValueError(
                "No positive eigenvalues in selected subspace. "
                "Try reducing rank or check input data."
            )
        V_r = V_r[:, pos_mask]
        D_r = D_r[pos_mask]
        r = len(D_r)

    # Transform to principal subspace
    R_bar_d_pca = V_r.T @ R_bar_d @ V_r
    R_bar_pca = np.diag(D_r)

    # Symmetrize for numerical stability
    R_bar_d_pca = 0.5 * (R_bar_d_pca + R_bar_d_pca.T)

    # Solve GEP in reduced space
    lambdas_pca, W_pca = eigh(R_bar_d_pca, R_bar_pca)

    # Transform filters back to sensor space: W = V_r @ W_pca
    W = V_r @ W_pca  # (N, r)

    # Renormalize filters: w.T @ R_bar @ w = 1
    for i in range(r):
        w = W[:, i]
        norm_sq = w @ R_bar @ w
        if norm_sq > 1e-12:
            W[:, i] = w / np.sqrt(norm_sq)

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

    As per the paper:
    - If W is square and invertible: A = W^(-1), where rows are topographies.
    - For non-square W (after rank reduction): A = (W.T @ R_bar @ W)^(-1) @ W.T @ R_bar

    Args:
        W: Spatial filters (N, r), columns are filters.
        R_bar: Mean difference covariance matrix (N, N).

    Returns:
        A: Pattern matrix (r, N), rows are topographies.
    """
    N, r = W.shape
    R_bar = 0.5 * (R_bar + R_bar.T)

    if r == N:
        # Square case
        try:
            A = np.linalg.inv(W)  # (N, N), rows are topographies
            return A
        except np.linalg.LinAlgError:
            pass

    # General formula for non-square or singular W:
    # A = (W.T @ R_bar @ W)^(-1) @ W.T @ R_bar
    WtRW = W.T @ R_bar @ W  # (r, r)
    WtR = W.T @ R_bar       # (r, N)

    try:
        A = np.linalg.solve(WtRW, WtR)  # (r, N)
    except np.linalg.LinAlgError:
        # If still singular, use pseudo-inverse
        A = np.linalg.pinv(WtRW) @ WtR

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


def compute_pearson_scores(
    target_rdm: NDArray[np.floating],
    component_rdms: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Compute Pearson correlation between target RDM and each component RDM.

    Uses z-standardization for both vectors as per the ReDisCA paper formula (2):
        rho = (1/P) * sum(d_tilde_k * d_hat_tilde_k)

    where d_tilde and d_hat_tilde are z-scored upper triangle vectors.

    Args:
        target_rdm: Target RDM of shape (C, C).
        component_rdms: Component RDMs of shape (r, C, C).

    Returns:
        Pearson correlations for each component (r,).

    Note:
        Target RDM must not be constant (validated in fit_redisca).
        Component RDMs with constant values get score = 0.
    """
    r = component_rdms.shape[0]
    scores = np.zeros(r, dtype=np.float64)

    target_vec = vectorize_upper(target_rdm)
    target_z = standardize(target_vec)

    for comp in range(r):
        comp_vec = vectorize_upper(component_rdms[comp])
        comp_std = np.std(comp_vec, ddof=0)

        if comp_std < 1e-10:
            scores[comp] = 0.0
            continue

        comp_z = (comp_vec - np.mean(comp_vec)) / comp_std
        scores[comp] = np.mean(target_z * comp_z)

    return scores


