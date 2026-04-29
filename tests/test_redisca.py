import numpy as np
import pytest
import matplotlib
import matplotlib.pyplot as plt
import types
from numpy.testing import assert_allclose, assert_array_equal

from redisca import export_result, fit_redisca, validate_inputs, ReDisCAResult
from redisca.types import PermutationTestResult
from redisca.stats import permutation_test_redisca
from redisca.viz import (
    plot_rdm,
    plot_top_component_rdms,
    plot_component_scores,
    plot_component_lambdas,
    plot_component_timeseries,
    plot_patterns,
)
from redisca.viz_mne import (
    plot_pattern_topomaps,
    plot_compare_conditions,
    plot_condition_joint,
)
from redisca.core import (
    pair_indices,
    compute_R_ij,
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


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_data():
    """Simple synthetic data for basic tests."""
    np.random.seed(42)
    C, N, T = 4, 10, 100
    X = np.random.randn(C, N, T)
    target_rdm = np.array([
        [0, 1, 2, 2],
        [1, 0, 2, 2],
        [2, 2, 0, 1],
        [2, 2, 1, 0]
    ], dtype=float)
    return X, target_rdm


@pytest.fixture
def structured_data():
    """Data with known structure for testing RDM recovery.

    One latent source encodes the target contrast (groups {0,1} vs {2,3}),
    while additional nuisance sources and noise ensure that the first
    ReDisCA component is the dominant match rather than *all* components.
    """
    rng = np.random.default_rng(123)
    C, N, T = 4, 20, 200

    n_sources = 4
    A_mix = rng.standard_normal((N, n_sources))
    S = rng.standard_normal((n_sources, T))

    # source 0: target contrast; source 1: nuisance; sources 2-3: common noise
    amp = np.zeros((C, n_sources))
    amp[:, 0] = [1.0, 1.0, -1.0, -1.0]       # target
    amp[:, 1] = [1.0, -1.0, 1.0, -1.0]        # nuisance
    amp[:, 2:] = 1.0                            # noise (same for all conds)

    source_scale = np.array([3.0, 1.5, 3.0, 3.0])

    X = np.zeros((C, N, T))
    for c in range(C):
        for s in range(n_sources):
            X[c] += source_scale[s] * amp[c, s] * (
                A_mix[:, s:s+1] @ S[s:s+1, :]
            )
        X[c] += 0.5 * rng.standard_normal((N, T))

    # Target RDM: 0-1 similar, 2-3 similar, cross-group different
    target_rdm = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0]
    ], dtype=float)

    return X, target_rdm


@pytest.fixture
def result_with_pvalues():
    """Small result WITH permutation test for visualization tests."""
    np.random.seed(7)
    C, N, T = 4, 6, 20
    X = np.random.randn(C, N, T)
    target_rdm = np.array([
        [0, 1, 2, 2],
        [1, 0, 2, 2],
        [2, 2, 0, 1],
        [2, 2, 1, 0],
    ], dtype=float)
    return fit_redisca(
        X,
        target_rdm,
        permutation_test=True,
        n_perm=30,
        random_state=0,
    )


@pytest.fixture
def result_no_pvalues():
    """Small result WITHOUT permutation test for visualization tests."""
    np.random.seed(7)
    C, N, T = 4, 6, 20
    X = np.random.randn(C, N, T)
    target_rdm = np.array([
        [0, 1, 2, 2],
        [1, 0, 2, 2],
        [2, 2, 0, 1],
        [2, 2, 1, 0],
    ], dtype=float)
    return fit_redisca(X, target_rdm)


# =============================================================================
# Test: pair_indices
# =============================================================================

class TestPairIndices:
    """Tests for pair_indices function."""

    def test_pair_count(self):
        """Number of pairs should be C*(C-1)/2."""
        for C in [3, 4, 5, 10]:
            pairs = pair_indices(C)
            expected = C * (C - 1) // 2
            assert len(pairs) == expected, f"C={C}: expected {expected}, got {len(pairs)}"

    def test_pair_order(self):
        """Pairs should have i < j."""
        pairs = pair_indices(5)
        for i, j in pairs:
            assert i < j, f"Invalid pair: ({i}, {j})"

    def test_all_pairs_present(self):
        """All valid pairs should be present."""
        C = 4
        pairs = pair_indices(C)
        expected = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        assert pairs == expected


# =============================================================================
# Test: compute_R_ij
# =============================================================================

class TestComputeRij:
    """Tests for compute_R_ij function."""

    def test_shape(self):
        """R_ij should be (N, N)."""
        N, T = 10, 50
        X_i = np.random.randn(N, T)
        X_j = np.random.randn(N, T)
        R = compute_R_ij(X_i, X_j)
        assert R.shape == (N, N)

    def test_symmetry(self):
        """R_ij should be symmetric."""
        N, T = 10, 50
        X_i = np.random.randn(N, T)
        X_j = np.random.randn(N, T)
        R = compute_R_ij(X_i, X_j)
        assert_allclose(R, R.T)

    def test_positive_semidefinite(self):
        """R_ij should be positive semi-definite."""
        N, T = 10, 50
        X_i = np.random.randn(N, T)
        X_j = np.random.randn(N, T)
        R = compute_R_ij(X_i, X_j)
        eigenvalues = np.linalg.eigvalsh(R)
        assert np.all(eigenvalues >= -1e-10), "R_ij should be PSD"

    def test_identical_signals_zero(self):
        """R_ij should be zero if X_i == X_j."""
        N, T = 10, 50
        X = np.random.randn(N, T)
        R = compute_R_ij(X, X)
        assert_allclose(R, np.zeros((N, N)), atol=1e-14)


# =============================================================================
# Test: vectorize_upper
# =============================================================================

class TestVectorizeUpper:
    """Tests for vectorize_upper function."""

    def test_length(self):
        """Output length should be C*(C-1)/2."""
        C = 5
        D = np.random.randn(C, C)
        vec = vectorize_upper(D)
        expected_len = C * (C - 1) // 2
        assert len(vec) == expected_len

    def test_correct_elements(self):
        """Should extract upper triangle correctly."""
        D = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ])
        vec = vectorize_upper(D)
        expected = np.array([1, 2, 3])
        assert_array_equal(vec, expected)

    def test_explicit_and_implicit_pairs_match(self):
        """Explicit pairs should match auto-derived upper triangle indices."""
        D = np.array([
            [0, 1, 2, 4],
            [1, 0, 3, 5],
            [2, 3, 0, 6],
            [4, 5, 6, 0],
        ], dtype=float)
        pairs = pair_indices(D.shape[0])
        assert_array_equal(vectorize_upper(D), vectorize_upper(D, pairs))


# =============================================================================
# Test: standardize
# =============================================================================

class TestStandardize:
    """Tests for standardize function."""

    def test_mean_zero(self):
        """Standardized vector should have mean ≈ 0."""
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = standardize(vec)
        assert_allclose(np.mean(z), 0, atol=1e-10)

    def test_std_one(self):
        """Standardized vector should have std ≈ 1."""
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = standardize(vec)
        assert_allclose(np.std(z, ddof=0), 1, atol=1e-10)

    def test_constant_raises(self):
        """Constant vector should raise ValueError."""
        vec = np.array([5.0, 5.0, 5.0])
        with pytest.raises(ValueError, match="uninformative"):
            standardize(vec)


# =============================================================================
# Test: solve_gep
# =============================================================================

class TestSolveGep:
    """Tests for solve_gep function."""

    def test_output_shapes(self, simple_data):
        """W and lambdas should have correct shapes."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        W, lambdas = solve_gep(R_bar_d, R_bar, rank=None)

        assert W.shape == (N, N)
        assert lambdas.shape == (N,)

    def test_rank_reduction(self, simple_data):
        """Rank parameter should limit number of components."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        rank = 5
        W, lambdas = solve_gep(R_bar_d, R_bar, rank=rank)

        assert W.shape == (N, rank)
        assert lambdas.shape == (rank,)

    def test_lambdas_sorted_descending(self, simple_data):
        """Eigenvalues should be sorted in descending order."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        W, lambdas = solve_gep(R_bar_d, R_bar)

        assert np.all(lambdas[:-1] >= lambdas[1:]), "Lambdas should be descending"

    def test_filter_normalization(self, simple_data):
        """Filters should satisfy w.T @ R_bar @ w = 1."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        W, lambdas = solve_gep(R_bar_d, R_bar)

        for i in range(W.shape[1]):
            w = W[:, i]
            norm_sq = w @ R_bar @ w
            assert_allclose(norm_sq, 1.0, atol=1e-6,
                          err_msg=f"Filter {i} not normalized")


# =============================================================================
# Test: compute_patterns
# =============================================================================

class TestComputePatterns:
    """Tests for compute_patterns function."""

    def test_wt_a_identity(self, simple_data):
        """W.T @ A should be approximately identity."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        W, _ = solve_gep(R_bar_d, R_bar)
        A = compute_patterns(W, R_bar)

        WTA = W.T @ A
        assert_allclose(WTA, np.eye(N), atol=1e-6)

    def test_rank_reduced_wt_a(self, simple_data):
        """W.T @ A should be identity even with rank reduction."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        rank = 5
        W, _ = solve_gep(R_bar_d, R_bar, rank=rank)
        A = compute_patterns(W, R_bar)

        WTA = W.T @ A
        assert_allclose(WTA, np.eye(rank), atol=1e-6)


# =============================================================================
# Test: compute_component_rdms
# =============================================================================

class TestComputeComponentRdms:
    """Tests for compute_component_rdms function."""

    def test_shape(self, simple_data):
        """Component RDMs should have shape (r, C, C)."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        W, _ = solve_gep(R_bar_d, R_bar)
        D_hat = compute_component_rdms(W, R_list, pairs, C)

        assert D_hat.shape == (N, C, C)

    def test_symmetry(self, simple_data):
        """Each component RDM should be symmetric."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        W, _ = solve_gep(R_bar_d, R_bar)
        D_hat = compute_component_rdms(W, R_list, pairs, C)

        for n in range(N):
            assert_allclose(D_hat[n], D_hat[n].T,
                          err_msg=f"Component {n} RDM not symmetric")

    def test_diagonal_zero(self, simple_data):
        """Diagonal of each component RDM should be zero."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        W, _ = solve_gep(R_bar_d, R_bar)
        D_hat = compute_component_rdms(W, R_list, pairs, C)

        for n in range(N):
            assert_allclose(np.diag(D_hat[n]), np.zeros(C),
                          err_msg=f"Component {n} RDM diagonal not zero")

    def test_non_negative(self, simple_data):
        """Component RDM values should be non-negative (squared distances)."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        W, _ = solve_gep(R_bar_d, R_bar)
        D_hat = compute_component_rdms(W, R_list, pairs, C)

        assert np.all(D_hat >= -1e-10), "Component RDM values should be >= 0"


# =============================================================================
# Test: compute_pearson_scores
# =============================================================================

class TestComputePearsonScores:
    """Tests for compute_pearson_scores function."""

    def test_range(self, simple_data):
        """Pearson scores should be in [-1, 1]."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        W, _ = solve_gep(R_bar_d, R_bar)
        D_hat = compute_component_rdms(W, R_list, pairs, C)
        scores = compute_pearson_scores(target_rdm, D_hat)

        assert np.all(scores >= -1.0 - 1e-6)
        assert np.all(scores <= 1.0 + 1e-6)

    def test_perfect_correlation(self):
        """Identical RDMs should give correlation = 1."""
        target_rdm = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ], dtype=float)

        component_rdms = np.array([target_rdm])  # Shape (1, 3, 3)
        scores = compute_pearson_scores(target_rdm, component_rdms)

        assert_allclose(scores[0], 1.0, atol=1e-10)

    def test_explicit_pairs_match_implicit(self):
        """Explicit pairs should produce the same Pearson scores."""
        target_rdm = np.array([
            [0, 1, 2, 2],
            [1, 0, 2, 2],
            [2, 2, 0, 1],
            [2, 2, 1, 0]
        ], dtype=float)
        component_rdms = np.array([target_rdm, target_rdm * 2], dtype=float)
        pairs = pair_indices(target_rdm.shape[0])

        scores_implicit = compute_pearson_scores(target_rdm, component_rdms)
        scores_explicit = compute_pearson_scores(target_rdm, component_rdms, pairs=pairs)

        assert_allclose(scores_implicit, scores_explicit)


# =============================================================================
# Test: fit_redisca (integration)
# =============================================================================

class TestFitRedisca:
    """Integration tests for fit_redisca."""

    def test_returns_correct_type(self, simple_data):
        """fit_redisca should return ReDisCAResult."""
        X, target_rdm = simple_data
        result = fit_redisca(X, target_rdm)
        assert isinstance(result, ReDisCAResult)

    def test_output_shapes(self, simple_data):
        """All outputs should have correct shapes."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        result = fit_redisca(X, target_rdm)

        r = result.n_components
        assert result.W.shape == (N, r)
        assert result.A.shape == (N, r)
        assert result.lambdas.shape == (r,)
        assert result.pearson_scores.shape == (r,)
        assert result.component_timeseries.shape == (C, r, T)
        assert result.component_rdms.shape == (r, C, C)
        assert result.target_rdm.shape == (C, C)

    def test_metadata(self, simple_data):
        """Metadata should match input dimensions."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        result = fit_redisca(X, target_rdm)

        assert result.n_conditions == C
        assert result.n_channels == N
        assert result.n_timepoints == T

    def test_list_input(self, simple_data):
        """Should accept list of arrays as input."""
        X, target_rdm = simple_data
        X_list = [X[c] for c in range(X.shape[0])]

        result = fit_redisca(X_list, target_rdm)
        assert isinstance(result, ReDisCAResult)

    def test_rank_parameter(self, simple_data):
        """Rank parameter should limit components."""
        X, target_rdm = simple_data

        result = fit_redisca(X, target_rdm, rank=3)

        assert result.n_components == 3
        assert result.W.shape[1] == 3

    def test_deterministic(self, simple_data):
        """Results should be deterministic."""
        X, target_rdm = simple_data

        result1 = fit_redisca(X, target_rdm)
        result2 = fit_redisca(X, target_rdm)

        assert_allclose(result1.W, result2.W)
        assert_allclose(result1.lambdas, result2.lambdas)
        assert_allclose(result1.pearson_scores, result2.pearson_scores)

    def test_first_component_best_correlation(self, simple_data):
        """First component should have highest Pearson score (usually)."""
        X, target_rdm = simple_data
        result = fit_redisca(X, target_rdm)

        # First lambda is highest, and generally first Pearson score is high
        assert result.lambdas[0] >= result.lambdas[-1]

    def test_structured_data_high_correlation(self, structured_data):
        """With structured data, first component should have high correlation."""
        X, target_rdm = structured_data
        result = fit_redisca(X, target_rdm)

        # With well-structured data, we expect high correlation
        assert result.pearson_scores[0] > 0.5, \
            f"Expected high correlation, got {result.pearson_scores[0]}"


# =============================================================================
# Test: Validation
# =============================================================================

class TestValidation:
    """Tests for input validation."""

    def test_non_square_rdm_raises(self):
        """Non-square RDM should raise ValueError."""
        X = np.random.randn(4, 10, 100)
        target_rdm = np.random.randn(4, 5)

        with pytest.raises(ValueError, match="square"):
            fit_redisca(X, target_rdm)

    def test_mismatched_conditions_raises(self):
        """Mismatched C between X and RDM should raise ValueError."""
        X = np.random.randn(4, 10, 100)
        target_rdm = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ], dtype=float)

        with pytest.raises(ValueError, match="does not match"):
            fit_redisca(X, target_rdm)

    def test_asymmetric_rdm_raises(self):
        """Asymmetric RDM should raise ValueError."""
        X = np.random.randn(3, 10, 100)
        target_rdm = np.array([
            [0, 1, 2],
            [3, 0, 4],
            [5, 6, 0]
        ], dtype=float)

        with pytest.raises(ValueError, match="symmetric"):
            fit_redisca(X, target_rdm)

    def test_nonzero_diagonal_raises(self):
        """Non-zero diagonal RDM should raise ValueError."""
        X = np.random.randn(3, 10, 100)
        target_rdm = np.array([
            [1, 1, 2],
            [1, 1, 3],
            [2, 3, 1]
        ], dtype=float)

        with pytest.raises(ValueError, match="diagonal"):
            fit_redisca(X, target_rdm)

    def test_negative_rdm_raises(self):
        """Negative values in RDM should raise ValueError."""
        X = np.random.randn(3, 10, 100)
        target_rdm = np.array([
            [0, -1, 2],
            [-1, 0, 3],
            [2, 3, 0]
        ], dtype=float)

        with pytest.raises(ValueError, match="non-negative"):
            fit_redisca(X, target_rdm)

    def test_nan_in_x_raises(self):
        """NaN in X should raise ValueError."""
        X = np.random.randn(3, 10, 100)
        X[0, 0, 0] = np.nan
        target_rdm = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ], dtype=float)

        with pytest.raises(ValueError, match="NaN"):
            fit_redisca(X, target_rdm)

    def test_too_few_conditions_raises(self):
        """Less than 3 conditions should raise ValueError."""
        X = np.random.randn(2, 10, 100)
        target_rdm = np.array([
            [0, 1],
            [1, 0]
        ], dtype=float)

        with pytest.raises(ValueError, match="at least 3"):
            fit_redisca(X, target_rdm)


# =============================================================================
# Test: Mathematical Properties
# =============================================================================

class TestMathematicalProperties:
    """Tests verifying mathematical properties from the paper."""

    def test_lambda_equals_w_R_bar_d_w(self, simple_data):
        """Lambda should equal w.T @ R_bar_d @ w for each component."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        W, lambdas = solve_gep(R_bar_d, R_bar)

        for i in range(W.shape[1]):
            w = W[:, i]
            computed_lambda = w @ R_bar_d @ w
            assert_allclose(computed_lambda, lambdas[i], rtol=1e-6,
                          err_msg=f"Lambda mismatch for component {i}")

    def test_component_rdm_formula(self, simple_data):
        """D_hat[n,i,j] should equal w_n.T @ R_ij @ w_n."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        W, _ = solve_gep(R_bar_d, R_bar)
        D_hat = compute_component_rdms(W, R_list, pairs, C)

        # Check a few random elements
        for k, (i, j) in enumerate(pairs[:3]):
            R_ij = R_list[k]
            for n in range(min(3, W.shape[1])):
                w = W[:, n]
                expected = w @ R_ij @ w
                assert_allclose(D_hat[n, i, j], expected, rtol=1e-6)

    def test_timeseries_formula(self, simple_data):
        """U[c] should equal W.T @ X[c]."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        W, _ = solve_gep(R_bar_d, R_bar)
        U = compute_component_timeseries(W, X)

        for c in range(C):
            expected = W.T @ X[c]
            assert_allclose(U[c], expected, rtol=1e-10)


# =============================================================================
# Test: Permutation Test
# =============================================================================

class TestPermutationTest:
    """Tests for the permutation test."""

    @pytest.fixture
    def small_data(self):
        """Small synthetic data for permutation test smoke tests."""
        np.random.seed(99)
        C, N, T = 4, 5, 10
        X = np.random.randn(C, N, T)
        target_rdm = np.array([
            [0, 1, 2, 2],
            [1, 0, 2, 2],
            [2, 2, 0, 1],
            [2, 2, 1, 0]
        ], dtype=float)
        return X, target_rdm

    def test_smoke_shapes(self, small_data):
        """p_values and significant should have shape (r,)."""
        X, target_rdm = small_data
        result = fit_redisca(
            X, target_rdm,
            permutation_test=True, n_perm=50,
            random_state=0,
        )
        r = result.n_components
        assert result.p_values is not None
        assert result.significant is not None
        assert result.p_values.shape == (r,)
        assert result.significant.shape == (r,)

    def test_smoke_standalone(self, small_data):
        """Standalone permutation_test_redisca should return correct shapes."""
        X, target_rdm = small_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        R_bar = compute_R_bar(R_list)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)
        W, lambdas = solve_gep(R_bar_d, R_bar)

        perm_result = permutation_test_redisca(
            R_list=R_list,
            R_bar=R_bar,
            target_rdm=target_rdm,
            observed_lambdas=lambdas,
            n_perm=50,
            random_state=42,
            return_null=True,
        )
        assert isinstance(perm_result, PermutationTestResult)
        r = len(lambdas)
        assert perm_result.p_values.shape == (r,)
        assert perm_result.significant.shape == (r,)
        assert perm_result.null_max_lambdas is not None
        assert perm_result.null_max_lambdas.shape == (50,)

    def test_p_values_range(self, small_data):
        """p-values must be in (0, 1]."""
        X, target_rdm = small_data
        result = fit_redisca(
            X, target_rdm,
            permutation_test=True, n_perm=50,
            random_state=0,
        )
        assert np.all(result.p_values > 0)
        assert np.all(result.p_values <= 1)

    def test_determinism(self, small_data):
        """Same random_state must give identical p_values."""
        X, target_rdm = small_data

        r1 = fit_redisca(
            X, target_rdm,
            permutation_test=True, n_perm=50,
            random_state=123,
        )
        r2 = fit_redisca(
            X, target_rdm,
            permutation_test=True, n_perm=50,
            random_state=123,
        )
        assert_allclose(r1.p_values, r2.p_values)

    def test_different_seeds_differ(self, small_data):
        """Different random_state should (almost certainly) give different p_values."""
        X, target_rdm = small_data

        r1 = fit_redisca(
            X, target_rdm,
            permutation_test=True, n_perm=100,
            random_state=0,
        )
        r2 = fit_redisca(
            X, target_rdm,
            permutation_test=True, n_perm=100,
            random_state=999,
        )
        # With different seeds the null distributions should differ
        # so p_values are very unlikely to be identical
        assert not np.allclose(r1.p_values, r2.p_values)

    def test_constant_rdm_raises(self):
        """Constant target_rdm should raise ValueError (uninformative)."""
        np.random.seed(0)
        C, N, T = 4, 5, 10
        X = np.random.randn(C, N, T)
        # All off-diagonal elements are equal → std ≈ 0
        target_rdm = np.ones((C, C), dtype=float) - np.eye(C)

        with pytest.raises(ValueError, match="uninformative"):
            fit_redisca(
                X, target_rdm,
                permutation_test=True, n_perm=50,
                random_state=0,
            )

    def test_no_permutation_test_by_default(self, small_data):
        """Without permutation_test=True, p_values and significant are None."""
        X, target_rdm = small_data
        result = fit_redisca(X, target_rdm)
        assert result.p_values is None
        assert result.significant is None

    def test_return_null_false(self, small_data):
        """Standalone call with return_null=False should not include null distribution."""
        X, target_rdm = small_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        R_bar = compute_R_bar(R_list)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)
        _, lambdas = solve_gep(R_bar_d, R_bar)

        perm_result = permutation_test_redisca(
            R_list=R_list,
            R_bar=R_bar,
            target_rdm=target_rdm,
            observed_lambdas=lambdas,
            n_perm=30,
            random_state=0,
            return_null=False,
        )
        assert perm_result.null_max_lambdas is None

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_perm": 0}, "n_perm must be >= 1"),
            ({"n_perm": -1}, "n_perm must be >= 1"),
            ({"alpha": 0.0}, "alpha must satisfy 0 < alpha < 1"),
            ({"alpha": 1.0}, "alpha must satisfy 0 < alpha < 1"),
            ({"alpha": -0.1}, "alpha must satisfy 0 < alpha < 1"),
            ({"alpha": 1.5}, "alpha must satisfy 0 < alpha < 1"),
        ],
    )
    def test_invalid_fit_permutation_params_raise(self, small_data, kwargs, match):
        """fit_redisca should reject invalid permutation parameters."""
        X, target_rdm = small_data

        with pytest.raises(ValueError, match=match):
            fit_redisca(
                X,
                target_rdm,
                permutation_test=True,
                random_state=0,
                **kwargs,
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_perm": 0}, "n_perm must be >= 1"),
            ({"n_perm": -1}, "n_perm must be >= 1"),
            ({"alpha": 0.0}, "alpha must satisfy 0 < alpha < 1"),
            ({"alpha": 1.0}, "alpha must satisfy 0 < alpha < 1"),
            ({"alpha": -0.1}, "alpha must satisfy 0 < alpha < 1"),
            ({"alpha": 1.5}, "alpha must satisfy 0 < alpha < 1"),
        ],
    )
    def test_invalid_standalone_permutation_params_raise(self, small_data, kwargs, match):
        """Standalone permutation_test_redisca should reject invalid params."""
        X, target_rdm = small_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        R_bar = compute_R_bar(R_list)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)
        _, lambdas = solve_gep(R_bar_d, R_bar)

        with pytest.raises(ValueError, match=match):
            permutation_test_redisca(
                R_list=R_list,
                R_bar=R_bar,
                target_rdm=target_rdm,
                observed_lambdas=lambdas,
                random_state=0,
                **kwargs,
            )


# =============================================================================
# Test: Export
# =============================================================================

class TestExportResult:
    """Tests for export_result helper."""

    def test_export_bundle_writes_expected_files(self, simple_data, tmp_path):
        """Export should write the expected artifact bundle."""
        import csv
        import json

        X, target_rdm = simple_data
        result = fit_redisca(X, target_rdm)

        paths = export_result(result, tmp_path / "bundle")

        assert set(paths) == {"arrays", "component_scores", "target_rdm", "metadata"}
        for path in paths.values():
            assert path.exists()

        arrays = np.load(paths["arrays"])
        assert arrays["W"].shape == result.W.shape
        assert arrays["A"].shape == result.A.shape
        assert arrays["component_timeseries"].shape == result.component_timeseries.shape
        assert arrays["component_rdms"].shape == result.component_rdms.shape
        assert arrays["target_rdm"].shape == result.target_rdm.shape

        with paths["metadata"].open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        assert metadata["n_conditions"] == result.n_conditions
        assert metadata["n_channels"] == result.n_channels
        assert metadata["n_components"] == result.n_components

        with paths["component_scores"].open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == result.n_components


# =============================================================================
# Test: Visualizations
# =============================================================================

class TestVisualization:
    """Smoke tests for viz.py — every function must create a figure."""

    @pytest.fixture
    def result_with_pvalues(self):
        """Small result WITH permutation test for viz tests."""
        np.random.seed(7)
        C, N, T = 4, 6, 20
        X = np.random.randn(C, N, T)
        target_rdm = np.array([
            [0, 1, 2, 2],
            [1, 0, 2, 2],
            [2, 2, 0, 1],
            [2, 2, 1, 0],
        ], dtype=float)
        return fit_redisca(
            X, target_rdm,
            permutation_test=True, n_perm=30, random_state=0,
        )

    @pytest.fixture
    def result_no_pvalues(self):
        """Small result WITHOUT permutation test for viz tests."""
        np.random.seed(7)
        C, N, T = 4, 6, 20
        X = np.random.randn(C, N, T)
        target_rdm = np.array([
            [0, 1, 2, 2],
            [1, 0, 2, 2],
            [2, 2, 0, 1],
            [2, 2, 1, 0],
        ], dtype=float)
        return fit_redisca(X, target_rdm)

    # -- A) plot_rdm ------------------------------------------------------

    def test_plot_rdm_basic(self, result_no_pvalues):
        import matplotlib
        matplotlib.use("Agg")
        fig, ax = plot_rdm(result_no_pvalues.target_rdm, title="Target")
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_rdm_show_values(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, ax = plot_rdm(result_no_pvalues.target_rdm, show_values=True)
        assert fig is not None
        plt.close(fig)

    def test_plot_rdm_non_square_raises(self):
        import matplotlib; matplotlib.use("Agg")
        with pytest.raises(ValueError, match="square"):
            plot_rdm(np.zeros((3, 4)))

    # -- B) plot_top_component_rdms --------------------------------------

    def test_top_rdms_with_target(self, result_with_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, axes = plot_top_component_rdms(result_with_pvalues, k=2, include_target=True)
        assert len(axes) == 3  # target + 2 components
        plt.close(fig)

    def test_top_rdms_without_target(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, axes = plot_top_component_rdms(result_no_pvalues, k=2, include_target=False)
        assert len(axes) == 2
        plt.close(fig)

    def test_top_rdms_k_exceeds_components(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        r = result_no_pvalues.n_components
        fig, axes = plot_top_component_rdms(result_no_pvalues, k=r + 10, include_target=False)
        assert len(axes) == r
        plt.close(fig)

    def test_top_rdms_order_lambda(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, _ = plot_top_component_rdms(result_no_pvalues, k=2, order="lambda")
        assert fig is not None
        plt.close(fig)

    def test_top_rdms_bad_order(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        with pytest.raises(ValueError, match="order"):
            plot_top_component_rdms(result_no_pvalues, order="bad")

    # -- C) plot_component_scores ----------------------------------------

    def test_scores_basic(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, ax = plot_component_scores(result_no_pvalues)
        assert fig is not None
        plt.close(fig)

    def test_scores_with_p(self, result_with_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, ax = plot_component_scores(result_with_pvalues, show_p=True)
        assert fig is not None
        plt.close(fig)

    def test_scores_show_p_no_pvalues(self, result_no_pvalues):
        """show_p=True but p_values is None — should not crash."""
        import matplotlib; matplotlib.use("Agg")
        fig, ax = plot_component_scores(result_no_pvalues, show_p=True)
        assert fig is not None
        plt.close(fig)

    def test_scores_order_pearson(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, _ = plot_component_scores(result_no_pvalues, order="pearson")
        assert fig is not None
        plt.close(fig)

    # -- D) plot_component_lambdas ---------------------------------------

    def test_lambdas_basic(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, ax = plot_component_lambdas(result_no_pvalues)
        assert fig is not None
        plt.close(fig)

    # -- E) plot_component_timeseries -----------------------------------

    def test_component_timeseries_default(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, axes = plot_component_timeseries(result_no_pvalues)
        assert fig is not None
        assert len(axes) >= 1
        plt.close(fig)

    def test_component_timeseries_with_time_and_names(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        time = np.linspace(-0.1, 0.4, result_no_pvalues.n_timepoints)
        names = [f"Cond {i}" for i in range(result_no_pvalues.n_conditions)]
        fig, axes = plot_component_timeseries(
            result_no_pvalues,
            idxs=[0],
            time=time,
            condition_names=names,
        )
        assert fig is not None
        assert len(axes) == 1
        plt.close(fig)

    def test_component_timeseries_time_unit_ms_and_time_zero(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        time = np.linspace(-0.05, 0.2, result_no_pvalues.n_timepoints)
        fig, axes = plot_component_timeseries(
            result_no_pvalues,
            idxs=[0],
            time=time,
            time_unit="ms",
        )
        assert axes[0].get_xlabel() == "Time (ms)"
        assert any(
            line.get_linestyle() == ":"
            and np.asarray(line.get_xdata()).shape == (2,)
            and np.allclose(line.get_xdata(), [0.0, 0.0])
            for line in axes[0].lines
        )
        plt.close(fig)

    def test_component_timeseries_bad_time_unit(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        time = np.linspace(-0.05, 0.2, result_no_pvalues.n_timepoints)
        with pytest.raises(ValueError, match="time_unit"):
            plot_component_timeseries(result_no_pvalues, time=time, time_unit="minutes")

    def test_component_timeseries_figure_legend(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, axes = plot_component_timeseries(
            result_no_pvalues,
            idxs=[0, 1],
            legend="figure",
        )
        assert len(fig.legends) == 1
        plt.close(fig)

    def test_component_timeseries_separate_conditions(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, axes = plot_component_timeseries(
            result_no_pvalues,
            idxs=[0, 1],
            condition_layout="separate",
        )
        assert axes.shape == (2, result_no_pvalues.n_conditions)
        assert all(len(ax.lines) >= 2 for ax in axes.ravel())
        plt.close(fig)

    def test_component_timeseries_bad_condition_layout(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        with pytest.raises(ValueError, match="condition_layout"):
            plot_component_timeseries(
                result_no_pvalues,
                condition_layout="bad",
            )

    def test_component_timeseries_bad_time_len(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        with pytest.raises(ValueError, match="time must have shape"):
            plot_component_timeseries(result_no_pvalues, time=np.arange(3))

    def test_component_timeseries_bad_condition_names_len(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        with pytest.raises(ValueError, match="condition_names"):
            plot_component_timeseries(result_no_pvalues, condition_names=["a", "b"])

    # -- 2.1) plot_patterns ----------------------------------------------

    def test_patterns_default(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, axes = plot_patterns(result_no_pvalues)
        assert fig is not None
        assert len(axes) <= 3
        plt.close(fig)

    def test_patterns_with_names(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        names = [f"Ch{i}" for i in range(result_no_pvalues.n_channels)]
        fig, axes = plot_patterns(result_no_pvalues, channel_names=names)
        assert fig is not None
        plt.close(fig)

    def test_patterns_custom_idxs(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, axes = plot_patterns(result_no_pvalues, idxs=[0])
        assert len(axes) == 1
        plt.close(fig)

    def test_patterns_bad_mode(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        with pytest.raises(ValueError, match="mode"):
            plot_patterns(result_no_pvalues, mode="topo")

    def test_patterns_wrong_channel_names_len(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        with pytest.raises(ValueError, match="channel_names"):
            plot_patterns(result_no_pvalues, channel_names=["a", "b"])

    # -- new: pearson_mode -----------------------------------------------

    def test_top_rdms_pearson_pos(self, result_no_pvalues):
        """pearson_mode='pos' should pick the highest positive r."""
        import matplotlib; matplotlib.use("Agg")
        fig, _ = plot_top_component_rdms(
            result_no_pvalues, k=2, order="pearson", pearson_mode="pos",
        )
        assert fig is not None
        plt.close(fig)

    def test_top_rdms_pearson_abs(self, result_no_pvalues):
        """pearson_mode='abs' should still work."""
        import matplotlib; matplotlib.use("Agg")
        fig, _ = plot_top_component_rdms(
            result_no_pvalues, k=2, order="pearson", pearson_mode="abs",
        )
        assert fig is not None
        plt.close(fig)

    def test_top_rdms_bad_pearson_mode(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        with pytest.raises(ValueError, match="pearson_mode"):
            plot_top_component_rdms(
                result_no_pvalues, order="pearson", pearson_mode="bad",
            )

    def test_scores_pearson_pos(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, _ = plot_component_scores(
            result_no_pvalues, order="pearson", pearson_mode="pos",
        )
        assert fig is not None
        plt.close(fig)

    def test_scores_pearson_abs(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, _ = plot_component_scores(
            result_no_pvalues, order="pearson", pearson_mode="abs",
        )
        assert fig is not None
        plt.close(fig)

    # -- new: normalize_rdms ---------------------------------------------

    def test_top_rdms_no_normalize(self, result_no_pvalues):
        """normalize_rdms=False should show raw values."""
        import matplotlib; matplotlib.use("Agg")
        fig, _ = plot_top_component_rdms(
            result_no_pvalues, k=1, normalize_rdms=False,
        )
        assert fig is not None
        plt.close(fig)

    def test_top_rdms_shared_colorbar(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, _ = plot_top_component_rdms(
            result_no_pvalues, k=2, include_target=True, shared_colorbar=True,
        )
        assert len(fig.axes) == 4  # 3 panels + 1 shared colorbar
        plt.close(fig)

    def test_top_rdms_individual_colorbars(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, _ = plot_top_component_rdms(
            result_no_pvalues, k=2, include_target=True, shared_colorbar=False,
        )
        assert len(fig.axes) == 6  # 3 panels + 3 per-panel colorbars
        plt.close(fig)

    # -- new: pattern normalize ------------------------------------------

    def test_patterns_normalize_none(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, _ = plot_patterns(result_no_pvalues, normalize="none")
        assert fig is not None
        plt.close(fig)

    def test_patterns_normalize_maxabs(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, _ = plot_patterns(result_no_pvalues, normalize="maxabs")
        assert fig is not None
        plt.close(fig)

    def test_patterns_normalize_zscore(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        fig, _ = plot_patterns(result_no_pvalues, normalize="zscore")
        assert fig is not None
        plt.close(fig)

    def test_patterns_bad_normalize(self, result_no_pvalues):
        import matplotlib; matplotlib.use("Agg")
        with pytest.raises(ValueError, match="normalize"):
            plot_patterns(result_no_pvalues, normalize="bad")


class TestMNEVisualization:
    """Tests for the optional MNE-based visualization helpers."""

    def test_pattern_topomaps_requires_mne(self, result_no_pvalues, monkeypatch):
        def _raise():
            raise ImportError("MNE missing")

        monkeypatch.setattr("redisca.viz_mne._require_mne", _raise)
        info = types.SimpleNamespace(ch_names=[f"Ch{i}" for i in range(result_no_pvalues.n_channels)])

        with pytest.raises(ImportError, match="MNE missing"):
            plot_pattern_topomaps(result_no_pvalues, info)

    def test_pattern_topomaps_with_fake_mne(self, result_no_pvalues, monkeypatch):
        import matplotlib; matplotlib.use("Agg")

        calls = []

        def fake_plot_topomap(data, info, **kwargs):
            calls.append(
                {
                    "data": np.asarray(data),
                    "info": info,
                    "kwargs": kwargs,
                }
            )
            image = kwargs["axes"].imshow(np.array([[0.0, 1.0], [1.0, 0.0]]), cmap=kwargs["cmap"])
            return image, None

        fake_mne = types.SimpleNamespace(
            viz=types.SimpleNamespace(
                plot_topomap=fake_plot_topomap,
            )
        )
        monkeypatch.setattr("redisca.viz_mne._require_mne", lambda: fake_mne)

        info = types.SimpleNamespace(ch_names=[f"Ch{i}" for i in range(result_no_pvalues.n_channels)])
        fig, axes = plot_pattern_topomaps(result_no_pvalues, info, idxs=[0, 1], vlim="joint")

        assert len(axes) == 2
        assert len(calls) == 2
        assert calls[0]["kwargs"]["vlim"] == calls[1]["kwargs"]["vlim"]
        assert np.isclose(abs(calls[0]["kwargs"]["vlim"][0]), calls[0]["kwargs"]["vlim"][1])
        plt.close(fig)

    def test_pattern_topomaps_info_channel_mismatch_raises(self, result_no_pvalues, monkeypatch):
        fake_mne = types.SimpleNamespace(
            viz=types.SimpleNamespace(
                plot_topomap=lambda *args, **kwargs: (kwargs["axes"].imshow(np.zeros((2, 2))), None),
            )
        )
        monkeypatch.setattr("redisca.viz_mne._require_mne", lambda: fake_mne)

        info = types.SimpleNamespace(ch_names=["Ch0", "Ch1"])
        with pytest.raises(ValueError, match="channel count"):
            plot_pattern_topomaps(result_no_pvalues, info)

    def test_plot_compare_conditions_calls_mne(self, monkeypatch):
        captured = {}

        def fake_compare(evokeds, **kwargs):
            captured["evokeds"] = evokeds
            captured["kwargs"] = kwargs
            return "compare-figure"

        fake_mne = types.SimpleNamespace(
            viz=types.SimpleNamespace(plot_compare_evokeds=fake_compare)
        )
        monkeypatch.setattr("redisca.viz_mne._require_mne", lambda: fake_mne)

        out = plot_compare_conditions({"A": object()}, title="Demo")
        assert out == "compare-figure"
        assert captured["kwargs"]["show"] is False
        assert captured["kwargs"]["title"] == "Demo"

    def test_plot_condition_joint_calls_mne(self, monkeypatch):
        captured = {}

        def fake_joint(evoked, **kwargs):
            captured["evoked"] = evoked
            captured["kwargs"] = kwargs
            return "joint-figure"

        fake_mne = types.SimpleNamespace(
            viz=types.SimpleNamespace(plot_evoked_joint=fake_joint)
        )
        monkeypatch.setattr("redisca.viz_mne._require_mne", lambda: fake_mne)

        evoked = object()
        out = plot_condition_joint(evoked, title="Joint")
        assert out == "joint-figure"
        assert captured["evoked"] is evoked
        assert captured["kwargs"]["show"] is False
        assert captured["kwargs"]["title"] == "Joint"


class TestPackageImport:
    """Tests for lightweight package import behavior."""

    def test_root_import_does_not_import_matplotlib(self):
        """`import redisca` should not eagerly import matplotlib."""
        import subprocess
        import sys
        from pathlib import Path

        src_dir = Path(__file__).resolve().parents[1] / "src"
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(src_dir)!r})\n"
            "import redisca\n"
            "print('matplotlib' in sys.modules)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )

        assert result.stdout.strip() == "False"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
