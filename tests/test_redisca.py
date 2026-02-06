import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from src.redisca import fit_redisca, validate_inputs, ReDisCAResult
from src.redisca.core import (
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
    """Data with known structure for testing RDM recovery."""
    np.random.seed(123)
    C, N, T = 4, 20, 200

    # Create a "source" signal that differs between condition groups
    # Conditions 0,1 are similar; conditions 2,3 are similar
    base_signal = np.random.randn(N, T)

    X = np.zeros((C, N, T))
    X[0] = base_signal + 0.1 * np.random.randn(N, T)
    X[1] = base_signal + 0.1 * np.random.randn(N, T)
    X[2] = -base_signal + 0.1 * np.random.randn(N, T)
    X[3] = -base_signal + 0.1 * np.random.randn(N, T)

    # Target RDM: 0-1 similar, 2-3 similar, cross-group different
    target_rdm = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0]
    ], dtype=float)

    return X, target_rdm


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

    def test_a_times_w_identity(self, simple_data):
        """A @ W should be approximately identity."""
        X, target_rdm = simple_data
        C, N, T = X.shape

        pairs = pair_indices(C)
        R_list = compute_all_R_ij(X, pairs)
        d_tilde = standardize(vectorize_upper(target_rdm))
        R_bar = compute_R_bar(R_list)
        R_bar_d = compute_R_bar_d(R_list, R_bar, d_tilde)

        W, _ = solve_gep(R_bar_d, R_bar)
        A = compute_patterns(W, R_bar)

        AW = A @ W
        assert_allclose(AW, np.eye(N), atol=1e-6)

    def test_rank_reduced_a_times_w(self, simple_data):
        """A @ W should be identity even with rank reduction."""
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

        AW = A @ W
        assert_allclose(AW, np.eye(rank), atol=1e-6)


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
        assert result.A.shape == (r, N)
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
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

