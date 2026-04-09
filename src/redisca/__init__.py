"""ReDisCA — Representational Dissimilarity Component Analysis.

A library for EEG/MEG data analysis using the ReDisCA method,
which finds spatial components whose dissimilarity structure
matches a given theoretical RDM.
"""

__version__ = "0.1.0"

from .fit import fit_redisca
from .core import compute_pearson_scores
from .stats import permutation_test_redisca
from .types import ReDisCAResult, ValidationResult, PermutationTestResult
from .validation import validate_inputs
from .viz import (
    plot_rdm,
    plot_top_component_rdms,
    plot_component_scores,
    plot_component_lambdas,
    plot_patterns,
)

__all__ = [
    "__version__",
    "fit_redisca",
    "compute_pearson_scores",
    "permutation_test_redisca",
    "PermutationTestResult",
    "ReDisCAResult",
    "ValidationResult",
    "validate_inputs",
    "plot_rdm",
    "plot_top_component_rdms",
    "plot_component_scores",
    "plot_component_lambdas",
    "plot_patterns",
]
