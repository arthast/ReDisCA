"""ReDisCA — Representational Dissimilarity Component Analysis.

A library for EEG/MEG data analysis using the ReDisCA method,
which finds spatial components whose dissimilarity structure
matches a given theoretical RDM.

Numerical functionality is available from the package root.
Visualization helpers live in the ``redisca.viz`` submodule.
"""

__version__ = "0.1.0"

from .fit import fit_redisca
from .core import compute_pearson_scores
from .export import export_result
from .stats import permutation_test_redisca
from .types import ReDisCAResult, ValidationResult, PermutationTestResult
from .validation import validate_inputs

__all__ = [
    "__version__",
    "fit_redisca",
    "compute_pearson_scores",
    "export_result",
    "permutation_test_redisca",
    "PermutationTestResult",
    "ReDisCAResult",
    "ValidationResult",
    "validate_inputs",
]
