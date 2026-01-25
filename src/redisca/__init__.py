"""ReDisCA — Representational Dissimilarity Component Analysis.

A library for EEG/MEG data analysis using the ReDisCA method,
which finds spatial components whose dissimilarity structure
matches a given theoretical RDM.
"""

__version__ = "0.1.0"

from .types import ReDisCAResult, ValidationResult
from .validation import validate_inputs

__all__ = [
    "__version__",
    "ReDisCAResult",
    "ValidationResult",
    "validate_inputs",
]
