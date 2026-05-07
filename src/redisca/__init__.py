"""ReDisCA — Representational Dissimilarity Component Analysis.

A library for EEG/MEG data analysis using the ReDisCA method,
which finds spatial components whose dissimilarity structure
matches a given theoretical RDM.

Core fitting, simple RDM construction, sliding-window helpers, and lightweight
MNE interoperability helpers are available from the package root. Plotting and
batch-report helpers live in the ``redisca.viz``, ``redisca.viz_mne``, and
``redisca.report`` submodules.
"""

__version__ = "0.1.0"

from .fit import fit_redisca
from .core import compute_pearson_scores
from .export import export_result
from .mne_utils import (
    average_conditions,
    condition_epoch_counts,
    evokeds_to_tensor,
    fit_redisca_evokeds,
    make_montage_from_electrodes,
    sliding_window_fit_redisca_evokeds,
)
from .rdm import binary_rdm
from .stats import permutation_test_redisca
from .types import (
    EvokedReDisCAResult,
    EvokedSlidingWindowReDisCAResult,
    ReDisCAResult,
    SlidingWindowReDisCAResult,
    ValidationResult,
    PermutationTestResult,
)
from .validation import validate_inputs
from .windowed import (
    best_window_index,
    ms_to_samples,
    sliding_window_fit_redisca,
    sliding_window_fit_redisca_ms,
)

__all__ = [
    "__version__",
    "fit_redisca",
    "binary_rdm",
    "average_conditions",
    "condition_epoch_counts",
    "EvokedReDisCAResult",
    "EvokedSlidingWindowReDisCAResult",
    "evokeds_to_tensor",
    "fit_redisca_evokeds",
    "make_montage_from_electrodes",
    "sliding_window_fit_redisca_evokeds",
    "sliding_window_fit_redisca",
    "sliding_window_fit_redisca_ms",
    "best_window_index",
    "ms_to_samples",
    "compute_pearson_scores",
    "export_result",
    "permutation_test_redisca",
    "PermutationTestResult",
    "ReDisCAResult",
    "SlidingWindowReDisCAResult",
    "ValidationResult",
    "validate_inputs",
]
