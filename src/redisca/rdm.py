"""Helpers for constructing target representational dissimilarity matrices."""

from __future__ import annotations

from collections.abc import Sequence, Set

import numpy as np
from numpy.typing import NDArray


def binary_rdm(
    condition_order: Sequence[str],
    positive_conditions: Set[str],
) -> NDArray[np.float64]:
    """Create a binary detector-style RDM from named conditions.

    Entries are ``1`` when exactly one condition in the pair belongs to
    ``positive_conditions`` and ``0`` otherwise. This is useful for simple
    "one group vs another group" hypotheses such as meaningful vs meaningless
    or face vs all other conditions.
    """
    if len(condition_order) == 0:
        raise ValueError("condition_order must contain at least one condition")

    missing = set(positive_conditions).difference(condition_order)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "positive_conditions contains names absent from condition_order: "
            f"{missing_str}"
        )

    C = len(condition_order)
    is_positive = np.array(
        [condition in positive_conditions for condition in condition_order],
        dtype=bool,
    )

    rdm = np.zeros((C, C), dtype=np.float64)
    for i in range(C):
        for j in range(i + 1, C):
            rdm[i, j] = float(is_positive[i] != is_positive[j])
            rdm[j, i] = rdm[i, j]
    return rdm


__all__ = [
    "binary_rdm",
]
