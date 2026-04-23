#!/usr/bin/env python
"""Visualise ReDisCA results on synthetic data.

Run from the repository root:

    python examples/visualize_synthetic.py

Figures are saved to ``examples/figures/``.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from redisca import export_result, fit_redisca
from redisca.viz import (
    plot_top_component_rdms,
    plot_component_scores,
    plot_component_lambdas,
    plot_component_timeseries,
    plot_patterns,
)

FIGURES_DIR = Path(__file__).parent / "figures"
EXPORT_DIR = Path(__file__).parent / "output"


def make_synthetic_data(
    C: int = 5,
    N: int = 16,
    T: int = 300,
    seed: int = 42,
):
    """Generate synthetic EEG-like data with one target-aligned source.

    The data contain several latent sources mixed through random spatial
    patterns.  Only **one** source carries the condition structure that
    matches the target RDM; the remaining sources act as structured
    nuisance (different condition contrasts) or unstructured noise.

    This means ReDisCA should find 1–2 components with high Pearson
    correlation to the target, while the rest show noticeably lower scores
    and a clear drop in the eigenvalue spectrum.

    Condition design (C = 5):
        Groups: {0, 1} vs {2, 3, 4}
        target RDM: within-group distance = 0, between-group = 1

    Source design:
        s0  — target source: amplitude +1 for conds 0,1 and −1 for 2,3,4
        s1  — nuisance source: different contrast (odd vs even conditions)
        s2  — nuisance source: random per-condition amplitude
        s3..s5 — pure noise (identical across conditions)
    """
    rng = np.random.default_rng(seed)

    n_sources = 6

    # --- spatial mixing matrix A_mix (N, n_sources) ----------------------
    A_mix = rng.standard_normal((N, n_sources))

    # --- temporal waveforms per source (n_sources, T) --------------------
    S = rng.standard_normal((n_sources, T))

    # --- per-condition amplitude modulation (C, n_sources) ---------------
    #  source 0: encodes the target structure
    amp = np.zeros((C, n_sources))
    amp[:, 0] = [1.0, 1.0, -1.0, -1.0, -1.0]   # target contrast
    amp[:, 1] = [1.0, -1.0, 1.0, -1.0, 0.5]     # nuisance contrast
    amp[:, 2] = rng.standard_normal(C)            # random contrast
    amp[:, 3:] = 1.0                              # noise — same for all

    # --- strength of each source -----------------------------------------
    source_scale = np.array([3.0, 2.0, 1.5, 4.0, 4.0, 4.0])

    # --- assemble X (C, N, T) -------------------------------------------
    X = np.zeros((C, N, T))
    for c in range(C):
        for s_idx in range(n_sources):
            X[c] += source_scale[s_idx] * amp[c, s_idx] * (
                A_mix[:, s_idx:s_idx + 1] @ S[s_idx:s_idx + 1, :]
            )
        # add sensor noise
        X[c] += 0.5 * rng.standard_normal((N, T))

    # --- target RDM: {0,1} vs {2,3,4} -----------------------------------
    target_rdm = np.array([
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ], dtype=float)

    return X, target_rdm


def main() -> None:
    # --- data -----------------------------------------------------------
    X, target_rdm = make_synthetic_data()
    channel_names = [f"Ch{i}" for i in range(X.shape[1])]

    # --- fit ------------------------------------------------------------
    result = fit_redisca(
        X,
        target_rdm,
        permutation_test=True,
        n_perm=500,
        random_state=0,
    )

    np.set_printoptions(precision=6, suppress=False)
    print("=== ReDisCA fit complete ===")
    print(f"  Components : {result.n_components}")
    print(f"  Lambdas    : {result.lambdas}")
    print(f"  Pearson    : {result.pearson_scores}")
    if result.p_values is not None:
        print(f"  p-values   : {result.p_values}")
        print(f"  Significant: {result.significant}")

    # --- 1. Top component RDMs + target ---------------------------------
    plot_top_component_rdms(
        result, k=3, order="pearson", include_target=True,
        show_values=True,
        save_path=FIGURES_DIR / "top_component_rdms.png",
    )

    # --- 2. Pearson scores (with significance stars) --------------------
    plot_component_scores(
        result, order="lambda", show_p=True,
        save_path=FIGURES_DIR / "pearson_scores.png",
    )

    # --- 3. Eigenvalue spectrum -----------------------------------------
    plot_component_lambdas(
        result,
        save_path=FIGURES_DIR / "lambdas.png",
    )

    # --- 4. Component time series for top 3 components ------------------
    plot_component_timeseries(
        result, idxs=[0, 1, 2],
        save_path=FIGURES_DIR / "component_timeseries.png",
    )

    # --- 5. Spatial patterns for top 3 components -----------------------
    plot_patterns(
        result, idxs=[0, 1, 2], channel_names=channel_names,
        save_path=FIGURES_DIR / "patterns.png",
    )

    exported = export_result(result, EXPORT_DIR)

    print(f"\nAll figures saved to {FIGURES_DIR.resolve()}")
    print("Exported bundle:")
    for name, path in exported.items():
        print(f"  {name:>16}: {path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
