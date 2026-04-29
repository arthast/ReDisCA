# ReDisCA
Realization of algorithm Representational dissimilarity component analysis

## Installation

```bash
pip install -e .
```

Optional MNE-based sensor/topomap visualizations:

```bash
pip install -e ".[mne]"
```

## Quick start

```python
import numpy as np
from redisca import export_result, fit_redisca
from redisca.viz import plot_component_timeseries

X = np.random.randn(4, 16, 200)            # (C, N, T)
target_rdm = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 0],
], dtype=float)

result = fit_redisca(X, target_rdm, permutation_test=True, n_perm=500, random_state=0)
fig, axes = plot_component_timeseries(result, idxs=[0])
paths = export_result(result, "redisca_output")

print(result.pearson_scores)
print(result.p_values)
print(paths)
```

Sliding-window scans are available through `sliding_window_fit_redisca`:

```python
from redisca import sliding_window_fit_redisca

scan = sliding_window_fit_redisca(
    X,
    target_rdm,
    window_size=50,
    step_size=10,
)

pearson_over_time = scan.component_metric_matrix("pearson_scores")
```

## Visualization example

A ready-made script that generates synthetic data, fits the model, and
displays diagnostic plots:

```bash
python3 examples/visualize_synthetic.py
```

The script produces:

1. **Top component RDMs** + the target RDM (`plot_top_component_rdms`)
2. **Pearson correlation scores** per component (`plot_component_scores`)
3. **Eigenvalue (λ) spectrum** (`plot_component_lambdas`)
4. **Component time series** by condition (`plot_component_timeseries`)
5. **Spatial pattern weights** per channel for the top 3 components (`plot_patterns`)
6. **Export bundle** with arrays, summary CSV, and metadata JSON (`export_result`)

All visualization functions return `(fig, ax)` / `(fig, axes)` and can be
used standalone:

```python
from redisca import export_result
from redisca.viz import (
    plot_top_component_rdms,
    plot_component_scores,
    plot_component_timeseries,
)

fig, axes = plot_top_component_rdms(result, k=3)
fig, ax   = plot_component_scores(result, show_p=True)
fig, axes = plot_component_timeseries(result, idxs=[0, 1])
paths = export_result(result, "redisca_output")
```

## Visualization layers

`redisca.viz` contains lightweight Matplotlib plots that do not require
sensor geometry:

- RDM heatmaps
- Pearson score bars
- eigenvalue spectra
- component time courses
- fallback bar-plots for patterns

If you want EEG/MEG-aware sensor topographies and standard MNE condition
plots, use the optional `redisca.viz_mne` module:

```python
from redisca.viz_mne import (
    plot_pattern_topomaps,
    plot_compare_conditions,
    plot_condition_joint,
)

# Requires MNE and an aligned mne.Info object:
fig, axes = plot_pattern_topomaps(result, info)
```

`plot_patterns()` remains useful as a fallback when you do not have channel
locations, while `plot_pattern_topomaps()` is the preferred final
visualization for EEG/MEG figures.

## Article-style reproduction

The repository now includes two reproducible workflows inspired by the
ReDisCA paper.

### 1. Open-data EEG example (ERP CORE N170)

This script downloads ERP CORE `sub-001` from the public OSF dataset, builds
four averaged conditions (`face`, `car`, `scrambled_face`,
`scrambled_car`), runs a 150 ms meaningful-vs-meaningless sliding-window
scan, and then runs a fixed N170-centered face-specific analysis.

Requires the optional MNE dependency:

```bash
pip install -e ".[mne]"
python3 examples/reproduce_erpcore_n170.py --n-perm 50
```

Outputs are saved under `examples/repro_outputs/erpcore_n170/` and include:

- sliding-window Pearson and p-value heatmaps
- top component RDMs and time courses for the best meaningfulness window
- sensor topomaps for the recovered patterns
- an export bundle plus `summary.json`

This is an article-style public-data workflow, not a byte-for-byte
reconstruction of every published figure.

### 2. Paper-like simulation benchmark

This benchmark script runs Monte Carlo simulations in the spirit of the paper:
a single-source detection scenario and a multi-source recovery scenario with
source-specific RDMs.

```bash
python3 examples/paper_like_benchmark.py --n-iter 30
```

Outputs are saved under `examples/repro_outputs/paper_like_benchmark/` and
include:

- `single_source_trials.csv`
- `multi_source_trials.csv`
- `benchmark_overview.png`
- `summary.json`

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
