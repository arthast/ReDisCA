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

## Visualization example

A ready-made script that generates synthetic data, fits the model, and
displays diagnostic plots:

```bash
python examples/visualize_synthetic.py
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

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
