# ReDisCA
Realization of algorithm Representational dissimilarity component analysis

## Installation

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from redisca import fit_redisca

X = np.random.randn(4, 16, 200)            # (C, N, T)
target_rdm = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 0],
], dtype=float)

result = fit_redisca(X, target_rdm, permutation_test=True, n_perm=500, random_state=0)
print(result.pearson_scores)
print(result.p_values)
```

## Visualization example

A ready-made script that generates synthetic data, fits the model, and
displays four diagnostic plots:

```bash
python examples/visualize_synthetic.py
```

The script produces:

1. **Top component RDMs** + the target RDM (`plot_top_component_rdms`)
2. **Pearson correlation scores** per component (`plot_component_scores`)
3. **Eigenvalue (λ) spectrum** (`plot_component_lambdas`)
4. **Spatial pattern weights** per channel for the top 3 components (`plot_patterns`)

All visualization functions return `(fig, ax)` / `(fig, axes)` and can be
used standalone:

```python
from redisca.viz import plot_top_component_rdms, plot_component_scores

fig, axes = plot_top_component_rdms(result, k=3)
fig, ax   = plot_component_scores(result, show_p=True)
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

